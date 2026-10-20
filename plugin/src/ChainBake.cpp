#include "ChainBake.h"

#include "json.hpp"
#include <regex>

#include "NAM/get_dsp.h"
#include "NamEngine.h"
#include "Processor.h"

namespace t3k::bake {

namespace {

constexpr double kRate = 48000.0;
constexpr int kBlock = 512;

// Overall progress budget: the render is quick, the trainer is the long pole.
constexpr float kRenderEnd = 0.12f;
constexpr float kTrainEnd = 0.95f;

// The load wait gives up after this long; cached models load in well under a
// second, an uncached tone needs the network.
constexpr int kLoadTimeoutMs = 90000;
constexpr int kLoadPollMs = 100;

juce::File appDataRoot() {
  juce::File base = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
#if JUCE_MAC
  base = base.getChildFile("Application Support");
#endif
  return base.getChildFile("TONE3000");
}

}  // namespace

// ---------------------------------------------------------------------------
// Helpers

std::vector<juce::File> trainingInputSearchPaths() {
  std::vector<juce::File> paths;
  if (const auto env = juce::SystemStats::getEnvironmentVariable("T3K_TRAINING_INPUT", {});
      env.isNotEmpty())
    paths.emplace_back(env);
  paths.push_back(appDataRoot().getChildFile("Training").getChildFile(kTrainingInputName));
  paths.push_back(juce::File::getSpecialLocation(juce::File::commonApplicationDataDirectory)
#if JUCE_MAC
                      .getChildFile("Application Support")
#endif
                      .getChildFile("TONE3000")
                      .getChildFile("Training")
                      .getChildFile(kTrainingInputName));
  // Running from a source checkout (standalone dev builds, tests).
  auto dir = juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory();
  for (int i = 0; i < 8 && dir.exists(); ++i, dir = dir.getParentDirectory())
    paths.push_back(dir.getChildFile("resources").getChildFile("training").getChildFile(kTrainingInputName));
  return paths;
}

juce::File findTrainingInput() {
  for (const auto& f : trainingInputSearchPaths())
    if (f.existsAsFile())
      return f;
  return {};
}

std::vector<float> readMono48k(const juce::File& file, juce::String& error) {
  juce::AudioFormatManager formats;
  formats.registerBasicFormats();
  std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
  if (reader == nullptr || reader->lengthInSamples <= 0) {
    error = "Couldn't read " + file.getFileName();
    return {};
  }
  const auto frames = static_cast<int>(reader->lengthInSamples);
  const int channels = static_cast<int>(juce::jmax(1u, reader->numChannels));
  juce::AudioBuffer<float> buffer(channels, frames);
  reader->read(&buffer, 0, frames, 0, true, true);

  std::vector<float> mono(static_cast<size_t>(frames), 0.0f);
  for (int c = 0; c < channels; ++c) {
    const float* src = buffer.getReadPointer(c);
    for (int i = 0; i < frames; ++i)
      mono[static_cast<size_t>(i)] += src[i] / static_cast<float>(channels);
  }
  if (std::abs(reader->sampleRate - kRate) < 0.5)
    return mono;

  const double ratio = reader->sampleRate / kRate;
  const auto outFrames = static_cast<int>(static_cast<double>(frames) / ratio);
  std::vector<float> resampled(static_cast<size_t>(outFrames), 0.0f);
  juce::LagrangeInterpolator interpolator;
  interpolator.process(ratio, mono.data(), resampled.data(), outFrames);
  return resampled;
}

bool writeMono48kWav24(const juce::File& file, const std::vector<float>& samples) {
  file.deleteFile();
  std::unique_ptr<juce::FileOutputStream> stream(file.createOutputStream());
  if (stream == nullptr)
    return false;
  juce::WavAudioFormat wav;
  std::unique_ptr<juce::AudioFormatWriter> writer(
      wav.createWriterFor(stream.get(), kRate, 1, 24, {}, 0));
  if (writer == nullptr)
    return false;
  stream.release();  // the writer owns it now

  juce::AudioBuffer<float> chunk(1, 1 << 16);
  for (size_t off = 0; off < samples.size(); off += static_cast<size_t>(chunk.getNumSamples())) {
    const int n = static_cast<int>(
        std::min(samples.size() - off, static_cast<size_t>(chunk.getNumSamples())));
    for (int i = 0; i < n; ++i)
      chunk.setSample(0, i, juce::jlimit(-1.0f, 1.0f, samples[off + static_cast<size_t>(i)]));
    if (!writer->writeFromAudioSampleBuffer(chunk, 0, n))
      return false;
  }
  return true;
}

double errorToSignalRatio(const std::vector<float>& reference, const std::vector<float>& test,
                          size_t skip, bool fitGain) {
  const size_t n = std::min(reference.size(), test.size());
  if (skip >= n)
    return -1.0;
  double gain = 1.0;
  if (fitGain) {
    double xy = 0.0, yy = 0.0;
    for (size_t i = skip; i < n; ++i) {
      xy += static_cast<double>(reference[i]) * test[i];
      yy += static_cast<double>(test[i]) * test[i];
    }
    gain = yy > 0.0 ? xy / yy : 1.0;
  }
  double err = 0.0, sig = 0.0;
  for (size_t i = skip; i < n; ++i) {
    const double d = static_cast<double>(reference[i]) - gain * test[i];
    err += d * d;
    sig += static_cast<double>(reference[i]) * reference[i];
  }
  return sig > 0.0 ? err / sig : -1.0;
}

std::vector<juce::StringArray> pythonCandidates() {
  std::vector<juce::StringArray> out;
  if (const auto env = juce::SystemStats::getEnvironmentVariable("T3K_PYTHON", {}); env.isNotEmpty())
    out.push_back(juce::StringArray(env));
#if JUCE_WINDOWS
  out.push_back({"py", "-3"});
  out.push_back({"python"});
  out.push_back({"python3"});
#else
  out.push_back({"python3"});
  out.push_back({"python"});
#endif
  return out;
}

juce::StringArray findTrainerPython() {
  for (auto argv : pythonCandidates()) {
    // find_spec doesn't import torch, so the probe answers in milliseconds.
    argv.add("-c");
    argv.add("import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('nam') else 1)");
    juce::ChildProcess probe;
    if (!probe.start(argv, 0))
      continue;
    if (probe.waitForProcessToFinish(15000) && probe.getExitCode() == 0) {
      argv.removeRange(argv.size() - 2, 2);
      return argv;
    }
    if (probe.isRunning())
      probe.kill();
  }
  return {};
}

const char* trainerScript() {
  return R"PY(# Written by the TONE3000 plugin (chain bake). Trains one .nam from an
# input/output pair rendered through the plugin's chain.
import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--name", required=True)
p.add_argument("--log", required=True)
p.add_argument("--epochs", type=int, default=100)
a = p.parse_args()

# The plugin tails this file; keeping the pipes out of it means a cancel can
# never deadlock on a full pipe.
log = open(a.log, "w", buffering=1, encoding="utf-8")
sys.stdout = sys.stderr = log

import matplotlib
matplotlib.use("Agg")

from nam.models.metadata import UserMetadata
from nam.train import core

meta = UserMetadata(name=a.name, modeled_by="TONE3000 plugin (baked chain)")
out = core.train(
    a.input,
    a.output,
    a.outdir,
    epochs=a.epochs,
    silent=True,
    modelname="baked",
    ignore_checks=True,
    user_metadata=meta,
)
if out.model is None:
    print("T3K_FAILED training produced no model")
    sys.exit(2)
out.model.net.export(
    a.outdir,
    basename="baked",
    user_metadata=meta,
    other_metadata={"training": out.metadata.model_dump(mode="json")},
)
if out.metadata.validation_esr is not None:
    print("T3K_ESR", out.metadata.validation_esr)
print("T3K_DONE")
)PY";
}

int parseEpoch(const juce::String& text) {
  static const std::regex re(R"(Epoch\s+(\d+))");
  const std::string s = text.toStdString();
  int last = -1;
  for (auto it = std::sregex_iterator(s.begin(), s.end(), re); it != std::sregex_iterator(); ++it)
    last = std::stoi((*it)[1].str());
  return last;
}

bool parseEsr(const juce::String& text, double& esr) {
  static const std::regex re(R"(T3K_ESR\s+([0-9eE+.\-]+))");
  const std::string s = text.toStdString();
  std::smatch m;
  if (!std::regex_search(s, m, re))
    return false;
  esr = std::stod(m[1].str());
  return true;
}

// ---------------------------------------------------------------------------
// Job

// State shared between the message thread and the worker. The final progress
// post releases the processor on the message thread.
struct ChainBake::Shared : std::enable_shared_from_this<Shared> {
  ProgressFn onProgress;
  std::atomic<bool> alive{true};
  std::atomic<bool> cancel{false};
  std::atomic<bool>* running = nullptr;
  std::unique_ptr<TONE3000Processor> processor;

  void post(Progress p) {
    const bool final = p.phase == Progress::Phase::done || p.phase == Progress::Phase::failed ||
                       p.phase == Progress::Phase::cancelled;
    juce::MessageManager::callAsync([self = shared_from_this(), p = std::move(p), final] {
      if (!self->alive.load())
        return;
      if (final) {
        self->processor.reset();
        self->running->store(false);
      }
      if (self->onProgress)
        self->onProgress(p);
    });
  }
};

class ChainBake::Job : public juce::Thread {
public:
  Job(std::shared_ptr<Shared> s, Settings st)
      : juce::Thread("ChainBake"), shared(std::move(s)), settings(std::move(st)) {}
  ~Job() override { stopThread(15000); }

  juce::StringArray python;
  juce::File sweep;

  void run() override {
    try {
      bake();
    } catch (const std::exception& e) {
      fail(juce::String("Unexpected error: ") + e.what());
    }
  }

private:
  std::shared_ptr<Shared> shared;
  Settings settings;

  void report(Progress::Phase phase, float fraction, const juce::String& message) {
    Progress p;
    p.phase = phase;
    p.fraction = fraction;
    p.message = message;
    shared->post(std::move(p));
  }
  void fail(const juce::String& why) { report(Progress::Phase::failed, 0.0f, why); }
  bool cancelled() const { return shared->cancel.load() || threadShouldExit(); }
  void cancelOut() { report(Progress::Phase::cancelled, 0.0f, "Cancelled"); }

  void bake() {
    juce::String err;
    report(Progress::Phase::rendering, 0.0f, "Rendering the chain…");

    auto input = readMono48k(sweep, err);
    if (input.empty())
      return fail(err);

    // Render through the private processor.
    std::vector<float> output(input.size(), 0.0f);
    {
      auto& proc = *shared->processor;
      const int latency = proc.getLatencySamples();
      juce::AudioBuffer<float> buffer(2, kBlock);
      juce::MidiBuffer midi;
      const size_t total = input.size() + static_cast<size_t>(latency);
      for (size_t off = 0; off < total; off += kBlock) {
        if (cancelled())
          return cancelOut();
        buffer.clear();
        for (int i = 0; i < kBlock; ++i) {
          const size_t pos = off + static_cast<size_t>(i);
          if (pos < input.size())
            buffer.setSample(0, i, input[pos]);
        }
        proc.processBlock(buffer, midi);
        for (int i = 0; i < kBlock; ++i) {
          const size_t pos = off + static_cast<size_t>(i);
          if (pos < static_cast<size_t>(latency) || pos - static_cast<size_t>(latency) >= input.size())
            continue;
          output[pos - static_cast<size_t>(latency)] =
              0.5f * (buffer.getSample(0, i) + buffer.getSample(1, i));
        }
        if ((off / kBlock) % 64 == 0)
          report(Progress::Phase::rendering,
                 kRenderEnd * static_cast<float>(off) / static_cast<float>(total), "Rendering the chain…");
      }
    }

    // The trainer normalizes the output level itself, so scaling a hot render
    // back under full scale loses nothing; clipping it would.
    float peak = 0.0f;
    for (float s : output)
      peak = std::max(peak, std::abs(s));
    if (peak > 0.98f)
      for (auto& s : output)
        s *= 0.98f / peak;
    if (peak < 1.0e-4f)
      return fail("The chain produced silence. Check that every block has loaded and is enabled.");

    const auto work = juce::File::getSpecialLocation(juce::File::tempDirectory)
                          .getChildFile("T3K-bake-" + juce::String::toHexString(juce::Time::currentTimeMillis()));
    if (!work.createDirectory())
      return fail("Couldn't create a temporary folder.");
    struct Cleanup {
      juce::File dir;
      ~Cleanup() { dir.deleteRecursively(); }
    } cleanup{work};

    const auto inWav = work.getChildFile("input.wav");
    const auto outWav = work.getChildFile("output.wav");
    // The trainer recognizes the standard input by its content, so train on
    // the sweep file itself rather than a re-encoded copy.
    if (!sweep.copyFileTo(inWav) || !writeMono48kWav24(outWav, output))
      return fail("Couldn't write the training files.");

    // Train.
    const juce::String name = settings.modelName.isNotEmpty()
                                  ? settings.modelName
                                  : settings.destination.getFileNameWithoutExtension();
    const auto scriptFile = work.getChildFile("t3k_bake_train.py");
    const auto logFile = work.getChildFile("train.log");
    scriptFile.replaceWithText(trainerScript(), false, false, "\n");

    juce::StringArray argv = python;
    argv.add("-u");
    argv.add(scriptFile.getFullPathName());
    argv.add("--input");
    argv.add(inWav.getFullPathName());
    argv.add("--output");
    argv.add(outWav.getFullPathName());
    argv.add("--outdir");
    argv.add(work.getFullPathName());
    argv.add("--name");
    argv.add(name);
    argv.add("--log");
    argv.add(logFile.getFullPathName());
    argv.add("--epochs");
    argv.add(juce::String(settings.epochs));

    report(Progress::Phase::training, kRenderEnd, "Starting the trainer…");
    juce::ChildProcess trainer;
    if (!trainer.start(argv, 0))
      return fail("Couldn't start Python.");

    while (!trainer.waitForProcessToFinish(250)) {
      if (cancelled()) {
        trainer.kill();
        return cancelOut();
      }
      const auto log = logFile.loadFileAsString();
      const int epoch = parseEpoch(log.getLastCharacters(4000));
      if (epoch >= 0) {
        const float t = juce::jlimit(0.0f, 1.0f, static_cast<float>(epoch) / static_cast<float>(settings.epochs));
        report(Progress::Phase::training, kRenderEnd + (kTrainEnd - kRenderEnd) * t,
               "Training: epoch " + juce::String(epoch + 1) + " of " + juce::String(settings.epochs));
      } else {
        report(Progress::Phase::training, kRenderEnd, "Starting the trainer…");
      }
    }
    if (cancelled())
      return cancelOut();

    const auto log = logFile.loadFileAsString();
    const auto produced = work.getChildFile("baked.nam");
    if (trainer.getExitCode() != 0 || !produced.existsAsFile()) {
      auto lines = juce::StringArray::fromLines(log.trim());
      juce::String tail = lines.isEmpty() ? juce::String("no output") : lines[lines.size() - 1];
      return fail("Training failed: " + tail.substring(0, 200));
    }
    double trainerEsr = -1.0;
    parseEsr(log, trainerEsr);

    // Verify natively: the saved model against what the chain rendered.
    report(Progress::Phase::verifying, kTrainEnd, "Checking the result…");
    const double nativeEsr = nativeEsrOf(produced, input, output);

    const auto dest = settings.destination;
    dest.getParentDirectory().createDirectory();
    if (!produced.copyFileTo(dest))
      return fail("Couldn't write " + dest.getFullPathName());

    Progress done;
    done.phase = Progress::Phase::done;
    done.fraction = 1.0f;
    done.message = "Saved " + dest.getFileName();
    done.trainerEsr = trainerEsr;
    done.nativeEsr = nativeEsr;
    done.output = dest;
    shared->post(std::move(done));
  }

  // Run the sweep through the baked .nam with the plugin's own engine and
  // compare it to the chain render. -1 when the model can't be run.
  static double nativeEsrOf(const juce::File& nam, const std::vector<float>& input,
                            const std::vector<float>& reference) {
    try {
      const auto text = nam.loadFileAsString().toStdString();
      const auto config = nlohmann::json::parse(text);
      std::vector<std::unique_ptr<nam::DSP>> instances;
      instances.push_back(nam::get_dsp(config));
      if (instances.front() == nullptr)
        return -1.0;
      NamEngine engine(std::move(instances), 1, 1);
      engine.prepare(kBlock);

      std::vector<float> rendered(input.size(), 0.0f);
      juce::AudioBuffer<float> buffer(1, kBlock);
      for (size_t off = 0; off < input.size(); off += kBlock) {
        const int n = static_cast<int>(std::min<size_t>(kBlock, input.size() - off));
        buffer.clear();
        for (int i = 0; i < n; ++i)
          buffer.setSample(0, i, input[off + static_cast<size_t>(i)]);
        engine.process(buffer);
        for (int i = 0; i < n; ++i)
          rendered[off + static_cast<size_t>(i)] = buffer.getSample(0, i);
      }
      // Skip the sweep file's first 10 s (latency blips and silence) so
      // prewarm and alignment don't count against the model.
      return errorToSignalRatio(reference, rendered, static_cast<size_t>(10 * kRate), true);
    } catch (...) {
      return -1.0;
    }
  }
};

// ---------------------------------------------------------------------------
// ChainBake

// The message-thread wait for the private processor's models. getChainState
// is a message-thread API, so the wait is a timer, not a worker loop.
class ChainBake::LoadWait : private juce::Timer {
public:
  LoadWait(TONE3000Processor& p, std::function<void(bool)> done) : proc(p), onDone(std::move(done)) {
    startTimer(kLoadPollMs);
  }
  ~LoadWait() override { stopTimer(); }

private:
  void timerCallback() override {
    elapsed += kLoadPollMs;
    bool loaded = true;
    const juce::var state = proc.getChainState(-1);
    for (const auto* lane : {state["chain"].getArray(), state["chainRight"].getArray()}) {
      if (lane == nullptr)
        continue;
      for (const auto& item : *lane)
        if (item["kind"].toString() == "tone" && !static_cast<bool>(item["loaded"]))
          loaded = false;
    }
    if (loaded && !proc.isChainEditFadeHeld())
      return finish(true);
    if (elapsed >= kLoadTimeoutMs)
      finish(false);
  }
  void finish(bool ok) {
    stopTimer();
    auto cb = std::move(onDone);
    cb(ok);  // may destroy this
  }

  TONE3000Processor& proc;
  std::function<void(bool)> onDone;
  int elapsed = 0;
};

ChainBake::ChainBake() = default;

ChainBake::~ChainBake() {
  if (auto* j = job.get())
    j->signalThreadShouldExit();
  if (shared != nullptr) {
    shared->cancel = true;
    shared->alive = false;
  }
  loadWait.reset();
  job.reset();
}

juce::String ChainBake::start(TONE3000Processor& source, Settings settings, ProgressFn onProgress) {
  if (running.load())
    return "A chain bake is already running.";

  const auto sweep = findTrainingInput();
  if (!sweep.existsAsFile())
    return juce::String("Training input not found. Copy ") + kTrainingInputName + " (from the plugin's "
           "resources/training folder) to " +
           appDataRoot().getChildFile("Training").getFullPathName() + ".";

  const auto python = findTrainerPython();
  if (python.isEmpty())
    return "Python with the NAM trainer wasn't found. Install it with: pip install neural-amp-modeler "
           "(or set T3K_PYTHON to the Python that has it).";

  // Retire the previous run.
  loadWait.reset();
  job.reset();
  if (shared != nullptr) {
    shared->alive = false;
    shared->processor.reset();
  }

  shared = std::make_shared<Shared>();
  shared->onProgress = std::move(onProgress);
  shared->running = &running;

  juce::MemoryBlock state;
  source.getStateInformation(state);

  auto proc = std::make_unique<TONE3000Processor>();
  proc->setPlayConfigDetails(2, 2, kRate, kBlock);
  proc->prepareToPlay(kRate, kBlock);
  proc->setStateInformation(state.getData(), static_cast<int>(state.getSize()));
  // A guitar is a mono source: feed channel 0 and let the processor fan it
  // out, whatever input mode the live rig uses.
  proc->setInputMode(TONE3000Processor::InputMode::Left);
  shared->processor = std::move(proc);
  running = true;

  auto newJob = std::make_unique<Job>(shared, std::move(settings));
  newJob->python = python;
  newJob->sweep = sweep;
  auto* jobPtr = newJob.get();
  job = std::move(newJob);

  if (shared->onProgress) {
    Progress p;
    p.message = "Loading the chain…";
    shared->onProgress(p);
  }

  auto sh = shared;
  loadWait = std::make_unique<LoadWait>(*shared->processor, [this, sh, jobPtr](bool ok) {
    // Deferred: finishing from inside the timer callback must not destroy it.
    juce::MessageManager::callAsync([this, sh, jobPtr, ok] {
      if (!sh->alive.load())
        return;
      loadWait.reset();
      if (!ok) {
        Progress p;
        p.phase = Progress::Phase::failed;
        p.message = "The chain's models didn't finish loading. Check your connection and try again.";
        sh->post(std::move(p));
        return;
      }
      if (job.get() == jobPtr)
        jobPtr->startThread();
    });
  });
  return {};
}

void ChainBake::cancel() {
  if (shared == nullptr || !running.load())
    return;
  shared->cancel = true;
  // Still waiting on loads: no worker yet to report it.
  if (loadWait != nullptr) {
    loadWait.reset();
    Progress p;
    p.phase = Progress::Phase::cancelled;
    p.message = "Cancelled";
    shared->post(std::move(p));
  }
}

}  // namespace t3k::bake
