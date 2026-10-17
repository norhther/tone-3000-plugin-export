#include "RigRender.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "Processor.h"

namespace t3k::rigrender {

namespace {

// The workstation parameters a rig's sound depends on but a state restore may
// leave at the clone's own defaults (see RigRender.h).
constexpr const char* kMatchedParameterIds[] = {"calibrateInput", "inputCalibrationLevel",
                                                "osEnabled", "osFactor"};

// Same condition the DSP tests wait on: every tone block loaded and the
// restore's chain-edit mute released (it runs on a wall-clock waiter, so
// rendering faster than real time would otherwise print the mute).
bool waitForChainLoaded(TONE3000Processor& proc, int timeoutMs) {
  const auto deadline = juce::Time::getMillisecondCounter() + static_cast<juce::uint32>(timeoutMs);
  while (juce::Time::getMillisecondCounter() < deadline) {
    const juce::var state = proc.getChainState(-1);
    bool allLoaded = true;
    for (const auto* lane : {state["chain"].getArray(), state["chainRight"].getArray()}) {
      if (lane == nullptr) continue;
      for (const auto& item : *lane)
        if (item["kind"].toString() == "tone" && !static_cast<bool>(item["loaded"])) allLoaded = false;
    }
    if (allLoaded && !proc.isChainEditFadeHeld()) return true;
    juce::Thread::sleep(20);
  }
  return false;
}

Result fail(const juce::String& message) {
  Result r;
  r.error = message;
  return r;
}

}  // namespace

Result render(TONE3000Processor& rig, const std::vector<float>& input, std::vector<float>& output,
              const Options& options, const std::function<bool(float)>& progress) {
  if (input.empty()) return fail("The sweep is empty.");
  const int block = std::max(options.blockSize, 64);

  // Private copy of the rig: 48 kHz, one channel in and out (see RigRender.h).
  juce::MemoryBlock state;
  rig.getStateInformation(state);

  auto clone = std::make_unique<TONE3000Processor>();
  clone->setPlayConfigDetails(1, 1, kSampleRate, block);
  clone->prepareToPlay(kSampleRate, block);
  clone->setStateInformation(state.getData(), static_cast<int>(state.getSize()));

  // Match the workstation parameters, then re-prepare: oversampling changes
  // are applied by prepareToPlay synchronously (the live-toggle path needs a
  // running message pump this thread does not have).
  bool changed = false;
  for (const auto* id : kMatchedParameterIds) {
    auto* from = rig.parameters.getParameter(id);
    auto* to = clone->parameters.getParameter(id);
    if (from == nullptr || to == nullptr) continue;
    if (std::abs(from->getValue() - to->getValue()) > 1.0e-6f) {
      to->setValueNotifyingHost(from->getValue());
      changed = true;
    }
  }
  if (changed) clone->prepareToPlay(kSampleRate, block);

  if (!waitForChainLoaded(*clone, options.loadTimeoutMs))
    return fail("The rig's models did not finish loading.");

  const int latency = std::max(clone->getLatencySamples(), 0);
  const int primeFrames = std::max(options.primeMs, 0) * static_cast<int>(kSampleRate) / 1000;
  const int sweepFrames = static_cast<int>(input.size());
  // Everything the clone has to produce: silence to settle, the latency that
  // gets dropped, then the sweep itself.
  const int total = primeFrames + latency + sweepFrames;
  const int blocks = (total + block - 1) / block;

  output.assign(input.size(), 0.0f);
  juce::AudioBuffer<float> buffer(1, block);
  juce::MidiBuffer midi;
  float peak = 0.0f;

  // Frame f of the stream is silence before primeFrames, then the sweep (and
  // silence again past its end, so the delayed tail of the last frames drains).
  for (int b = 0; b < blocks; ++b) {
    if (progress && !progress(static_cast<float>(b) / static_cast<float>(blocks))) return fail("Cancelled");

    float* data = buffer.getWritePointer(0);
    for (int i = 0; i < block; ++i) {
      const int sweepIndex = b * block + i - primeFrames;
      data[i] = (sweepIndex >= 0 && sweepIndex < sweepFrames) ? input[static_cast<size_t>(sweepIndex)] : 0.0f;
    }
    clone->processBlock(buffer, midi);

    // Output frame g carries the sweep sample g - primeFrames - latency.
    const float* out = buffer.getReadPointer(0);
    for (int i = 0; i < block; ++i) {
      const int outIndex = b * block + i - primeFrames - latency;
      if (outIndex < 0 || outIndex >= sweepFrames) continue;
      const float s = out[i];
      output[static_cast<size_t>(outIndex)] = s;
      peak = std::max(peak, std::abs(s));
    }
  }
  if (progress) progress(1.0f);

  if (!std::isfinite(peak)) return fail("The rig produced invalid audio (NaN or infinity).");

  Result r;
  r.ok = true;
  r.frames = sweepFrames;
  r.peak = peak;
  r.clipped = peak >= 1.0f;
  r.latencySamples = latency;
  return r;
}

juce::String readSweep(const juce::File& file, std::vector<float>& samples) {
  if (!file.existsAsFile()) return "That file doesn't exist.";
  juce::AudioFormatManager formats;
  formats.registerBasicFormats();
  std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
  if (reader == nullptr) return "That isn't an audio file this plugin can read (use the WAV).";
  if (reader->numChannels != 1) return "The sweep must be mono.";
  if (std::abs(reader->sampleRate - kSampleRate) > 0.5) return "The sweep must be 48 kHz.";
  if (reader->lengthInSamples <= 0 || reader->lengthInSamples > 0x7fffffffLL / 2) return "The sweep is empty or too long.";

  const int n = static_cast<int>(reader->lengthInSamples);
  juce::AudioBuffer<float> buffer(1, n);
  if (!reader->read(&buffer, 0, n, 0, true, false)) return "Couldn't read the sweep.";
  samples.assign(buffer.getReadPointer(0), buffer.getReadPointer(0) + n);
  return {};
}

juce::String writeCapture(const juce::File& file, const std::vector<float>& samples) {
  if (samples.empty()) return "Nothing to write.";
  if (file.existsAsFile() && !file.deleteFile()) return "Couldn't replace the existing file.";

  std::unique_ptr<juce::FileOutputStream> stream(file.createOutputStream());
  if (stream == nullptr) return "Couldn't create the file.";

  juce::WavAudioFormat wav;
  std::unique_ptr<juce::AudioFormatWriter> writer(
      wav.createWriterFor(stream.get(), kSampleRate, 1, 24, {}, 0));
  if (writer == nullptr) return "Couldn't start the WAV writer.";
  stream.release();  // the writer owns the stream now

  const float* channel = samples.data();
  if (!writer->writeFromFloatArrays(&channel, 1, static_cast<int>(samples.size()))) return "Couldn't write the WAV.";
  return {};
}

}  // namespace t3k::rigrender
