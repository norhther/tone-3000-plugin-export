#pragma once

// Chain bake: render the current signal chain to a single .nam capture.
//
//   live chain state ──▶ second (headless) processor ──▶ NAM standard input
//        │                                                     │
//        │                              rendered output ◀──────┘
//        ▼                                     │
//   input.wav + output.wav ──▶ `nam` trainer (external Python) ──▶ model.nam
//                                                                    │
//                          native ESR check (NamEngine) ◀────────────┘
//
// The chain is never touched: the render runs on a private TONE3000Processor
// restored from the live processor's state blob, so playing keeps working and
// the user's edits during a bake don't leak into it.
//
// Training is delegated to the official trainer (`pip install
// neural-amp-modeler`): the plugin has no trainer of its own and the stock
// trainer only accepts the NAM standard input files, so the training signal is
// the T3K-sweep-v3.wav shipped under resources/training (byte-identical to
// the NAM v3.0.0 input). See plugin/docs/chain-bake.md.
//
// Threading: start()/cancel() and the progress callback are message-thread.
// The heavy lifting runs on a worker thread; the processor is built and
// destroyed on the message thread, like any plugin instance.

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_events/juce_events.h>

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

class TONE3000Processor;

namespace t3k::bake {

/** Pure helpers, exposed for tests. */

/** Plain-ASCII file name of the training input the trainer recognizes. */
constexpr const char* kTrainingInputName = "T3K-sweep-v3.wav";

/** Where the training input is looked up, most specific first. */
std::vector<juce::File> trainingInputSearchPaths();

/** First existing training input, or a non-existent File. */
juce::File findTrainingInput();

/** Mono-mix a WAV (any channel count) at its own rate to 48 kHz. Empty on
    failure, with `error` set. */
std::vector<float> readMono48k(const juce::File& file, juce::String& error);

/** 24-bit mono 48 kHz PCM, the format the trainer reads. */
bool writeMono48kWav24(const juce::File& file, const std::vector<float>& samples);

/** Error-to-signal ratio of `test` against `reference` over [skip, end).
    With `fitGain`, `test` is first scaled by the least-squares gain, which
    makes the figure insensitive to the trainer's output level normalization. */
double errorToSignalRatio(const std::vector<float>& reference, const std::vector<float>& test,
                          size_t skip = 0, bool fitGain = false);

/** Python interpreter candidates for the trainer, most specific first
    (T3K_PYTHON override, then the platform's usual names). Each entry is a
    full argv prefix: {"py", "-3"} or {"python3"}. */
std::vector<juce::StringArray> pythonCandidates();

/** The first candidate that can `import nam`; empty when none can. */
juce::StringArray findTrainerPython();

/** Source of the helper script the plugin writes next to the training files. */
const char* trainerScript();

/** Pull "Epoch N" out of trainer console output. -1 when absent. */
int parseEpoch(const juce::String& text);

/** "ESR: 0.0123" / "validation_esr=…" out of trainer console output. */
bool parseEsr(const juce::String& text, double& esr);

struct Progress {
  enum class Phase { rendering, training, verifying, done, failed, cancelled };
  Phase phase = Phase::rendering;
  float fraction = 0.0f;  // overall, 0..1
  juce::String message;
  // Set with Phase::done.
  double trainerEsr = -1.0;  // validation ESR reported by the trainer, -1 = unknown
  double nativeEsr = -1.0;   // ESR of the saved .nam against the rendered chain, -1 = unknown
  juce::File output;
};

class ChainBake {
public:
  struct Settings {
    juce::File destination;  // the .nam to write
    juce::String modelName;  // metadata name; defaults to the file's base name
    int epochs = 100;
  };

  using ProgressFn = std::function<void(const Progress&)>;

  ChainBake();
  ~ChainBake();

  /** Snapshot `source`'s chain and bake it. Returns an error message when
      the bake can't even start (no sweep file, no trainer, already running);
      empty on success. `onProgress` is called on the message thread. */
  juce::String start(TONE3000Processor& source, Settings settings, ProgressFn onProgress);

  /** Ask the running bake to stop (kills the trainer). The progress callback
      reports Phase::cancelled once it has. */
  void cancel();

  bool isRunning() const { return running.load(); }

private:
  class Job;
  class LoadWait;
  struct Shared;
  std::unique_ptr<Job> job;
  std::unique_ptr<LoadWait> loadWait;
  std::shared_ptr<Shared> shared;
  std::atomic<bool> running{false};
};

}  // namespace t3k::bake
