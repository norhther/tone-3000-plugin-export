#pragma once
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <functional>
#include <vector>

class TONE3000Processor;

/**
 * Rig render: plays a probe signal through a private copy of the current rig
 * and returns what comes out, offline and faster than real time. This is the
 * "reamp" half of capturing the rig as a new NAM model: the player supplies
 * the TONE3000 sweep (T3K-sweep-v3.wav, 48 kHz mono, 24-bit), the rig's
 * response is rendered and written back in the format the TONE3000 trainer's
 * sweep method expects, ready to upload at tone3000.com/capture.
 *
 * The render never touches the live processor's audio: the rig is cloned
 * through get/setStateInformation (the state carries every block's model
 * bytes, so nothing is downloaded) into a second processor prepared at
 * 48 kHz, mono in / mono out. Mono out matters twice: a NAM capture is mono,
 * and a processor with a single output channel folds stereo chains with the
 * same sum a mono host track gets, so the render is what that rig sounds
 * like on one channel.
 *
 * What is copied from the live processor besides the state: the four
 * machine-default parameters (input calibration on/off and level,
 * oversampling on/off and factor). They are workstation settings, not part
 * of a rig's state, so a plain state copy would silently take the clone's own
 * defaults instead of what the player is hearing.
 *
 * Output contract (what the trainer wants):
 *  - exactly input.size() frames: the trainer rejects a length mismatch;
 *  - the chain's reported latency removed, so the response lines up with the
 *    sweep (model-baked delay is the trainer's to align, as with any
 *    hardware capture);
 *  - the rig's real output level, untouched. A peak at or above full scale is
 *    reported in `clipped` (the 24-bit WAV would clip there) rather than
 *    fixed, so the capture keeps the rig's gain.
 *
 * Threading: blocks until the chain's models have loaded and the whole sweep
 * is rendered (seconds of wall clock for a 190 s sweep). Call it from a
 * worker thread, not the message thread. `progress` (0..1) is called between
 * blocks; returning false cancels.
 */
namespace t3k::rigrender {

/** The only rate NAM A2 models and the TONE3000 sweep use. */
constexpr double kSampleRate = 48000.0;

struct Options {
  int blockSize = 512;
  /** Silence rendered before the sweep so the per-block fade-ins and any DC
      settle are not printed into the capture. */
  int primeMs = 250;
  /** How long to wait for every block's model to finish loading. */
  int loadTimeoutMs = 30000;
};

struct Result {
  bool ok = false;
  /** User-facing reason when !ok (also set to "Cancelled" on cancel). */
  juce::String error;
  int frames = 0;
  /** Largest |sample| in the rendered output, linear. */
  float peak = 0.0f;
  /** The render reached full scale: the 24-bit WAV is clipped there. */
  bool clipped = false;
  /** Samples of chain latency removed from the front. */
  int latencySamples = 0;
};

/** Render `input` (48 kHz mono) through a copy of `rig` into `output`
    (resized to input.size()). */
Result render(TONE3000Processor& rig, const std::vector<float>& input,
              std::vector<float>& output, const Options& options = {},
              const std::function<bool(float)>& progress = {});

/** Read a mono 48 kHz WAV (the sweep) as float. Returns "" on success, else
    a user-facing error. */
juce::String readSweep(const juce::File& file, std::vector<float>& samples);

/** Write mono samples as a 24-bit 48 kHz WAV (the capture the trainer takes).
    Returns "" on success, else a user-facing error. Overwrites `file`. */
juce::String writeCapture(const juce::File& file, const std::vector<float>& samples);

}  // namespace t3k::rigrender
