// Rig render (RigRender.h): the offline reamp half of "capture my rig as a
// NAM model". Pins the output contract the TONE3000 sweep method needs (exact
// length, the rig's own level, 24-bit mono 48 kHz files) and that the render
// really is the rig: a clone of the live chain, not an approximation of it.
#include "RigRender.h"

#include "Processor.h"
#include "chain_test_helpers.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr int kBlock = 512;

// One NAM block in the left lane of a mono rig (cache-first load, no network).
void seedMonoNamChain(ChainTestProcessor& proc, const char* namFile = "a2-amp-test.nam") {
  juce::ValueTree state("ChainSnapshot");
  state.setProperty("stereoEnabled", false, nullptr);
  juce::ValueTree left("ChainBlocks");
  left.appendChild(makeNamBlockTree("blk-nam", 1, 100, namFile), nullptr);
  state.appendChild(left, nullptr);
  proc.restoreFromTree(state);
}

// A live-rig stand-in: 48 kHz mono, as a standalone rig on a one-channel
// output would be.
void prepareLiveRig(ChainTestProcessor& proc) {
  proc.setPlayConfigDetails(1, 1, kFs, kBlock);
  proc.prepareToPlay(kFs, kBlock);
}

std::vector<float> processMono(TONE3000Processor& proc, const std::vector<float>& in) {
  std::vector<float> out(in.size(), 0.0f);
  juce::AudioBuffer<float> buffer(1, kBlock);
  juce::MidiBuffer midi;
  for (size_t off = 0; off + kBlock <= in.size(); off += kBlock) {
    buffer.copyFrom(0, 0, in.data() + off, kBlock);
    proc.processBlock(buffer, midi);
    std::copy(buffer.getReadPointer(0), buffer.getReadPointer(0) + kBlock, out.begin() + static_cast<long>(off));
  }
  return out;
}

float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b, size_t from, size_t to) {
  float worst = 0.0f;
  for (size_t i = from; i < to; ++i)
    worst = std::max(worst, std::abs(a[i] - b[i]));
  return worst;
}

// A mono WAV of arbitrary rate/channels, to feed the reader's rejections.
juce::File writeTestWav(const juce::File& dir, const juce::String& name, double rate, int channels) {
  const auto file = dir.getChildFile(name);
  std::unique_ptr<juce::FileOutputStream> stream(file.createOutputStream());
  juce::WavAudioFormat wav;
  std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.get(), rate, static_cast<unsigned>(channels), 24, {}, 0));
  stream.release();
  juce::AudioBuffer<float> buffer(channels, 480);
  buffer.clear();
  writer->writeFromAudioSampleBuffer(buffer, 0, 480);
  return file;
}

}  // namespace

TEST(RigRenderTest, OutputIsExactlyTheSweepLengthAtTheRigsLevel) {
  ChainTestProcessor rig;
  prepareLiveRig(rig);
  seedMonoNamChain(rig);
  ASSERT_TRUE(waitForChainLoaded(rig));

  // Deliberately not a multiple of the block size: the trainer rejects a
  // capture that is even one sample off the sweep's length.
  const std::vector<float> sweep = makeNoise(40 * kBlock + 137, 99, 0.1f);
  std::vector<float> out;
  const auto result = t3k::rigrender::render(rig, sweep, out);

  ASSERT_TRUE(result.ok) << result.error.toStdString();
  EXPECT_EQ(out.size(), sweep.size());
  EXPECT_EQ(result.frames, static_cast<int>(sweep.size()));
  EXPECT_EQ(result.latencySamples, 0) << "48 kHz chain reports no latency";
  EXPECT_GT(result.peak, 0.0f) << "the rig produced silence";
  EXPECT_TRUE(std::isfinite(result.peak));
  EXPECT_FALSE(result.clipped);
}

TEST(RigRenderTest, RenderIsTheLiveRigNotAnApproximation) {
  ChainTestProcessor rig;
  prepareLiveRig(rig);
  seedMonoNamChain(rig);
  ASSERT_TRUE(waitForChainLoaded(rig));
  letAudioGoIdle();

  const std::vector<float> sweep = makeNoise(120 * kBlock, 7, 0.1f);
  std::vector<float> rendered;
  const auto result = t3k::rigrender::render(rig, sweep, rendered);
  ASSERT_TRUE(result.ok) << result.error.toStdString();

  // The same input straight through the live processor. Skip the opening
  // samples: the live rig's block fade-in (it has just loaded) and the
  // clone's primed-from-silence model state differ there and nowhere else.
  const auto live = processMono(rig, sweep);
  EXPECT_LT(maxAbsDiff(rendered, live, 8192, sweep.size()), 1.0e-3f);
  EXPECT_GT(result.peak, 0.01f);
}

TEST(RigRenderTest, RenderLeavesTheLiveRigUntouched) {
  ChainTestProcessor rig;
  prepareLiveRig(rig);
  seedMonoNamChain(rig);
  ASSERT_TRUE(waitForChainLoaded(rig));
  const auto revisionBefore = rig.getChainState(-1)["revision"];

  const std::vector<float> sweep = makeNoise(20 * kBlock, 5, 0.1f);
  std::vector<float> out;
  ASSERT_TRUE(t3k::rigrender::render(rig, sweep, out).ok);

  const juce::var after = rig.getChainState(-1);
  EXPECT_EQ(after["revision"].toString(), revisionBefore.toString()) << "rendering edited the rig";
  EXPECT_EQ(after["chain"].getArray()->size(), rig.getChainState(-1)["chain"].getArray()->size());
}

TEST(RigRenderTest, EmptyRigPassesTheSweepThrough) {
  ChainTestProcessor rig;
  prepareLiveRig(rig);

  const std::vector<float> sweep = makeNoise(60 * kBlock, 3, 0.25f);
  std::vector<float> out;
  const auto result = t3k::rigrender::render(rig, sweep, out);

  ASSERT_TRUE(result.ok) << result.error.toStdString();
  ASSERT_EQ(out.size(), sweep.size());
  EXPECT_LT(maxAbsDiff(out, sweep, 4096, sweep.size()), 0.005f)
      << "an empty chain must not colour or delay the sweep";
}

TEST(RigRenderTest, FullScaleOutputIsReportedNotFixed) {
  ChainTestProcessor rig;
  prepareLiveRig(rig);

  // Loud enough that an empty chain's output reaches full scale: the render
  // keeps the rig's real level and says so, it does not normalize.
  std::vector<float> sweep = makeSine(60 * kBlock, 440.0, 1.5f);
  std::vector<float> out;
  const auto result = t3k::rigrender::render(rig, sweep, out);

  ASSERT_TRUE(result.ok) << result.error.toStdString();
  EXPECT_TRUE(result.clipped);
  EXPECT_GE(result.peak, 1.0f);
}

TEST(RigRenderTest, CancelStopsTheRender) {
  ChainTestProcessor rig;
  prepareLiveRig(rig);

  const std::vector<float> sweep = makeNoise(60 * kBlock, 11, 0.1f);
  std::vector<float> out;
  int calls = 0;
  const auto result = t3k::rigrender::render(rig, sweep, out, {}, [&calls](float) { return ++calls < 3; });

  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.error, "Cancelled");
}

TEST(RigRenderTest, EmptySweepIsRejected) {
  ChainTestProcessor rig;
  prepareLiveRig(rig);
  std::vector<float> out;
  const auto result = t3k::rigrender::render(rig, {}, out);
  EXPECT_FALSE(result.ok);
  EXPECT_TRUE(result.error.isNotEmpty());
}

TEST(RigRenderTest, CaptureWavRoundTripsAs24BitMono48k) {
  const auto dir = juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("t3k-rig-render-test");
  dir.createDirectory();
  const auto file = dir.getChildFile("capture.wav");

  const auto samples = makeSine(48000 + 13, 997.0, 0.5f);
  ASSERT_EQ(t3k::rigrender::writeCapture(file, samples), juce::String());

  juce::AudioFormatManager formats;
  formats.registerBasicFormats();
  std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
  ASSERT_NE(reader, nullptr);
  EXPECT_EQ(reader->numChannels, 1u);
  EXPECT_EQ(reader->bitsPerSample, 24u);
  EXPECT_DOUBLE_EQ(reader->sampleRate, 48000.0);
  EXPECT_EQ(reader->lengthInSamples, static_cast<juce::int64>(samples.size()));

  std::vector<float> back;
  ASSERT_EQ(t3k::rigrender::readSweep(file, back), juce::String());
  ASSERT_EQ(back.size(), samples.size());
  EXPECT_LT(maxAbsDiff(back, samples, 0, samples.size()), 5.0e-7f) << "24-bit quantization only";

  dir.deleteRecursively();
}

TEST(RigRenderTest, SweepReaderRejectsWhatTheTrainerWouldNot) {
  const auto dir = juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("t3k-rig-render-test-reject");
  dir.createDirectory();
  std::vector<float> samples;

  EXPECT_TRUE(t3k::rigrender::readSweep(dir.getChildFile("missing.wav"), samples).isNotEmpty());
  EXPECT_TRUE(t3k::rigrender::readSweep(writeTestWav(dir, "stereo.wav", 48000.0, 2), samples).contains("mono"));
  EXPECT_TRUE(t3k::rigrender::readSweep(writeTestWav(dir, "cd.wav", 44100.0, 1), samples).contains("48 kHz"));
  EXPECT_EQ(t3k::rigrender::readSweep(writeTestWav(dir, "ok.wav", 48000.0, 1), samples), juce::String());
  EXPECT_EQ(samples.size(), 480u);

  dir.deleteRecursively();
}
