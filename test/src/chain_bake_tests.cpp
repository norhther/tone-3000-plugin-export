// Chain bake: the pure helpers around the render-to-.nam pipeline (WAV
// export/import, ESR, trainer output parsing). The trainer itself is external
// Python and isn't exercised here.

#include "ChainBake.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

using namespace t3k::bake;

TEST(ChainBakeTest, EsrOfIdenticalSignalsIsZero) {
  const auto x = makeNoise(4800, 1, 0.5f);
  EXPECT_NEAR(errorToSignalRatio(x, x), 0.0, 1e-12);
}

TEST(ChainBakeTest, EsrMatchesDefinition) {
  // ŷ = 0.9·y has error 0.01·|y|², so ESR = 0.01.
  const auto y = makeNoise(4800, 2, 0.5f);
  std::vector<float> test(y.size());
  for (size_t i = 0; i < y.size(); ++i)
    test[i] = 0.9f * y[i];
  EXPECT_NEAR(errorToSignalRatio(y, test), 0.01, 1e-5);
}

TEST(ChainBakeTest, GainFitIgnoresLevelButNotShape) {
  const auto y = makeNoise(4800, 3, 0.5f);
  std::vector<float> scaled(y.size());
  for (size_t i = 0; i < y.size(); ++i)
    scaled[i] = 0.25f * y[i];
  EXPECT_NEAR(errorToSignalRatio(y, scaled, 0, true), 0.0, 1e-9);
  EXPECT_GT(errorToSignalRatio(y, scaled, 0, false), 0.5);

  const auto other = makeNoise(4800, 4, 0.5f);
  EXPECT_GT(errorToSignalRatio(y, other, 0, true), 0.5);
}

TEST(ChainBakeTest, EsrSkipsLeadingSamples) {
  auto y = makeNoise(1000, 5, 0.5f);
  auto test = y;
  for (size_t i = 0; i < 100; ++i)
    test[i] = 0.0f;  // wrong only in the skipped prefix
  EXPECT_GT(errorToSignalRatio(y, test, 0), 0.0);
  EXPECT_NEAR(errorToSignalRatio(y, test, 100), 0.0, 1e-12);
}

TEST(ChainBakeTest, Wav24RoundTripsAtMono48k) {
  const auto dir = juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("t3k-bake-test");
  dir.createDirectory();
  const auto file = dir.getChildFile("rt.wav");
  const auto x = makeSine(48000, 440.0, 0.5f);
  ASSERT_TRUE(writeMono48kWav24(file, x));

  juce::String err;
  const auto back = readMono48k(file, err);
  ASSERT_EQ(back.size(), x.size()) << err;
  EXPECT_LT(errorToSignalRatio(x, back), 1e-9);  // 24-bit quantization only
  dir.deleteRecursively();
}

TEST(ChainBakeTest, OtherRatesAreResampledTo48k) {
  const auto dir = juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("t3k-bake-test2");
  dir.createDirectory();
  const auto file = dir.getChildFile("44k.wav");
  {
    std::unique_ptr<juce::FileOutputStream> stream(file.createOutputStream());
    ASSERT_NE(stream, nullptr);
    juce::WavAudioFormat wav;
    std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.get(), 44100.0, 2, 24, {}, 0));
    ASSERT_NE(writer, nullptr);
    stream.release();
    juce::AudioBuffer<float> buf(2, 44100);
    for (int i = 0; i < 44100; ++i) {
      const float v = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(i) / 44100.0f);
      buf.setSample(0, i, v);
      buf.setSample(1, i, v);
    }
    writer->writeFromAudioSampleBuffer(buf, 0, 44100);
  }
  juce::String err;
  const auto out = readMono48k(file, err);
  EXPECT_NEAR(static_cast<double>(out.size()), 48000.0, 2.0) << err;
  dir.deleteRecursively();
}

TEST(ChainBakeTest, ParsesTrainerOutput) {
  EXPECT_EQ(parseEpoch("nothing yet"), -1);
  EXPECT_EQ(parseEpoch("Epoch 3: 100%|####| 40/40\rEpoch 12:  5%|#"), 12);

  double esr = 0.0;
  EXPECT_FALSE(parseEsr("T3K_DONE", esr));
  EXPECT_TRUE(parseEsr("blah\nT3K_ESR 0.00123\nT3K_DONE\n", esr));
  EXPECT_NEAR(esr, 0.00123, 1e-9);
  EXPECT_TRUE(parseEsr("T3K_ESR 1.5e-03\n", esr));
  EXPECT_NEAR(esr, 0.0015, 1e-9);
}

TEST(ChainBakeTest, TrainerScriptIsSelfContained) {
  const juce::String script(trainerScript());
  EXPECT_TRUE(script.contains("core.train("));
  EXPECT_TRUE(script.contains("ignore_checks=True"));
  EXPECT_TRUE(script.contains("silent=True"));  // silent=False would block on plt.show()
  EXPECT_TRUE(script.contains("T3K_ESR"));
}

TEST(ChainBakeTest, ShippedSweepIsFoundFromTheSourceTree) {
  // The tests run from the build tree; the lookup walks up to resources/training.
  bool listed = false;
  for (const auto& p : trainingInputSearchPaths())
    listed = listed || p.getFileName() == kTrainingInputName;
  EXPECT_TRUE(listed);
}
