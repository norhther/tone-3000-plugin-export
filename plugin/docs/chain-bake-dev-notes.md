# Chain bake: development handoff

Status notes for continuing issue #250 (render the chain to one `.nam`).
User-facing design is in [`chain-bake.md`](chain-bake.md); this file is what
a developer needs to pick the work back up.

## State

- Branch `feature/render-chain-to-nam` on the fork
  `norhther/tone-3000-plugin-export` (remote `fork`; `origin` is upstream).
  No PR opened yet.
- CI (Linux): builds, the 9 `ChainBakeTest.*` tests pass, UI testbed passes.
- **Never run end to end.** The offline render, the Python trainer call and
  the Settings UI have not been exercised by hand. Windows/macOS unbuilt.
- Branch `ci-verify` (fork only) holds a throwaway `.github/workflows/
  verify.yml`: Linux build + ChainBake tests + testbed, with the `eigen`
  submodule fetched from `eigen-mirror/eigen` because GitLab was flaky. Do
  not PR it. To reuse: merge the feature branch into it and push.

## Files

| File | Role |
| --- | --- |
| `plugin/include/ChainBake.h`, `plugin/src/ChainBake.cpp` | Everything: helpers, trainer script (embedded string), `ChainBake` job |
| `plugin/ui/backend/Backend.h` | Optional `canBakeChain` / `startChainBake` / `cancelChainBake` + `BakeStatus` (defaults make the testbed mock a no-op) |
| `plugin/ui/backend/ProcessorBackend.*` | Owns a `ChainBake`, maps `Progress` to `BakeStatus` |
| `plugin/ui/views/settings/PluginSettingsPage.*` | The "Bake Chain to .nam" row, save dialog, progress text, cancel |
| `resources/training/T3K-sweep-v3.wav` | NAM v3.0.0 standard input (md5 `36cd1af6…2422`) |
| `script/installer/windows/tone3000.iss` | Installs the sweep to `{commonappdata}\TONE3000\Training` |
| `test/src/chain_bake_tests.cpp` | Helper tests |

## Flow and threading (ChainBake.cpp)

1. `start()` (message thread): finds the sweep and a Python that has `nam`,
   snapshots `getStateInformation`, builds a private `TONE3000Processor`
   (48 kHz, 512 block, input mode Left), restores the state.
2. `LoadWait` (message-thread timer): polls `getChainState` until every tone
   is loaded and `isChainEditFadeHeld()` is false (90 s timeout).
   `getChainState` is not known to be thread-safe, hence the timer.
3. Worker thread `Job`: reads the sweep, renders block by block, writes
   `input.wav` (copy of the sweep) and `output.wav` (24-bit) in a temp dir,
   runs `python -u t3k_bake_train.py …`, tails `train.log` for `Epoch N`,
   then checks the result with `NamEngine` and copies it to the destination.
4. Final progress is posted with `callAsync`; that post also destroys the
   private processor on the message thread and clears `running`.

Trainer output goes to a log file, not pipes, so cancel (kill) can never
deadlock on a full pipe. The script forces `silent=True` (otherwise the
trainer blocks on `plt.show()`) and `ignore_checks=True`.

## Constraints discovered

- The plugin only loads A2 WaveNets; the current `neural-amp-modeler`
  trainer produces packed A2 by default (`config_model_packed.json`).
- `nam.train.core.train` raises for any input that isn't a known NAM
  standard file, so a user DI cannot be used without a custom data config.
- ESR in the UI is the native check with a least-squares gain fit, skipping
  the first 10 s. It is not the trainer's validation ESR (that one is parsed
  into `Progress::trainerEsr` but not shown).
- Upstream CI (`build.yml`) is `workflow_dispatch` only and needs the
  `builds` environment secrets; forks don't get it registered.

## Known gaps / ideas

- Run it for real; expect fixes in: Windows `ChildProcess` quoting of the
  `py -3` launcher, `UserMetadata`/`export` kwargs if the trainer API moved,
  time-to-first-epoch messaging, and `Epoch N` parsing against real logs.
- A/B preview (original chain vs baked) from the issue is not built; the
  render and the baked model are both available in `Job::bake`, so a
  preview needs those kept around plus a UI to audition them.
- Show trainer ESR next to the native one; surface the elapsed time.
- Epoch count and model name are fixed (`Settings`) with no UI.
- Mac/Linux installers and `script/install-plugin.sh` don't lay down the
  sweep; only the Windows `.iss` does. The 27 MB asset is committed to git.
- Closing the editor cancels the bake (owned by `ProcessorBackend`); moving
  it to the processor would let it survive.
- Time-based effects (reverb/delay/modulation) don't bake faithfully; maybe
  warn when such blocks are in the chain.
- An integration test needs Python + the trainer; none exists. A fake
  trainer script selectable via `T3K_PYTHON` would allow one in CI.
- Before PR: drop nothing from `ci-verify` in, squash if wanted, and check
  the 27 MB WAV is acceptable to maintainers (alternative: download on
  demand or generate it).

## Dev setup

No toolchain is needed to edit; to build locally install CMake + MSVC (or
use the fork's CI). Env overrides: `T3K_PYTHON` (interpreter),
`T3K_TRAINING_INPUT` (sweep path). Trainer: `pip install neural-amp-modeler`.
