# Chain bake: render the chain to a single `.nam`

Plugin Settings → **Bake Chain to .nam** turns the current chain (every NAM
and IR block, EQ, gains, mix, stereo/branch layout) into one standard NAM
capture, for pedals, hardware and hosts that load a single model. Issue
[#250](https://github.com/tone-3000/tone3000-plugin/issues/250).

## How it works

```
live state ─▶ private headless processor ─▶ NAM standard input ─▶ rendered output
                                                     │                   │
                                     input.wav ◀─────┘     output.wav ◀──┘
                                           └──▶ `nam` trainer (Python) ──▶ model.nam
                                                                              │
                                  native ESR check (NamEngine) ◀──────────────┘
```

1. **Snapshot.** `getStateInformation` on the live processor; a second
   `TONE3000Processor` restores it (cached models ship in the state, so no
   network unless a tone isn't cached). The live chain is never touched and
   edits made during a bake don't leak in. Input mode is forced to mono
   (channel 0 fans out), as a guitar would feed it.
2. **Render.** `T3K-sweep-v3.wav` runs through that processor in 512-sample
   blocks at 48 kHz, compensating reported latency. Stereo chains are mixed
   to mono. A hot render is scaled under full scale (the trainer normalizes
   output level anyway); a silent one fails with a message.
3. **Train.** The plugin writes `input.wav` (the sweep, byte-identical) and
   `output.wav` (24-bit), then runs a small helper script with the official
   trainer: `nam.train.core.train(...)` (A2 packed WaveNet, the trainer's
   default) and `net.export(...)`. Progress comes from the trainer's `Epoch N`
   lines in a log file the plugin tails (no pipes, so cancel can't deadlock);
   cancel kills the process.
4. **Check.** The saved `.nam` is run with the plugin's own `NamEngine` over
   the same input and compared to the rendered chain: ESR with a
   least-squares gain fit (insensitive to the trainer's level normalization),
   skipping the first 10 s (blips and silence). Shown next to the save toast.
5. **Save.** Native save dialog, default `<preset>_combined.nam`.

## Why the standard input, and why not a user DI

The stock trainer refuses any input it doesn't recognize as one of the NAM
standard files (`_detect_input_version`), and its latency calibration relies
on that file's blips. `resources/training/T3K-sweep-v3.wav` is byte-identical
to the NAM v3.0.0 input (md5 `36cd1af62985c2fac3e654333e36431e`). Training on
a user-supplied DI would need a custom data config in the trainer; not done.

Lookup order: `T3K_TRAINING_INPUT`, `<app-data>/TONE3000/Training/`, the
machine-wide `…/TONE3000/Training/` (the Windows installer lays it down), then
`resources/training` next to a source checkout.

## Requirements and limits

- Python with `neural-amp-modeler` installed. Found via `T3K_PYTHON`, then
  `py -3` / `python` (Windows) or `python3` / `python`. A GPU makes this
  minutes instead of hours; the UI just shows epoch progress.
- Closing the plugin window cancels the bake (it belongs to the editor).
- The result approximates the chain at the sweep's level; like any NAM
  capture it is level-dependent, and time-based effects (reverb tails,
  delay, modulation) don't bake faithfully.
- Not done: the A/B preview from the issue. The ESR is reported instead.
