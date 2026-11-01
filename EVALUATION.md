# Does int8 cost anything? An objective check

Listening to one sample is not evidence. This note records what the objective metrics actually say, because the
obvious comparison — same seed, listen to both — does not work here, and it is worth explaining why before the
numbers.

## Why A/B comparison fails

At a fixed seed, bf16 and int8 produce **different performances**, not an original and a distorted copy:

| | duration | waveform correlation with bf16 |
|---|---|---|
| bf16, seed 1234, sampled (T=0.8) | 6.16 s | — |
| int8, seed 1234, sampled (T=0.8) | 6.48 s | 0.016 |
| bf16, greedy (T=0) | 6.96 s | — |
| int8, greedy (T=0) | 7.44 s | 0.016 |

Greedy decoding does not rescue it either. A small numeric difference flips one codebook token, and the
autoregressive path never returns to the bf16 trajectory. So "int8 reproduces bf16" is not a property that can be
tested, and asking a listener to compare two clips is asking them to compare two valid but different takes.

What can be tested is whether each precision, scored on its own over a set of utterances, is of the same quality.

## Protocol

20 Chinese sentences covering statements, questions, exclamations, short and long lines, and the interjection-led
style this voice is used for. Each rendered once with bf16 weights and once with int8 (torchao
`Int8WeightOnlyConfig`), temperature 0.8, seed 1234, same reference clip.

| Metric | What it measures | Tool |
|---|---|---|
| CER | does it still say the right words | Paraformer (sherpa-onnx), punctuation stripped |
| SECS | is it still the same speaker | ECAPA-TDNN (speechbrain) cosine vs the reference clip |
| UTMOS | does it still sound natural | SpeechMOS `utmos22_strong` |

Thresholds, set before running: CER no worse than +1.0 point, SECS no worse than −0.01, UTMOS no worse than −0.05.

## Results

| Weights | n | CER % | SECS | UTMOS | mean duration |
|---|---|---|---|---|---|
| bf16 | 20 | 1.23 | 0.670 | 3.21 | 2.54 s |
| int8 | 20 | 0.56 | 0.672 | 3.29 | 2.55 s |
| **difference** | | **−0.67** | **+0.002** | **+0.077** | +0.01 s |

All three differences fall inside the thresholds, and all three point slightly in int8's favour — which is noise,
not an improvement. With n=20 and a single seed, a 0.7-point CER difference means approximately one character
across the whole set.

Audio for all 40 clips is in `samples/eval/`, with `manifest.json` mapping file id to the sentence it should say.
Judge for yourself; that is the point of shipping them.

## What this does not establish

- **n=20, one seed, one voice.** Enough to catch a real regression, not enough to certify equality.
- **CER measures intelligibility, not prosody.** int8 could sound flatter without changing a single character. The
  UTMOS reading argues against that but does not settle it.
- **UTMOS is a predictor, not a listener.** Its correlation with human MOS is good but not perfect, and it was
  trained on read speech from a different distribution.
- **SECS 0.67 is not a quality score.** It is cosine similarity to a 1.5 s reference clip across *different*
  sentences, so the absolute value is low by construction. Only the difference between the two rows means
  anything.

## Reproducing

The generation and scoring scripts are not published here yet — they depend on a local ASR and speaker-encoder
setup. What is published is the input manifest, all 40 outputs, and the metric definitions above, which is enough
for anyone to re-score the same audio with their own tooling.
