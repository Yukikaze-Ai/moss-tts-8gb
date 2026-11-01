# Running MOSS-TTS-Realtime at RTF 0.95 on a single 8 GB consumer GPU

Field notes on making [MOSS-TTS-Realtime](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime) (1.7B, Apache-2.0)
sustain real-time streaming on a GeForce RTX 3060 Ti with 8 GB of VRAM, where the audio codec has to live on the
CPU because the talker and the codec together need about 11 GB.

Everything below was measured on this one machine. Your bottleneck may sit somewhere else — the last section says
how to find out where yours is.

## Result

Same sentence, same machine, codec on CPU, warm graphs:

| | first audio | total | RTF | steady step |
|---|---|---|---|---|
| as shipped (eager) | 1.02 s | 13.5 s | 2.08 | 109.4 ms |
| **after the six conditions** | **0.92 s** | **7.10 s** | **0.90** | **61.5 ms** |

RTF is synthesis seconds per second of audio; below 1.0 is what streaming requires.

With `torchao` int8 weights the same path runs at RTF 0.94–0.98 while the talker drops from 4.34 GiB to 2.82 GiB,
so int8 is not a speed-for-memory trade here: it is the same speed for 1.5 GiB less. It costs startup — the
quantised graph takes 818 s to compile cold, about 300 s with a warm inductor cache — so a service should report
"starting" until a real synthesis has completed.

## The six conditions

Each one alone does nothing useful. That is the point.

### 1. Triton must actually be installed

```bash
pip install triton-windows==3.9.0.post30   # Windows
# Linux wheels already ship with torch
```

Without it, `torch.utils._triton.has_triton()` is `False`, inductor cannot run, and `torch.compile` silently falls
back to eager. Every "compile succeeded" measurement before this was measuring eager.

### 2. Turn off int specialisation

```python
torch._dynamo.config.specialize_int = False
```

`step()` takes `gen_step`, an `int` that increments every frame. Dynamo specialises on int arguments by default, so
every frame was a new specialisation and a recompile — the compiled model was **13x slower** end to end than
eager. This one line stopped it. After the fix, a whole utterance produced `unique_graphs: 2`.

### 3. Compile the depth transformer, never the talker

The talker is a 1.7B forward at about 31 ms/frame, close to the memory-bandwidth floor (3.4 GB of weights at
~448 GB/s ≈ 7.6 ms, plus KV attention and kernel launches). The depth transformer runs 16 codebook forwards per
frame and cost about 96 ms. Compiling the talker buys nothing and adds roughly 100 s of warmup.

### 4. Pin the KV cache

```python
from transformers.cache_utils import StaticCache

cache = StaticCache(
    config=model.config.language_config,   # MossTTSRealtimeConfig has no num_hidden_layers
    max_batch_size=1, max_cache_len=512,
    device="cuda:0", dtype=torch.bfloat16,
)
inferencer.past_key_values = cache          # must be set BEFORE prefill, or it is replaced
```

A `DynamicCache` grows every frame; a `StaticCache` is allocated once, so tensor shapes stop moving.

### 5. Fix the attention mask length

`step()` builds its mask with `torch.cat`, which changes its length every frame — another new shape. Preallocate
one buffer of `max_cache_len` and write the new position in place, passing `cache_position` explicitly:

```python
if getattr(self, "_fixed_mask", None) is None:
    self._fixed_mask = torch.ones(1, MAX_CACHE, dtype=torch.bool, device=device)
    self._cache_pos = torch.zeros(1, dtype=torch.long, device=device)
self._cache_pos.fill_(int(self.attention_mask.shape[-1]))

outputs = self.model(
    input_ids=step_ids, attention_mask=self._fixed_mask,
    past_key_values=self.past_key_values, cache_position=self._cache_pos,
    use_cache=True, return_dict=True,
)
```

Keep `max_cache_len` small. A 4096 mask costs more attention work than a 512 one and gives nothing back.

### 6. Decode in bigger chunks

`AudioStreamDecoder(chunk_frames=3)` pays a per-call cost that dominates. Measured on a 6.5 s utterance:

| chunk_frames | decode | first audio | RTF |
|---|---|---|---|
| 3 | 4.50 s | 0.45 s | 1.43 |
| **12** | 1.52 s | 0.92 s | **0.90** |
| 20 | 0.94 s | 1.30 s | 0.83 |
| 40 | 0.63 s | 2.40 s | 0.76 |

12 is the balance point we chose: under a second to first audio, still under 1.0 RTF.

### Also: warm up before announcing readiness

Compilation is charged to whoever triggers it. If a service reports ready after loading weights, the first user
waits tens of seconds. Run a real synthesis through the whole path first — and **use sentences of realistic
length**, because shapes follow text length. Warming with a five-character sentence left the first real request
paying 77 s.

## Where the time goes

Per audio frame, at 12.5 frames/s:

```
talker (1.7B)              31 ms
depth transformer (16x)    96 ms   -> 30 ms after condition 3
codec decode (CPU)        ~1.5 s per 6.5 s of audio at chunk_frames=12
```

Notice what is *not* on that list: the codec is not the bottleneck. Decoding a whole 6.5 s utterance in one call
takes 0.49 s on CPU and 0.46 s on this GPU — 1.1x. The 4.5 s that `chunk_frames=3` produced was per-call
overhead, not codec cost.

## Hardware context

For comparison, the same model on a rented 24 GB A10 with the codec on the GPU and **no** compile work applied:
first audio 698 ms, generation 8.76 s for 6.56 s of audio, decode 1.85 s, RTF 1.59. The larger card alone did not
cross 1.0 — the compile and chunking work above is what did.

That also means: if a 24 GB card is available, the arrangement is much less fragile, because talker and codec can
share the GPU. On 8 GB they cannot (4.34 + 6.6 GB), and the codec on CPU is part of the design.

## What is *not* claimed

- This is one machine, one model version, one prompt set. No averaging over seeds, no statistical treatment.
- The RTF numbers are single runs of a single sentence. Treat the third digit as noise.
- No audio-quality evaluation. int8 was judged by ear only, and an autoregressive sampler with temperature 0.8
  produces a different performance every run, so comparing two runs tells you almost nothing — fix the seed
  before comparing.
- No claim of novelty. Related work exists, and some of it targets this model:
  [vllm-omni PR #4221](https://github.com/vllm-project/vllm-omni/pull/4221) batches matmul and embedding gathers in
  the MOSS talker, and [Faster IndexTTS-2](https://arxiv.org/abs/2607.21042) accelerates a different streaming
  AR TTS model. What is here is shape and compilation hygiene, not a new kernel.

## Reproduction

`recipe.py` in this directory is self-contained: point `MOSS_TTS_DIR` and `MOSS_CODEC_DIR` at the two HuggingFace
repositories and run it. It prints first-audio latency, RTF, and the per-stage split, and it fakes `specialize_int`
and the mask patch exactly as described above.

## License

Apache-2.0 for the code, CC BY 4.0 for this text.

## Verifying it yourself in one command

```
MOSS_TTS_DIR=./moss-realtime MOSS_CODEC_DIR=./moss-codec REF_WAV=./reference.wav python recipe.py
```

Measured output on the machine these notes came from (RTX 3060 Ti 8 GB, codec on CPU):

```
torch 2.11.0+cu128  cuda True  codec on cpu  chunk_frames 12  max_cache 512
pass 1: first  19726 ms  audio  7.76 s  total  31.95 s  RTF  4.12  step  46.5 ms  decode  1.54 s
pass 2: first    939 ms  audio  7.76 s  total   7.21 s  RTF  0.93  step  46.9 ms  decode  1.58 s
```

Pass 1 pays for compilation; pass 2 is the steady state. If your pass 2 is not near 1.0, read the split that
recipe.py prints: `decode` large means condition 6, `step` large means conditions 2-5 did not take effect.

## Paper

`paper/` holds the same material as a 4-page technical report: `main.tex` is the source, `main.pdf` the compiled
output. Any LaTeX toolchain will rebuild it (`tectonic main.tex` needs no full TeX installation).
