"""MOSS-TTS-Realtime on a single consumer GPU: the six conditions, self-contained.

Point the two environment variables at the HuggingFace repositories, then run:

    MOSS_TTS_DIR=./moss-realtime MOSS_CODEC_DIR=./moss-codec python recipe.py

Optional:
    REF_WAV       reference clip for voice cloning (default: none, silence)
    TEXT          sentence to speak
    MAX_CACHE     StaticCache length, default 512 -- keep it small
    CHUNK_FRAMES  decoder chunk, default 12 -- the value that decides RTF
    CODEC_DEVICE  cuda or cpu; default: cuda when more than 10 GB is free, else cpu
    PASSES        how many utterances to run, default 2 (the second is the warm one)

It prints first-audio latency, total time, RTF, the steady per-step cost, and the
generation/decode split. The first pass pays compilation; the second is the number
you are usually after.
"""

from __future__ import annotations

import inspect
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

TTS_DIR = Path(os.environ["MOSS_TTS_DIR"])
CODEC_DIR = Path(os.environ["MOSS_CODEC_DIR"])
REF_WAV = os.environ.get("REF_WAV", "")
TEXT = os.environ.get("TEXT", "哎……下雨确实很烦人。不过今天就别出门了，窝在家里打打游戏也挺好的。")
MAX_CACHE = int(os.environ.get("MAX_CACHE", "512"))
CHUNK_FRAMES = int(os.environ.get("CHUNK_FRAMES", "12"))
PASSES = int(os.environ.get("PASSES", "2"))
SAMPLE_RATE = 24000
CHUNK_DURATION = 0.24


def free_vram_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.mem_get_info()[0] / 1024**3


CODEC_DEVICE = os.environ.get("CODEC_DEVICE") or ("cuda" if free_vram_gib() > 10 else "cpu")

# ---------------------------------------------------------------------------------------------
# Condition 2: stop dynamo from specialising on the incrementing gen_step. Without this the depth
# transformer is recompiled every frame and the compiled model is far slower than eager.
# ---------------------------------------------------------------------------------------------
torch._dynamo.config.specialize_int = False
torch.compiler.set_stance("default")     # condition 1: requires a working triton

# The model repository imports its own modules relatively, so they have to be loaded as a package.
_pkg = Path(".recipe_pkg") / "mossttsrealtime"
_pkg.mkdir(parents=True, exist_ok=True)
for _f in TTS_DIR.glob("*.py"):
    shutil.copy(_f, _pkg / _f.name)
(_pkg / "__init__.py").write_text("", encoding="utf-8")
sys.path.insert(0, str(Path(".recipe_pkg").resolve()))

import torchaudio                                                       # noqa: E402
from transformers import AutoModel, AutoTokenizer                       # noqa: E402

import mossttsrealtime.configuration_mossttsrealtime as mcfg            # noqa: E402

# Newer transformers configs expect rope_scaling on the local transformer config.
for _name in dir(mcfg):
    _cls = getattr(mcfg, _name)
    if isinstance(_cls, type) and not hasattr(_cls, "rope_scaling"):
        try:
            _cls.rope_scaling = {"rope_type": "linear", "factor": 1.0}
        except Exception:
            pass

import transformers.masking_utils as masking                            # noqa: E402

# The model calls create_causal_mask with input_embeds / cache_position; transformers 5 renamed the
# first and dropped the second. Adapt to whatever signature is installed.
_original_mask = masking.create_causal_mask
_mask_params = inspect.signature(_original_mask).parameters


def _compat_mask(*args, **kwargs):
    if "input_embeds" in kwargs and "inputs_embeds" in _mask_params:
        kwargs.setdefault("inputs_embeds", kwargs.pop("input_embeds"))
    if "cache_position" in kwargs and "cache_position" not in _mask_params:
        kwargs.pop("cache_position")
    return _original_mask(*args, **kwargs)


masking.create_causal_mask = _compat_mask

from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime            # noqa: E402
from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor  # noqa: E402
from mossttsrealtime.streaming_mossttsrealtime import (                          # noqa: E402
    AudioStreamDecoder,
    MossTTSRealtimeInference,
    MossTTSRealtimeStreamingSession,
)


def install_fixed_mask_step() -> None:
    """Conditions 4 and 5: a preallocated mask written in place, plus an explicit cache_position."""

    def step(self, text_token=None, temperature=0.8, top_p=0.6, top_k=30,
             do_sample=True, repetition_penalty=1.1, repetition_window=50):
        if self._last_audio_tokens is None or self.attention_mask is None:
            raise ValueError("You must call prefill() before step().")
        if self.is_finished:
            return self._last_audio_tokens

        batch = self._last_audio_tokens.shape[0]
        if text_token is None:
            tokens = [self.text_pad_id] * batch
        elif isinstance(text_token, torch.Tensor):
            tokens = text_token.detach().cpu().tolist()
        elif isinstance(text_token, (list, tuple, np.ndarray)):
            tokens = list(text_token)
        else:
            tokens = [int(text_token)]

        device = self._last_audio_tokens.device
        text_t = torch.tensor(tokens, device=device, dtype=torch.long)
        step_ids = torch.cat([text_t[:, None, None], self._last_audio_tokens.unsqueeze(1)], dim=2)

        if getattr(self, "_fixed_mask", None) is None:
            self._fixed_mask = torch.ones(1, MAX_CACHE, dtype=torch.bool, device=device)
            self._cache_pos = torch.zeros(1, dtype=torch.long, device=device)
        self._cache_pos.fill_(int(self.attention_mask.shape[-1]))

        outputs = self.model(
            input_ids=step_ids, attention_mask=self._fixed_mask,
            past_key_values=self.past_key_values, cache_position=self._cache_pos,
            use_cache=True, return_dict=True,
        )
        self.past_key_values = outputs.past_key_values
        self.attention_mask = torch.cat(
            [self.attention_mask, (~self._is_stopping).unsqueeze(-1)], dim=-1
        )
        hidden = outputs.last_hidden_state[:, -1:, :]
        history = torch.stack(self._generated_tokens, dim=1) if self._generated_tokens else None
        audio_tokens = self.generate_local_transformer(
            hidden_states=hidden, temperature=temperature, top_p=top_p, top_k=top_k,
            do_sample=do_sample, repetition_penalty=repetition_penalty,
            repetition_window=repetition_window, generated_tokens=history, gen_step=self._step_idx,
        )
        self._generated_tokens.append(audio_tokens)
        self._last_audio_tokens = audio_tokens
        self._is_stopping |= audio_tokens[:, 0] == self.audio_eos_token
        self._step_idx += 1
        return audio_tokens

    MossTTSRealtimeInference.step = step


def main() -> int:
    print(f"  torch {torch.__version__}  cuda {torch.cuda.is_available()}  "
          f"codec on {CODEC_DEVICE}  chunk_frames {CHUNK_FRAMES}  max_cache {MAX_CACHE}")
    from torch.utils._triton import has_triton
    if not has_triton():
        print("  WARNING: no triton -- torch.compile cannot run, measurements below are eager")

    install_fixed_mask_step()

    tokenizer = AutoTokenizer.from_pretrained(str(TTS_DIR), trust_remote_code=True)
    processor = MossTTSRealtimeProcessor(tokenizer)
    model = MossTTSRealtime.from_pretrained(
        str(TTS_DIR), attn_implementation="sdpa", torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0" if torch.cuda.is_available() else "cpu").eval()
    codec = AutoModel.from_pretrained(str(CODEC_DIR), trust_remote_code=True).eval()
    codec = codec.to(CODEC_DEVICE)

    if REF_WAV and Path(REF_WAV).exists():
        data, rate = sf.read(REF_WAV, always_2d=True)
        mono = torch.tensor(data.mean(axis=1), dtype=torch.float32)
        if rate != SAMPLE_RATE:
            mono = torchaudio.functional.resample(mono, rate, SAMPLE_RATE)
    else:
        mono = torch.zeros(SAMPLE_RATE, dtype=torch.float32)

    with torch.inference_mode():
        enc = codec.encode(mono.unsqueeze(0).to(CODEC_DEVICE), chunk_duration=CHUNK_DURATION)
    voice = enc["audio_codes"].cpu().numpy().squeeze(1)

    system_prompt = processor.make_ensemble(voice)
    prefix_ids = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n")
    prefix = np.full((len(prefix_ids), system_prompt.shape[1]),
                     fill_value=processor.audio_channel_pad, dtype=np.int64)
    prefix[:, 0] = prefix_ids
    input_ids = np.concatenate([system_prompt, prefix], axis=0)

    step_times: list[float] = []
    original_step = MossTTSRealtimeInference.step

    def timed_step(self, *a, **k):
        t = time.perf_counter()
        out = original_step(self, *a, **k)
        step_times.append((time.perf_counter() - t) * 1000)
        return out

    MossTTSRealtimeInference.step = timed_step

    from transformers.cache_utils import StaticCache

    for pass_no in range(1, PASSES + 1):
        torch.manual_seed(1234)                     # fix the sampler: temperature 0.8 diverges otherwise
        torch.cuda.manual_seed_all(1234)
        inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=MAX_CACHE)
        inferencer.reset_generation_state(keep_cache=False)
        session = MossTTSRealtimeStreamingSession(
            inferencer, processor, codec=codec, codec_sample_rate=SAMPLE_RATE,
            codec_encode_kwargs={"chunk_duration": CHUNK_DURATION},
            prefill_text_len=processor.delay_tokens_len,
            temperature=0.8, top_p=0.6, top_k=30, do_sample=True,
            repetition_penalty=1.1, repetition_window=50,
        )
        session.set_voice_prompt_tokens(voice)
        session.reset_turn(input_ids=input_ids, include_system_prompt=False, reset_cache=True)
        # Condition 4: attach before prefill, or the model replaces it with a DynamicCache.
        inferencer.past_key_values = StaticCache(
            config=model.config.language_config, max_batch_size=1,
            max_cache_len=MAX_CACHE, device=inferencer.device, dtype=torch.bfloat16,
        )
        decoder = AudioStreamDecoder(  # condition 6
            codec, chunk_frames=CHUNK_FRAMES, overlap_frames=0,
            decode_kwargs={"chunk_duration": -1}, device=torch.device(CODEC_DEVICE),
        )

        eos = int(getattr(inferencer, "audio_eos_token", 1026))
        codebook = int(getattr(codec, "codebook_size", 1024))
        pieces: list[np.ndarray] = []
        first: list[float] = []
        decode_s = [0.0]
        t0 = time.perf_counter()

        def take(frames):
            for frame in frames:
                f = frame[0] if frame.dim() == 3 else frame
                if f.dim() != 2 or f.numel() == 0:
                    continue
                eos_rows = (f[:, 0] == eos).nonzero(as_tuple=False)
                invalid = ((f < 0) | (f >= codebook)).any(dim=1)
                stop = int(eos_rows[0].item()) if eos_rows.numel() else None
                if invalid.any():
                    bad = int(invalid.nonzero(as_tuple=False)[0].item())
                    stop = bad if stop is None else min(stop, bad)
                if stop is not None:
                    f = f[:stop]
                if f.numel() == 0:
                    continue
                td = time.perf_counter()
                decoder.push_tokens(f.detach())
                for chunk in decoder.audio_chunks():
                    if chunk.numel():
                        if not first:
                            first.append(time.perf_counter() - t0)
                        pieces.append(chunk.detach().float().cpu().numpy().reshape(-1))
                decode_s[0] += time.perf_counter() - td

        step_times.clear()
        with torch.inference_mode(), codec.streaming(batch_size=1):
            for ch in TEXT:
                take(session.push_text(ch))
            take(session.end_text())
            while True:
                frames = session.drain(max_steps=1)
                if not frames:
                    break
                take(frames)
                if inferencer.is_finished:
                    break
            tail = decoder.flush()
            if tail is not None and tail.numel():
                pieces.append(tail.detach().float().cpu().numpy().reshape(-1))

        elapsed = time.perf_counter() - t0
        audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
        secs = len(audio) / SAMPLE_RATE
        tail_times = sorted(step_times[len(step_times) // 3:]) if step_times else [0.0]
        steady = tail_times[len(tail_times) // 2]
        print(f"  pass {pass_no}: first {first[0]*1000 if first else 0:6.0f} ms  audio {secs:5.2f} s  "
              f"total {elapsed:6.2f} s  RTF {elapsed/max(secs,1e-6):5.2f}  "
              f"step {steady:5.1f} ms  decode {decode_s[0]:5.2f} s")
        if pass_no == PASSES:
            out = Path(f"out_pass{pass_no}.wav")
            sf.write(out, audio, SAMPLE_RATE, subtype="PCM_16")
            print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
