# arXiv submission sheet

Paste these fields into the arXiv form. Nothing here needs editing except the code URL once the repository exists.

## Title

```
Making MOSS-TTS-Realtime Stream on an 8 GB Consumer GPU: Shape and Compilation Hygiene for Autoregressive Speech Synthesis
```

## Authors

```
Hongyun Du
```

## Abstract

```
Autoregressive text-to-speech models are usually deployed on server GPUs, where the audio codec and the acoustic model can share one device. On an 8 GB consumer card they cannot: the 1.7B talker of MOSS-TTS-Realtime occupies 4.34 GiB and its codec 6.6 GB, so the codec must run on the CPU. Under that arrangement the model measured RTF 2.08 (synthesis seconds per audio second), which cannot sustain streaming. We report six conditions that together bring the same utterance to RTF 0.90 with 0.92 s to first audio on a GeForce RTX 3060 Ti. Four of the six concern tensor shapes, not computation: a growing KV cache, a growing attention mask, and an integer argument that made the compiler re-specialise on every frame. The remaining two are a missing Triton installation that had silently disabled compilation, and a decoder chunk size whose per-call cost dominated decoding. Quantising the talker to int8 reduces weights to 2.82 GiB at the same steady-state speed, at the cost of a much longer first compilation. We describe the measurements, the wrong turns that preceded them, and what the numbers do not establish.
```

## Comments

```
4 pages, 2 tables. Code and reproduction script: <REPO URL>
```

## Category and license

```
Primary category  cs.SD (Sound)
Cross-list        eess.AS   (optional, adds reach with audio-signal audiences)
License           CC BY 4.0
```

## Files to upload

```
paper/main.tex        LaTeX source -- arXiv compiles it and shows a preview
```

Do not upload the PDF as the primary file; for cs categories the LaTeX source is compiled by arXiv, and a source
that fails to build is easier to fix before submitting than after.

## Order of operations

1. Create the public repository and confirm the URL resolves.
2. Replace `<REPO URL>` above and in the Comments field.
3. Start a new submission, paste the fields, upload main.tex.
4. Check the arXiv-generated PDF preview carefully -- it is built from source, not from our local build.
5. Submit.

## After submission

- The arXiv ID is citable immediately and is indexed by Google Scholar within days.
- Citation counts start at zero. What they measure is how useful others find the work, so the repository link in
  the Comments field matters more for early uptake than the paper's prose.
- Posting the repository to r/LocalLLaMA and to the model's Hugging Face discussion is what puts it in front of
  the people who would actually cite it.
