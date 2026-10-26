# Video Stories

This repository extracts short, self-contained story clips (approximately 30 and 45 seconds) from longer broadcast videos. Two domain-specific pipelines are provided: a text-centric pipeline for news videos, and a multimodal pipeline for cricket videos.

## Data

Source material is organised under `data/` as 720p MP4 files grouped by topic.


| Corpus             | Directory              | Videos | Content                                                                                                                                                                           |
| ------------------ | ---------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| News — Middle East | `data/middle_east_war` | 5      | International news broadcasting videos on regional conflict, oil markets, and related diplomatic reporting.                                                                       |
| News — BRICS       | `data/brics`           | 4      | Summit and geopolitical coverage from outlets such as FRANCE 24, Al Jazeera, CNBC-TV18, and Bloomberg.                                                                            |
| Sports — Cricket   | `data/india_cricket`   | 3      | English-language T20 highlight videos: India vs New Zealand (Sanju Samson; Tilak Varma) and India vs USA (Men’s T20 World Cup 2026), from Sony LIV, Sony Sports Network, and ICC. |


Derived artefacts are written under `outputs/` (audio, transcripts, shot cuts, clip candidates, and exported stories). Sports intermediates additionally use `outputs/sports/`. Model weights are cached under `models/`.

## Solution Approach

Both pipelines share a common objective: propose candidate windows aligned to natural edit points, score them for editorial importance, suppress redundant overlaps, and export ranked short clips. They differ in how candidates are proposed and scored.

### News

The news pipeline is transcript-driven and runs locally with a GPU.

1. **Transcription.** Audio is extracted to 16 kHz mono WAV via FFmpeg. `nvidia/parakeet-tdt-0.6b-v3` produces word-level timestamps; sentences are formed at terminal punctuation while preserving abbreviations such as `U.S.`.
2. **Shot alignment.** FFmpeg scene detection (threshold 0.3) yields cut times. Sentence boundaries are snapped to a nearby cut only when the cut lies within 0.4 seconds and falls in the inter-sentence gap.
3. **Candidate generation.** From each sentence start, contiguous sentence spans are grown up to 30 and 45 seconds. Spans shorter than 12 seconds are discarded.
4. **Scoring.** Each candidate receives four normalised component scores, combined as:
  > importance = 0.35 × centrality + 0.20 × lead + 0.15 × density + 0.30 × clarity
  - **Centrality** — semantic similarity of the clip text to the full-video sentence embedding centroid (`sentence-transformers/all-MiniLM-L6-v2`).
  - **Lead** — preference for material in the first 25% of the video.
  - **Density** — zero-shot factuality via `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, with MMR diversification (λ = 0.7).
  - **Clarity** — zero-shot preference for standalone news statements over greetings or sign-offs.
5. **Selection and export.** Clips with importance below 0.5 are removed. Within each duration bucket, a lower-scoring clip that overlaps a retained clip by more than 0.2 (fractional overlap) is dropped. The top three clips per video and duration are cut to `outputs/stories/clips/`.

### Sports

The cricket pipeline is intended for Google Colab (T4 GPU). It proposes candidates from audio–visual event cues and scores them with a multimodal model.

1. **Structural edges.** Shot cuts (scene threshold 0.3) and commentary pauses (below −32 dB for at least 0.8 seconds) define window boundaries, reused from `outputs/sports/` when present.
2. **Event detection.** Soundtrack loudness peaks are retained when a 2-second stretch is at least 1.25× as loud as the surrounding 20 seconds. Among peaks within 20 seconds, the louder peak is kept. `nvidia/parakeet-tdt-0.6b-v3` transcribes the soundtrack; a peak is accepted only if an event word (e.g. four, six, boundary, wicket, catch, century) occurs between 8 seconds before and 12 seconds after the peak.
3. **Windows.** Each accepted peak yields one window capped at 30 seconds and one at 45 seconds, bounded by the nearest prior and subsequent cut or pause end. Windows shorter than 12 seconds are discarded.
4. **Multimodal scoring.** `Qwen/Qwen2.5-Omni-3B` (4-bit) scores each window from eight sampled frames plus commentary, constrained to a JSON schema. Importance is discrete: 0 (toss/music/replay intro), 0.3 (routine ball), 0.5 (boundary/edge), 0.65 (dismissal or landmark score), or 1.0 (match-deciding moment). Overlap-filtered and unfiltered sets are written under `outputs/sports/stories/{30s,45s}/`.

## Results

### News

Across nine source packages, the pipeline produced scored story JSON under `outputs/stories/` (157 files for BRICS; 334 for Middle East coverage prior to per-video top-k filtering). After importance, overlap, and top-3 selection, exported video clips were:


| Topic       | 30-second clips | 45-second clips |
| ----------- | --------------- | --------------- |
| BRICS       | 11              | 11              |
| Middle East | 15              | 14              |


Highest observed importance scores were approximately 0.77 (BRICS) and 0.84 (Middle East), typically corresponding to early lead clips and dense factual openers. Each story JSON records the time range, transcript text, component scores, and aggregate importance.

### Sports

For the three cricket highlight videos, the pipeline exported:


| Duration   | Story clips |
| ---------- | ----------- |
| 30 seconds | 37          |
| 45 seconds | 33          |


Scored moments are concentrated on dismissals, catches, and high-impact batting events (importance commonly 0.65), with filenames encoding source shorthand, time range, and score (e.g. `samson_313-336_0.65`). Metadata includes the triggering cue word, peak time, and model rationale.

### Artefact layout

```
outputs/
  audio/            # extracted WAVs (news)
  transcripts/      # word/sentence transcripts (news)
  shots/            # scene-cut times (news)
  clips/            # scored candidate windows (news)
  stories/          # story JSON + clips/{thirty,forty_five}_sec_clips/
  sports/
    shots/ pauses/ transcripts/
    stories/30s/ stories/45s/
```

## Setup

Python 3.12 is used in this project.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

