# Language Video Workflow

Make a language-learning video from a topic or an existing script. Python runs the entire pipeline. No n8n, web server, or video generation API.

## Implementation plan

1. Use Gemini to write a short story at the requested CEFR level, or load an existing JSON script. Validate the text, speaker IDs, and delivery directions.
2. Assign fixed voices and portraits to Anna, Ben, Maya, and Martin. Send **the entire lesson in one Gemini TTS request**, with delivery instructions for each turn. Cache the returned WAV.
3. Align the original script turns to the recording locally with Whisper. Keep the recorded speech and add silence only where a requested `pause_ms` is longer than the natural gap. Use the resulting timeline for the active portrait and caption.
4. Export an MP4 with burned-in captions, a separate SRT, and the audio and timings for reuse. Verify the renderer offline before running a real voice test.

## Run

Requires Python 3.10+ and FFmpeg with `libx264` on PATH. Install in a virtual environment if preferred.

```sh
python -m pip install -r requirements.txt
# Offline check. The audio is test tones, not spoken words.
python lesson.py --script example.json --demo --output output/demo
```

For real speech, install the local aligner in the same Python environment. These CPU wheels avoid a CUDA installation on Windows and Linux.

```sh
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install "stable-ts==2.19.1"
```

The first real render downloads the multilingual Whisper `base` model. Alignment then runs locally and does not use another API request. Set `GEMINI_API_KEY` in your shell or runtime secrets. One [Google AI Studio API key](https://aistudio.google.com/apikey) is enough for writing and speech. Don't put the key in the repo.

```sh
# Test real TTS without a separate story-writing request.
python lesson.py --script example.json --output output/cafe

# Generate a script and render it.
python lesson.py --topic "Buying a train ticket" --language de-DE --level A1 --mode auto --output output/train

# Narrator only.
python lesson.py --topic "A lost dog finds its way home" --language en-US --level A2 --mode narration --output output/dog
```

`--language` accepts `de-DE`, `en-US`, and `es-ES`. `--level` accepts A1–C2. `--mode` can be `auto`, `dialogue`, `narration`, or `mixed` (Martin plus one character). The default 12 speaking turns still cost one TTS request. A topic adds a separate writing request. [Gemini supports at most two voices in one multi-speaker request](https://ai.google.dev/gemini-api/docs/speech-generation#limitations), so scripts with a third speaker are rejected before TTS.

The default models are Gemini 3.5 Flash-Lite for writing and Gemini 3.8 Flash-Lite TTS for speech. If you hit a 429, the error shows the model and any reported limit or retry delay. The script does not retry or change the speech model on an API error. Check your [active Google AI Studio limits](https://ai.google.dev/gemini-api/docs/rate-limits) before resuming.

The output folder contains `video.mp4`, `audio.wav`, `subtitles.srt`, `script.json`, `timeline.json`, and a cached original `speech-*.wav`. To resume, run with `--script path/to/script.json`, the **same output folder**, and the **same TTS model**. A matching cached WAV costs no new TTS request. Changing the spoken text, voices, or delivery regenerates the whole take. Old per-turn `clips/` caches are not reused by the new path. JSON input supports UTF-8 with or without a BOM. Script output is always UTF-8 without a BOM.

Portraits and the reusable backdrop are in `assets/`. The renderer uses still portraits with speaker highlights. It doesn't generate new imagery for each topic.

## Optional free script writer

Set `OPENROUTER_API_KEY` in your environment, then pass the exact ID of an [OpenRouter free text model](https://openrouter.ai/collections/free-models) that supports [structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).

```sh
python lesson.py --topic "Buying a train ticket" --language de-DE --level A1 --openrouter-text-model openai/gpt-oss-20b:free --output output/train
```

OpenRouter only receives the story prompt. Speech still uses the same Gemini TTS model, voice IDs, and single-request audio path. No OpenRouter account or key is needed with `--script`. To use one Gemini project, set `GEMINI_API_KEY` or select a different environment variable with `--gemini-key-env GOOGLE_AI_API_KEY2`.

## Lesson Studio

```sh
python lesson_studio.py --demo
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Choose the language and level, optionally enter a topic, and copy the prompt into any AI chat. A blank topic tells the AI to choose one. Paste the resulting JSON into the ticket field and render. The browser previews the video and offers the MP4, audio, subtitles, and ticket for download. The prompt is also available in [lesson_prompt.txt](lesson_prompt.txt).

Use **Load example** to test without a chat AI. Demo mode uses test tones. After installing the local aligner and setting a Gemini key, restart with `python lesson_studio.py` for real voices. The studio listens only on your machine and never asks you to paste credentials into the browser.

## Optional TTS project budgets

Copy [tts-routes.example.json](tts-routes.example.json) to `tts-routes.json`. Keep **only environment variable names and daily request counts** in the file. Set the matching keys outside the repo. Use one entry per independently authorized project, since two keys for the same Gemini project share Google's quota.

```sh
python lesson_studio.py --tts-routes tts-routes.json
# Or use the same route budgets from the CLI.
python lesson.py --script example.json --tts-routes tts-routes.json --output output/cafe
```

The app uses routes in file order and moves to the next one when it reaches an app-set daily request cap. A cap of zero disables a route. A local SQLite file next to the config tracks attempts across restarts and resets the count by UTC date. Cache hits use no budget. The counts only cover this app and are not a spending limit or a Google quota meter. If Google itself returns 429, the app reports it and stops. Every route uses the same Gemini speech model and voice IDs.

## Check

```sh
python -m unittest -v
```

The tests need FFmpeg and ffprobe. They check video timing, speaker changes, Windows JSON compatibility, provider routing, studio renders, budget handoff, and caching without live API requests.
