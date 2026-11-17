import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import wave

from PIL import Image, ImageFont

import lesson


EXAMPLE = Path(lesson.__file__).with_name("example.json")


class LessonTests(unittest.TestCase):
    def test_offline_video_switches_speaker_on_the_measured_frame(self):
        font = ImageFont.truetype(lesson.FONT, 36)
        missing = bytes(font.getmask(chr(0x10FFFF)))
        self.assertTrue(all(bytes(font.getmask(c)) != missing for c in "äöüßñáéíóú¿¡€"))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "output"
            script = Path(tmp) / "windows-script.json"
            script.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8-sig")
            subprocess.check_call([sys.executable, lesson.__file__, "--script", str(script),
                                   "--output", str(folder), "--demo"])
            video = folder / "video.mp4"
            self.assertFalse((folder / "script.json").read_bytes().startswith(b"\xef\xbb\xbf"))
            turns = json.loads((folder / "timeline.json").read_text(encoding="utf-8"))
            self.assertEqual([turn["start_frame"] for turn in turns], [0, 44, 90, 133])
            frames = turns[-1]["next_frame"]
            with wave.open(str(folder / "audio.wav")) as audio:
                self.assertEqual(audio.getnframes(), frames * lesson.SAMPLES_PER_FRAME)
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "stream=codec_name,nb_frames",
                "-of", "json", str(video),
            ]))
            self.assertEqual([s["codec_name"] for s in probe["streams"]], ["h264", "aac"])
            self.assertEqual(int(probe["streams"][0]["nb_frames"]), frames)
            srt = (folder / "subtitles.srt").read_text(encoding="utf-8")
            self.assertIn("00:00:01,467 --> 00:00:02,485\nGerne. Möchten Sie Zucker?", srt)
            for frame, active in [(0, 0), (43, None), (44, 1)]:
                png = subprocess.check_output([
                    "ffmpeg", "-loglevel", "error", "-i", str(video),
                    "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", "-threads", "1",
                    "-f", "image2pipe", "-vcodec", "png", "-",
                ])
                with Image.open(io.BytesIO(png)) as img:
                    badge_red = [img.getpixel((x, 553))[1] < 110 for x in (252, 892)]
                self.assertEqual(badge_red, [active == 0, active == 1])

    def test_one_gemini_speech_request_for_all_turns_and_cache(self):
        story = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        content = {"type": "audio", "data": base64.b64encode(lesson.tone_wav("anna")).decode()}
        result = {"steps": [{"type": "model_output", "content": [content]}]}
        with tempfile.TemporaryDirectory() as tmp:
            for multi in (True, False):
                if not multi:
                    for turn in story["turns"]:
                        turn["speaker"] = "martin"
                with patch.object(lesson, "urlopen", return_value=io.BytesIO(json.dumps(result).encode())) as request:
                    args = (story, Path(tmp), "fake-key", lesson.TTS_MODEL)
                    self.assertEqual(lesson.speech_track(*args), lesson.speech_track(*args))
                    self.assertEqual(request.call_count, 1)
                    body = json.loads(request.call_args.args[0].data)
                    parts = body["input"][0]["content"]
                    self.assertEqual([part["text"] for part in parts], [turn["text"] for turn in story["turns"]])
                    for part, turn in zip(parts, story["turns"]):
                        speech = part["annotations"][0]
                        self.assertEqual(speech.get("speaker"), turn["speaker"] if multi else None)
                        self.assertIn(turn["delivery"], speech["style"])
                    expected = {"speakers": [{"speaker": "anna", "voice": "Kore"},
                                              {"speaker": "ben", "voice": "Puck"}]} if multi else [{"voice": "Charon"}]
                    self.assertEqual(body["generation_config"]["speech_config"], expected)
                    self.assertFalse(body["store"])

    def test_add_pauses_preserves_recording_and_moves_later_intervals(self):
        rate = lesson.RATE
        pcm = b"\x01\x02" * (rate * 2)
        intervals = [(0, rate), (rate + rate // 10, rate * 2)]
        rendered, adjusted = lesson.add_pauses(pcm, intervals, [{"pause_ms": 400}, {"pause_ms": 0}])
        self.assertEqual(adjusted, [(0, rate), (rate * 14 // 10, rate * 23 // 10)])
        self.assertEqual(rendered.replace(b"\x00\x00", b""), pcm)

    def test_openrouter_writes_script_but_gemini_keeps_voice_and_key(self):
        story = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        script = {"choices": [{"message": {"content": json.dumps(story)}}]}
        audio = {"steps": [{"type": "model_output", "content": [{
            "type": "audio", "data": base64.b64encode(lesson.tone_wav("anna")).decode(),
        }]}]}
        free_model = "openai/gpt-oss-20b:free"
        responses = [io.BytesIO(json.dumps(result).encode("utf-8")) for result in (script, audio)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            args = ["lesson.py", "--topic", "At the café", "--mode", "dialogue", "--turns", "4",
                    "--openrouter-text-model", free_model, "--gemini-key-env", "GOOGLE_AI_API_KEY2",
                    "--output", str(output)]
            env = {"OPENROUTER_API_KEY": "fake-router", "GOOGLE_AI_API_KEY2": "fake-google"}
            with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", args), \
                    patch.object(lesson.shutil, "which", return_value="ffmpeg"), \
                    patch.object(lesson, "urlopen", side_effect=responses) as call, \
                    patch.object(lesson, "render", side_effect=lambda s, o, k, m, _, routes=None:
                                 lesson.speech_track(s, o, k, m, routes=routes)[0]):
                lesson.main()
            self.assertEqual(call.call_count, 2)
            writer, speaker = [invocation.args[0] for invocation in call.call_args_list]
            self.assertEqual(writer.full_url, lesson.OPENROUTER_API)
            self.assertEqual(writer.get_header("Authorization"), "Bearer fake-router")
            self.assertIsNone(writer.get_header("X-goog-api-key"))
            written = json.loads(writer.data)
            self.assertEqual(written["model"], free_model)
            self.assertEqual(written["provider"], {"require_parameters": True})
            self.assertEqual(written["response_format"]["type"], "json_schema")
            self.assertNotIn("models", written)
            self.assertEqual(speaker.full_url, lesson.API)
            self.assertEqual(speaker.get_header("X-goog-api-key"), "fake-google")
            self.assertIsNone(speaker.get_header("Authorization"))
            spoken = json.loads(speaker.data)
            self.assertEqual(spoken["model"], lesson.TTS_MODEL)
            self.assertEqual(spoken["generation_config"]["speech_config"]["speakers"],
                             [{"speaker": "anna", "voice": "Kore"}, {"speaker": "ben", "voice": "Puck"}])

    def test_openrouter_rejects_paid_or_automatic_model_ids_before_http(self):
        with patch.object(lesson, "urlopen") as call:
            for model in ("openai/gpt-oss-20b", "openrouter/free"):
                with self.assertRaisesRegex(ValueError, "exact OpenRouter text model"):
                    lesson.openrouter_text("fake", model, "Write a script.")
        call.assert_not_called()

    def test_quota_message_preserves_useful_fields_without_retrying(self):
        payloads = [
            ({"details": [{"retryDelay": "42.4s"}, {"violations": [
                {"quotaId": "RequestsPerMinutePerProjectPerModel-FreeTier", "quotaValue": "3"},
            ]}]}, "3 requests per minute", "43s"),
            ({"message": "limit: 10 requests per day on Free Tier. Please retry in 17h35m40s. Private."},
             "10 requests per day", "17h 35m 40s"),
        ]
        for error, limit, delay in payloads:
            response = HTTPError(lesson.API, 429, "Quota", {},
                                 io.BytesIO(json.dumps({"error": error}).encode()))
            with patch.object(lesson, "urlopen", side_effect=response) as request:
                with self.assertRaises(RuntimeError) as problem:
                    lesson.gemini("fake-key", "audio", model=lesson.TTS_MODEL)
            message = str(problem.exception)
            for expected in (limit, delay, lesson.TTS_MODEL, "same TTS model", "same --output"):
                self.assertIn(expected, message)
            self.assertNotIn(lesson.ALT_TTS_MODEL, message)
            self.assertNotIn("Private", message)
            self.assertNotIn("fake-key", message)
            self.assertEqual(request.call_count, 1)

    def test_three_speaker_script_is_rejected_before_synthesis(self):
        story = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        story["turns"][2]["speaker"] = "martin"
        with self.assertRaisesRegex(ValueError, "at most two speakers"):
            lesson.validate_story(story)


if __name__ == "__main__":
    unittest.main()
