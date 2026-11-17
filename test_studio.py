import base64
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import lesson
from lesson_studio import create_app
from tts_routes import TTSRoutes


EXAMPLE = Path(lesson.__file__).with_name("example.json")


class StudioTests(unittest.TestCase):
    def test_ticket_form_validates_and_renders_a_fenced_demo(self):
        example = EXAMPLE.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            client = create_app(Path(tmp), demo=True).test_client()
            page = client.get("/").get_data(as_text=True)
            csrf = re.search(r'name="token" value="([^"]+)"', page)[1]
            self.assertIn("Copy prompt", page)
            self.assertEqual(client.post("/", data={"ticket": example}).status_code, 403)
            self.assertEqual(client.get("/", headers={"Host": "example.org"}).status_code, 400)
            invalid = client.post("/", data={"ticket": "invalid", "token": csrf})
            self.assertEqual(invalid.status_code, 400)
            self.assertIn('role="alert"', invalid.get_data(as_text=True))
            data = {"ticket": "```json\n" + example + "\n```", "token": csrf}
            result = client.post("/", data=data)
            self.assertEqual(result.status_code, 200)
            path = re.search(r'<video[^>]+src="([^"]+)"', result.get_data(as_text=True))[1]
            with client.get(path) as mp4:
                self.assertEqual(mp4.mimetype, "video/mp4")
                self.assertGreater(mp4.content_length, 1000)
            self.assertEqual(client.get(path.replace("video.mp4", "private.sqlite3")).status_code, 404)
            with patch.object(lesson, "render", side_effect=AssertionError("Repeated render")):
                self.assertEqual(client.post("/", data=data).status_code, 200)

    def test_budgets_move_in_order_and_share_a_ledger_across_instances(self):
        keys = {"GOOGLE_AI_API_KEY": "fake-one", "GOOGLE_AI_API_KEY2": "fake-two"}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, keys, clear=True):
            path = Path(tmp) / "tts-routes.json"
            path.write_text(json.dumps(dict.fromkeys(keys, 1)), encoding="utf-8")
            self.assertEqual(TTSRoutes(path).reserve(), "fake-one")
            self.assertEqual(TTSRoutes(path).reserve(), "fake-two")
            with self.assertRaisesRegex(RuntimeError, "remaining local budget"):
                TTSRoutes(path).reserve()
            disk = path.with_suffix(".sqlite3").read_bytes()
            self.assertNotIn(b"fake-one", disk)
            self.assertNotIn(b"fake-two", disk)

    def test_cache_and_429_do_not_reserve_another_key(self):
        story = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        for failure in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                wav = {"data": base64.b64encode(lesson.tone_wav("anna")).decode()}
                with patch("tts_routes.TTSRoutes.reserve", return_value="fake-route") as reserved, \
                        patch.object(lesson, "gemini", return_value=wav,
                                     side_effect=RuntimeError("HTTP 429") if failure else None) as gemini:
                    config = path / "tts-routes.json"
                    config.write_text('{"GOOGLE_AI_API_KEY":1,"GOOGLE_AI_API_KEY2":1}', encoding="utf-8")
                    routes = TTSRoutes(config)
                    args = (story, path, None, lesson.TTS_MODEL)
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                            lesson.speech_track(*args, routes=routes)
                    else:
                        lesson.speech_track(*args, routes=routes)
                        lesson.speech_track(*args, routes=routes)
                    reserved.assert_called_once()
                    gemini.assert_called_once()
                    self.assertEqual(gemini.call_args.args, ("fake-route", "audio"))
                    self.assertEqual(gemini.call_args.kwargs["model"], lesson.TTS_MODEL)


if __name__ == "__main__":
    unittest.main()
