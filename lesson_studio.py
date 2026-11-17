"""Local browser UI for turning a pasted lesson ticket into a video."""

import argparse
import hashlib
from hmac import compare_digest
import json
import os
from pathlib import Path
import re
from secrets import token_urlsafe
import shutil
import subprocess
from threading import Lock
import wave

from flask import Flask, abort, render_template, request, send_from_directory

import lesson
from tts_routes import TTSRoutes, is_key_env


HERE = Path(__file__).resolve().parent
DOWNLOADS = {"video.mp4", "audio.wav", "subtitles.srt", "script.json"}


def read_ticket(text):
    text = text.strip().removeprefix("\ufeff").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n?```", text, flags=re.I | re.S)
    if fenced:
        text = fenced[1]
    ticket = json.loads(text)
    lesson.validate_story(ticket)
    return ticket


def create_app(output, *, key=None, model=lesson.TTS_MODEL, routes=None, demo=False):
    app = Flask(__name__, static_folder=None)
    app.config.update(TRUSTED_HOSTS=["127.0.0.1", "localhost"], MAX_CONTENT_LENGTH=128 * 1024)
    output = Path(output).resolve()
    token, running = token_urlsafe(32), Lock()
    prompt = (HERE / "lesson_prompt.txt").read_text(encoding="utf-8")
    example = (HERE / "example.json").read_text(encoding="utf-8")

    @app.route("/", methods=["GET", "POST"])
    def index():
        page = dict(prompt=prompt, example=example, token=token, demo=demo,
                    model=model, languages=lesson.LANGUAGES, levels=lesson.LEVELS,
                    text="", error=None, job=None, title=None)
        if request.method == "GET":
            return render_template("studio.html", **page)
        supplied = request.form.get("token", "")
        if not compare_digest(supplied.encode("utf-8"), token.encode("ascii")):
            abort(403)
        page["text"] = request.form.get("ticket", "")
        try:
            ticket = read_ticket(page["text"])
        except ValueError as exc:
            page["error"] = str(exc)
            return render_template("studio.html", **page), 400
        if not running.acquire(blocking=False):
            page["error"] = "A video is already rendering. Wait for it to finish, then try again."
            return render_template("studio.html", **page), 409
        try:
            identity = json.dumps([ticket, model, demo], sort_keys=True, ensure_ascii=False)
            job = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            folder = output / job
            if not all((folder / name).is_file() for name in DOWNLOADS):
                lesson.render(ticket, folder, key, model, demo, routes=routes)
            page.update(job=job, title=ticket["title"])
        except (OSError, ValueError, RuntimeError, EOFError, KeyError, wave.Error,
                subprocess.CalledProcessError) as exc:
            page["error"] = str(exc)
            return render_template("studio.html", **page), 422
        finally:
            running.release()
        return render_template("studio.html", **page)

    @app.get("/files/<job>/<name>")
    def file(job, name):
        if not re.fullmatch(r"[a-f0-9]{24}", job) or name not in DOWNLOADS:
            abort(404)
        return send_from_directory(output / job, name, as_attachment=request.args.get("download") == "1")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=Path("output/studio"))
    parser.add_argument("--tts-model", default=lesson.TTS_MODEL)
    parser.add_argument("--gemini-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--tts-routes", type=Path, metavar="FILE")
    parser.add_argument("--demo", action="store_true", help="Render offline test tones instead of live speech.")
    args = parser.parse_args()
    if not is_key_env(args.gemini_key_env):
        parser.error("--gemini-key-env must name a supported Gemini key variable.")
    if not shutil.which("ffmpeg"):
        parser.error("Install FFmpeg and put it on PATH.")
    key = os.environ.get(args.gemini_key_env)
    if not args.demo and not args.tts_routes and not key:
        parser.error(f"Set {args.gemini_key_env}, use --tts-routes FILE, or start with --demo.")
    try:
        routes = TTSRoutes(args.tts_routes) if args.tts_routes and not args.demo else None
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    create_app(args.output, key=key, model=args.tts_model, demo=args.demo, routes=routes).run(
        host="127.0.0.1", port=args.port, debug=False,
    )


if __name__ == "__main__":
    main()
