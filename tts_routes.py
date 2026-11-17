"""Choose an authorized Gemini project within app-owned daily request budgets."""

from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3


def is_key_env(name):
    return isinstance(name, str) and bool(re.fullmatch(r"(?:GEMINI|GOOGLE|GOOGLE_AI)_API_KEY[0-9]*", name))


class TTSRoutes:
    def __init__(self, config):
        config = Path(config).resolve()
        self.caps = json.loads(config.read_text(encoding="utf-8-sig"))
        if not isinstance(self.caps, dict) or not self.caps or any(
            not is_key_env(name) or type(limit) is not int or not 0 <= limit <= 1_000_000
            for name, limit in self.caps.items()
        ):
            raise ValueError("TTS routes must be a JSON object mapping Gemini key variable names to daily request caps.")
        self.ledger = config.with_suffix(".sqlite3")

    def reserve(self):
        day = datetime.now(timezone.utc).date().isoformat()
        keys = {name: os.environ.get(name, "").strip() for name in self.caps}
        present = [key for name, key in keys.items() if key and self.caps[name]]
        if len(present) != len(set(present)):
            raise ValueError("Two enabled TTS routes point to the same key. Give each project one route.")
        # ponytail: shared local ledger only; counters do not measure requests made by other apps.
        try:
            with closing(sqlite3.connect(self.ledger)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("""
                    CREATE TABLE IF NOT EXISTS usage (
                        day TEXT NOT NULL, route TEXT NOT NULL, requests INTEGER NOT NULL,
                        PRIMARY KEY (day, route)
                    )
                """)
                for name, cap in self.caps.items():
                    if not keys[name] or not cap:
                        continue
                    row = db.execute("SELECT requests FROM usage WHERE day = ? AND route = ?",
                                     (day, name)).fetchone()
                    used = row[0] if row else 0
                    if used < cap:
                        db.execute("""
                            INSERT INTO usage VALUES (?, ?, 1)
                            ON CONFLICT(day, route) DO UPDATE SET requests = requests + 1
                        """, (day, name))
                        print(f"TTS budget route {name}: {used + 1}/{cap} requests reserved today (UTC).", flush=True)
                        return keys[name]
        except sqlite3.Error as exc:
            raise RuntimeError("Could not update the local TTS budget ledger. No TTS request was sent.") from exc
        raise RuntimeError("No TTS route has an available key and remaining local budget for today (UTC).")
