"""SQLite persistence for detection history and settings."""
import json
import sqlite3
import threading
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  input_type TEXT NOT NULL,
  filename TEXT,
  status TEXT NOT NULL,
  total_objects INTEGER DEFAULT 0,
  avg_confidence REAL,
  max_confidence REAL,
  processing_ms INTEGER,
  summary TEXT,
  original TEXT, enhanced TEXT, result TEXT,
  error TEXT);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
"""


def display_id(i):
    return f"AV-{i:05d}"


class Store:
    def __init__(self, path):
        self.path = str(path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    # ---- detections ----------------------------------------------------
    def add(self, **f):
        f.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        f["summary"] = json.dumps(f.get("summary") or {})
        cols = ",".join(f)
        with self._lock, self._conn() as c:
            cur = c.execute(f"INSERT INTO detections({cols}) VALUES({','.join('?' * len(f))})", list(f.values()))
            return self.get(cur.lastrowid, c)

    def get(self, rid, conn=None):
        c = conn or self._conn()
        row = c.execute("SELECT * FROM detections WHERE id=?", (rid,)).fetchone()
        return self._row(row) if row else None

    def list(self, limit=None, input_type=None):
        q, args = "SELECT * FROM detections", []
        if input_type:
            q += " WHERE input_type=?"; args.append(input_type)
        q += " ORDER BY id DESC"
        if limit:
            q += " LIMIT ?"; args.append(limit)
        with self._conn() as c:
            return [self._row(r) for r in c.execute(q, args)]

    def delete(self, rid):
        with self._lock, self._conn() as c:
            row = c.execute("SELECT * FROM detections WHERE id=?", (rid,)).fetchone()
            c.execute("DELETE FROM detections WHERE id=?", (rid,))
        return self._row(row) if row else None

    def clear(self):
        with self._lock, self._conn() as c:
            rows = [self._row(r) for r in c.execute("SELECT * FROM detections")]
            c.execute("DELETE FROM detections")
        return rows

    @staticmethod
    def _row(r):
        d = dict(r)
        d["summary"] = json.loads(d["summary"] or "{}")
        d["display_id"] = display_id(d["id"])
        return d

    # ---- settings ------------------------------------------------------
    def settings(self):
        s = dict(config.DEFAULT_SETTINGS)
        with self._conn() as c:
            for r in c.execute("SELECT key,value FROM settings"):
                if r["key"] in s:
                    s[r["key"]] = json.loads(r["value"])
        return s

    def save_settings(self, new):
        with self._lock, self._conn() as c:
            for k, v in new.items():
                c.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (k, json.dumps(v)))
        return self.settings()

    # ---- aggregates (computed from real rows only) ------------------------
    def stats(self):
        rows = [r for r in self.list() if r["status"] == "completed"]
        by_class, hist, daily, types = {}, [0] * 10, {}, {"image": 0, "video": 0, "live": 0}
        conf_sum = conf_n = 0
        for r in rows:
            types[r["input_type"]] = types.get(r["input_type"], 0) + 1
            day = r["created_at"][:10]
            daily[day] = daily.get(day, 0) + r["total_objects"]
            s = r["summary"]
            for i, n in enumerate(s.get("hist", [])):
                hist[i] += n
            for o in s.get("objects", []):
                e = by_class.setdefault(o["name"], {"name": o["name"], "count": 0, "conf_sum": 0.0})
                e["count"] += o["count"]; e["conf_sum"] += o["avg_confidence"] * o["count"]
                conf_sum += o["avg_confidence"] * o["count"]; conf_n += o["count"]
        classes = sorted(({"name": e["name"], "count": e["count"],
                           "avg_confidence": round(e["conf_sum"] / e["count"], 4)} for e in by_class.values()),
                         key=lambda e: -e["count"])
        return {
            "total_detections": sum(r["total_objects"] for r in rows),
            "images_analyzed": types.get("image", 0),
            "videos_analyzed": types.get("video", 0),
            "average_confidence": round(conf_sum / conf_n, 4) if conf_n else None,
            "most_detected": classes[0] if classes else None,
            "classes": classes,
            "confidence_histogram": hist,
            "daily": [{"date": d, "count": c} for d, c in sorted(daily.items())[-14:]],
            "failed": sum(1 for r in self.list() if r["status"] == "failed"),
        }
