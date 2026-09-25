from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS media (
    media_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    media_type TEXT NOT NULL CHECK(media_type IN ('image','video')),
    file_path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    original_size INTEGER NOT NULL,
    compressed_size INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    duration REAL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    processing_status TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_media_session_hash ON media(session_id, sha256);
CREATE INDEX IF NOT EXISTS idx_media_expires ON media(expires_at);
CREATE INDEX IF NOT EXISTS idx_media_session ON media(session_id);

CREATE TABLE IF NOT EXISTS analysis_results (
    analysis_id TEXT PRIMARY KEY,
    media_id TEXT NOT NULL,
    injury_labels TEXT NOT NULL,
    confidence REAL NOT NULL,
    ood_score REAL NOT NULL,
    disagreement REAL NOT NULL DEFAULT 0,
    severity TEXT NOT NULL,
    location TEXT,
    analysis_status TEXT NOT NULL,
    warnings TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(media_id) REFERENCES media(media_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_analysis_media ON analysis_results(media_id);

CREATE TABLE IF NOT EXISTS context_answers (
    context_id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL UNIQUE,
    mechanism TEXT,
    time_since_injury TEXT,
    bleeding_status TEXT,
    pain_level TEXT,
    movement_limitation TEXT,
    red_flags TEXT NOT NULL,
    FOREIGN KEY(analysis_id) REFERENCES analysis_results(analysis_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS guidance (
    guidance_id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL UNIQUE,
    risk_level TEXT NOT NULL,
    action TEXT NOT NULL,
    warning TEXT NOT NULL,
    source_ids TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(analysis_id) REFERENCES analysis_results(analysis_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    version TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            # Migrate databases created by earlier project versions.
            cols = {row[1] for row in conn.execute("PRAGMA table_info(analysis_results)").fetchall()}
            if "disagreement" not in cols:
                conn.execute("ALTER TABLE analysis_results ADD COLUMN disagreement REAL NOT NULL DEFAULT 0")
            media_sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='media'").fetchone()
            if media_sql and "sha256 TEXT NOT NULL UNIQUE" in (media_sql[0] or ""):
                conn.execute("PRAGMA foreign_keys=OFF")
                conn.execute("""CREATE TABLE media_new (
                    media_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, media_type TEXT NOT NULL CHECK(media_type IN ('image','video')),
                    file_path TEXT NOT NULL, mime_type TEXT NOT NULL, original_size INTEGER NOT NULL, compressed_size INTEGER NOT NULL,
                    width INTEGER, height INTEGER, duration REAL, sha256 TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
                    processing_status TEXT NOT NULL, FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )""")
                conn.execute("""INSERT INTO media_new(media_id,session_id,media_type,file_path,mime_type,original_size,compressed_size,width,height,duration,sha256,created_at,expires_at,processing_status)
                               SELECT media_id,session_id,media_type,file_path,mime_type,original_size,compressed_size,width,height,duration,sha256,created_at,expires_at,processing_status FROM media""")
                conn.execute("DROP TABLE media")
                conn.execute("ALTER TABLE media_new RENAME TO media")
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_media_session_hash ON media(session_id, sha256)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_media_expires ON media(expires_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_media_session ON media(session_id)")
                conn.execute("PRAGMA foreign_keys=ON")

    def create_user(self, user_id: str) -> None:
        with self.connect() as c:
            c.execute("INSERT OR IGNORE INTO users(user_id, created_at) VALUES(?, ?)", (user_id, utc_now()))

    def create_session(self, session_id: str, user_id: str | None) -> None:
        with self.connect() as c:
            if user_id is not None:
                c.execute("INSERT OR IGNORE INTO users(user_id, created_at) VALUES(?, ?)", (user_id, utc_now()))
            c.execute(
                "INSERT INTO sessions(session_id,user_id,status,created_at) VALUES(?,?,?,?)",
                (session_id, user_id, "ACTIVE", utc_now()),
            )

    def session_exists(self, session_id: str) -> bool:
        with self.connect() as c:
            return c.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone() is not None

    def create_media(self, row: dict[str, Any]) -> None:
        with self.connect() as c:
            c.execute(
                """INSERT INTO media(
                    media_id,session_id,media_type,file_path,mime_type,original_size,compressed_size,
                    width,height,duration,sha256,created_at,expires_at,processing_status
                ) VALUES(
                    :media_id,:session_id,:media_type,:file_path,:mime_type,:original_size,:compressed_size,
                    :width,:height,:duration,:sha256,:created_at,:expires_at,:processing_status
                )""",
                row,
            )

    def find_media_by_hash(self, sha256: str, session_id: str | None = None) -> sqlite3.Row | None:
        query = "SELECT * FROM media WHERE sha256=?"
        args: list[Any] = [sha256]
        if session_id is not None:
            query += " AND session_id=?"
            args.append(session_id)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self.connect() as c:
            return c.execute(query, args).fetchone()

    def create_analysis(self, row: dict[str, Any]) -> None:
        with self.connect() as c:
            c.execute(
                """INSERT INTO analysis_results(
                    analysis_id,media_id,injury_labels,confidence,ood_score,disagreement,severity,
                    location,analysis_status,warnings,created_at
                ) VALUES(
                    :analysis_id,:media_id,:injury_labels,:confidence,:ood_score,:disagreement,:severity,
                    :location,:analysis_status,:warnings,:created_at
                )""",
                row,
            )

    def create_context(self, row: dict[str, Any]) -> None:
        with self.connect() as c:
            c.execute(
                """INSERT INTO context_answers(
                    context_id,analysis_id,mechanism,time_since_injury,bleeding_status,
                    pain_level,movement_limitation,red_flags
                ) VALUES(
                    :context_id,:analysis_id,:mechanism,:time_since_injury,:bleeding_status,
                    :pain_level,:movement_limitation,:red_flags
                )""",
                row,
            )

    def create_guidance(self, row: dict[str, Any]) -> None:
        with self.connect() as c:
            c.execute(
                """INSERT INTO guidance(
                    guidance_id,analysis_id,risk_level,action,warning,source_ids,created_at
                ) VALUES(
                    :guidance_id,:analysis_id,:risk_level,:action,:warning,:source_ids,:created_at
                )""",
                row,
            )


    def create_analysis_bundle(self, analysis: dict[str, Any], context: dict[str, Any], guidance: dict[str, Any]) -> None:
        """Atomically persist analysis, user context and guidance."""
        with self.connect() as c:
            c.execute(
                """INSERT INTO analysis_results(
                    analysis_id,media_id,injury_labels,confidence,ood_score,disagreement,severity,
                    location,analysis_status,warnings,created_at
                ) VALUES(
                    :analysis_id,:media_id,:injury_labels,:confidence,:ood_score,:disagreement,:severity,
                    :location,:analysis_status,:warnings,:created_at
                )""",
                analysis,
            )
            c.execute(
                """INSERT INTO context_answers(
                    context_id,analysis_id,mechanism,time_since_injury,bleeding_status,
                    pain_level,movement_limitation,red_flags
                ) VALUES(
                    :context_id,:analysis_id,:mechanism,:time_since_injury,:bleeding_status,
                    :pain_level,:movement_limitation,:red_flags
                )""",
                context,
            )
            c.execute(
                """INSERT INTO guidance(
                    guidance_id,analysis_id,risk_level,action,warning,source_ids,created_at
                ) VALUES(
                    :guidance_id,:analysis_id,:risk_level,:action,:warning,:source_ids,:created_at
                )""",
                guidance,
            )

    def save_knowledge_documents(self, docs: list[dict[str, str]]) -> None:
        with self.connect() as c:
            c.executemany(
                """INSERT OR REPLACE INTO knowledge_documents(
                    document_id,title,source,version,updated_at
                ) VALUES(?,?,?,?,?)""",
                [(d["document_id"], d["title"], d["source"], d["version"], d.get("updated_at") or utc_now()) for d in docs],
            )

    def recent_history(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(100, int(limit)))
        with self.connect() as c:
            rows = c.execute(
                """SELECT a.analysis_id, a.media_id, m.media_type, m.compressed_size, a.injury_labels,
                          a.confidence, a.ood_score, a.disagreement, a.severity, a.location,
                          a.analysis_status, a.warnings, a.created_at
                     FROM analysis_results a
                     JOIN media m ON m.media_id=a.media_id
                    WHERE m.session_id=?
                    ORDER BY a.created_at DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def storage_stats(self) -> dict[str, int]:
        with self.connect() as c:
            row = c.execute(
                """SELECT COUNT(*) AS files,
                          COALESCE(SUM(original_size), 0) AS original_bytes,
                          COALESCE(SUM(compressed_size), 0) AS compressed_bytes
                     FROM media"""
            ).fetchone()
            return {"files": int(row["files"]), "original_bytes": int(row["original_bytes"]), "compressed_bytes": int(row["compressed_bytes"])}

    def cleanup_expired_media(self) -> list[Path]:
        now = utc_now()
        with self.connect() as c:
            rows = c.execute("SELECT media_id,file_path FROM media WHERE expires_at <= ?", (now,)).fetchall()
            paths = [Path(r["file_path"]) for r in rows]
            for r in rows:
                c.execute("DELETE FROM media WHERE media_id=?", (r["media_id"],))
        return paths

    def count(self, table: str) -> int:
        allowed = {"users", "sessions", "media", "analysis_results", "context_answers", "guidance", "knowledge_documents"}
        if table not in allowed:
            raise ValueError("invalid table")
        with self.connect() as c:
            return int(c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
