# -*- coding: utf-8 -*-
import re
import json
import sqlite3
from pathlib import Path

# file names and constants
TXT_FILE = "chat.txt"
PARTICIPANTS_FILE = "participants.json"
DB_FILE = "line_chat.db"
CONV_TITLE = "家庭群"

# datetime regex
DATE_RE = re.compile(
    r'^\s*(\d{4})[./-](\d{1,2})[./-](\d{1,2})'
    r'(?:\s*[（(]?\s*(?:星期|週|周)\s*[一二三四五六日天]\s*[)）]?)?\s*$'
)
TIME_RE = re.compile(r'^\s*(\d{1,2}):(\d{2})\s+(.*)$')

# message type detection
KEYWORD_TYPES = {
    "圖片": "image",
    "貼圖": "sticker",
    "影片": "video",
    "語音訊息": "audio",
    "檔案": "file",
    "位置訊息": "location",
    "相簿": "album"
}

def detect_msg_type(content: str) -> str:
    c = (content or "").strip()
    if c in KEYWORD_TYPES:
        return KEYWORD_TYPES[c]
    if c.startswith("http://") or c.startswith("https://"):
        return "link"
    return "text"

# handling participants (longest prefix match)
def load_participants(path: Path):
    names = json.loads(path.read_text(encoding="utf-8"))
    return sorted(names, key=len, reverse=True)  # longest name first

def longest_prefix_match(candidates, text: str):
    for name in candidates:
        if text.startswith(name + " "):
            return name, text[len(name) + 1:]
    return None, text

# ---- SQLite schema ----
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS participants(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS conversations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT
);
CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER,
    sender_id INTEGER,
    dt TEXT,
    date TEXT,
    time TEXT,
    content TEXT,
    msg_type TEXT
);
"""

def ensure_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)
    conn.commit()

def get_or_create_participant(conn: sqlite3.Connection, name: str) -> int:
    conn.execute("INSERT OR IGNORE INTO participants(name) VALUES(?)", (name,))
    conn.commit()
    row = conn.execute("SELECT id FROM participants WHERE name=?", (name,)).fetchone()
    return row[0]

def get_or_create_conversation(conn: sqlite3.Connection, title: str = "default") -> int:
    row = conn.execute("SELECT id FROM conversations WHERE title=?", (title,)).fetchone()
    if row:
        return row[0]
    conn.execute("INSERT INTO conversations(title) VALUES(?)", (title,))
    conn.commit()
    return conn.execute("SELECT id FROM conversations WHERE title=?", (title,)).fetchone()[0]

def parse_file(txt_path: Path, participants_path: Path, db_path: Path, conversation_title: str):
    participants = load_participants(participants_path)
    conn = sqlite3.connect(str(db_path))
    ensure_schema(conn)
    conv_id = get_or_create_conversation(conn, conversation_title)

    name_to_id = {n: get_or_create_participant(conn, n) for n in participants}
    unknown_id = get_or_create_participant(conn, "Unknown")

    current_date = None
    last_msg_rowid = None
    total_msgs = 0
    total_days = set()

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n").strip()
            if not line:
                continue

            # 1) date line
            m_date = DATE_RE.match(line)
            if m_date:
                y, mo, d = m_date.groups()
                current_date = f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
                total_days.add(current_date)
                continue

            # 2) new message line
            m_time = TIME_RE.match(line)
            if m_time:
                hh, mm, tail = m_time.groups()
                time_str = f"{int(hh):02d}:{int(mm):02d}"
                sender_name, content = longest_prefix_match(participants, tail)
                sender_id = name_to_id.get(sender_name, unknown_id)

                msg_type = detect_msg_type(content)
                dt = f"{current_date} {time_str}" if current_date else None

                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO messages(conversation_id, sender_id, dt, date, time, content, msg_type)
                       VALUES(?,?,?,?,?,?,?)""",
                    (conv_id, sender_id, dt, current_date, time_str, content, msg_type)
                )
                conn.commit()
                last_msg_rowid = cur.lastrowid
                total_msgs += 1
                continue

            # 3) continuation line
            if last_msg_rowid is not None:
                cur = conn.cursor()
                old = cur.execute("SELECT content FROM messages WHERE id=?", (last_msg_rowid,)).fetchone()[0] or ""
                new = (old + "\n" + line) if old else line
                cur.execute("UPDATE messages SET content=? WHERE id=?", (new, last_msg_rowid))
                conn.commit()

    conn.close()
    return total_msgs, len(total_days)

if __name__ == "__main__":
    msgs, days = parse_file(
        txt_path=Path(TXT_FILE),
        participants_path=Path(PARTICIPANTS_FILE),
        db_path=Path(DB_FILE),
        conversation_title=CONV_TITLE
    )
    print(f"✅ 完成！寫入 {msgs} 則訊息，涵蓋 {days} 個日期 → {DB_FILE}")
