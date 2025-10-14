# Desktop RPA Learner MVP (Recorder + Retrieval Policy)
# ----------------------------------------------------
# Windows-only, Python 3.10+
#
# What you get in this single file:
# 1) Hotkeys to toggle TRAIN mode and RUN mode
# 2) TRAIN: record your demo (screenshots + the UI element you clicked + typed keys)
# 3) RUN: try to reproduce the task by finding similar UI elements via UIA first, OCR fallback next
# 4) SQLite memory to store "what worked" across sessions
#
# Quick start
# -----------
# 1) pip install -r requirements.txt
#    (if you don't have Tesseract OCR and want OCR fallback, install it and set TESSERACT_PATH below)
# 2) Run: python rpa_learner.py
# 3) Press Ctrl+Alt+T to start/stop TRAIN (you perform the steps once, clicks + typing will be recorded)
# 4) Press Ctrl+Alt+R to RUN the learned sequence
# 5) Press Ctrl+Alt+P to PAUSE/RESUME, Ctrl+Alt+S to STOP program
#
# Notes
# -----
# - Prefer UIA (Windows UI Automation) for stability. We only OCR when UIA fails.
# - This is a minimal MVP: safe defaults, clear logs, and no background internet.
# - Everything is saved under ./data/ with one SQLite DB and a folder of screenshots.

import os
import sys
import time
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
import mss                          # screen capture module
import numpy as np                  # array and image processing
import cv2                          # OpenCV for image processing
import pyautogui                    # GUI automation (mouse/keyboard)
from pynput import mouse, keyboard  # global mouse/keyboard hooks
from PIL import Image               # image processing (Pillow)
import uiautomation as uia          # Windows UI Automation
from rapidfuzz import fuzz          # fuzzy string matching
import pytesseract                  # OCR

# -------------- CONFIG -----------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "memory.sqlite3"
SNAP_DIR = DATA_DIR / "snaps"
SNAP_DIR.mkdir(exist_ok=True)

# Tesseract not in PATH so we set it here
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
TESSERACT_LANG = "eng+chi_tra"

# Safety & UX
SLEEP_AFTER_CLICK = 0.25
SLEEP_AFTER_TYPE = 0.05
PAUSE_BETWEEN_STEPS = 0.6

# Hotkeys - using string format for keyboard.add_hotkey()
HOTKEY_TRAIN = 'ctrl+alt+t'
HOTKEY_RUN = 'ctrl+alt+r'
HOTKEY_PAUSE = 'ctrl+alt+p'
HOTKEY_STOP = 'ctrl+alt+s'

# ------------- END CONFIG --------------

# Global state flags
STATE = {
    'training': False,
    'running': False,
    'paused': False,
    'stopping': False,
    'current_session': None,
    'pressed': set(),
}

sct = mss.mss()

# ---------- Storage (SQLite) ----------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  name TEXT
);

CREATE TABLE IF NOT EXISTS steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  order_idx INTEGER NOT NULL,
  action_type TEXT NOT NULL,      -- 'click' | 'type' | 'hotkey'
  text TEXT,                      -- typed text or target label
  x INTEGER, y INTEGER,           -- screen coords used in training
  bbox TEXT,                      -- JSON of [left, top, width, height] of UIA/OCR element (if any)
  uia TEXT,                       -- JSON blob of UIA props (Name, ControlType, AutomationId, RuntimeId)
  snap_path TEXT,                 -- screenshot path for this step
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);
"""

conn = sqlite3.connect(DB_PATH)
conn.executescript(SCHEMA_SQL)
conn.commit()

# ------------- UTILITIES ---------------

def now_str():
    return datetime.now(timezone.utc).isoformat()

# ---- Screenshot Capture ----
# Capture full screen with mss, return as BGR numpy array
# mss object (sct) has 4 channels (BGRA)
# sct is converted to BGR for OpenCV compatibility
def grab_screen() -> np.ndarray:
    img = np.array(sct.grab(sct.monitors[1]))       # 1 means monitor no. 1
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)    # remove alpha(transparency) channel

# ---- Screeenshot Saving ----
# Save BGR numpy array as JPEG with timestamped filename, return path
def save_snap(img: np.ndarray, prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    path = SNAP_DIR / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(path), img)
    return str(path)

# ---- Uiautomation utilities ----
# Get UIA element at screen coords (x,y)
def uia_at(x: int, y: int):
    try:
        elem = uia.ControlFromPoint(x, y)
        return elem
    except Exception:
        return None

# ---- UIA element properties ----
# Return a dict of useful properties, or None if elem is None or error
def uia_props(elem):
    if not elem:
        return None
    try:
        return {
            'Name': elem.Name,
            'ControlType': str(elem.ControlTypeName),
            'AutomationId': elem.AutomationId,
            'ClassName': elem.ClassName,
            'BoundingRectangle': list(elem.BoundingRectangle) if elem.BoundingRectangle else None,
            'RuntimeId': list(elem.GetRuntimeId()) if hasattr(elem, 'GetRuntimeId') else None,
        }
    except Exception:
        return None

# ---- UIA search (breadth-first) ----
# Find best matching element by Name (and optional ControlType) using rapidfuzz
def uia_find_best(root: uia.Control, target_text: str, control_type: str | None = None):
    """Breadth-first scan to find element with Name similar to target_text.
    control_type: optional ControlTypeName filter (e.g., 'Button', 'Edit').
    """
    if not root:
        root = uia.GetRootControl()
    best = None
    best_score = -1

    def visit(e: uia.Control):
        nonlocal best, best_score
        try:
            name = e.Name or ""
            ctype = str(e.ControlTypeName)
            if control_type and ctype != control_type:
                return
            score = fuzz.partial_ratio(target_text.lower(), name.lower()) if target_text else -1
            if score > best_score:
                best, best_score = e, score
        except Exception:
            pass

    try:
        for e, _ in uia.WalkControl(root):
            visit(e)
    except Exception:
        pass
    return best, best_score

# ---- OCR utilities (fallback) ----
# Extract text boxes with pytesseract, return list of dicts with text, conf, bbox
# conf means confidence (0-100)
# bbox means bounding box [left, top, width, height]
def ocr_boxes(img: np.ndarray):
    try:
        data = pytesseract.image_to_data(img, lang=TESSERACT_LANG, output_type=pytesseract.Output.DICT)
        boxes = []
        n = len(data['text'])
        for i in range(n):
            txt = data['text'][i]
            if not txt:
                continue
            conf = float(data['conf'][i]) if data['conf'][i] not in (None, '-1') else 0.0
            if conf < 40:  # filter low conf
                continue
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            boxes.append({'text': txt, 'conf': conf, 'bbox': [x, y, w, h]})
        return boxes
    except Exception:
        return []

# ---- OCR search (fallback) ----
# Find best matching text box by target_text using rapidfuzz
def ocr_find_best(img: np.ndarray, target_text: str):
    boxes = ocr_boxes(img)
    best = None
    best_score = -1
    for b in boxes:
        score = fuzz.partial_ratio(target_text.lower(), b['text'].lower())
        if score > best_score:
            best, best_score = b, score
    return best, best_score

# ---- Recorder -----
# Global recorder object to handle mouse/keyboard events and store steps in DB
class Recorder:
    def __init__(self):
        self.session_id = None
        self.order_idx = 0
        self.kb_listener = None
        self.ms_listener = None
        self.buffer_typed = []

    # Start a new training session
    def start_session(self):
        cur = conn.cursor()
        cur.execute("INSERT INTO sessions(ts,name) VALUES(?,?)", (now_str(), None))
        conn.commit()
        self.session_id = cur.lastrowid
        self.order_idx = 0
        print(f"[TRAIN] Started session #{self.session_id}")

    # End current training session
    def end_session(self):
        print(f"[TRAIN] Ended session #{self.session_id}")
        self.session_id = None
        self.order_idx = 0

    # Record a step in the current session
    def record_step(self, action_type: str, text: str | None, x: int | None, y: int | None, uia_info: dict | None, bbox: list | None, snap_path: str | None):
        if self.session_id is None:
            return
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO steps(session_id, order_idx, action_type, text, x, y, bbox, uia, snap_path)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                self.session_id,
                self.order_idx,
                action_type,
                text,
                x, y,
                json.dumps(bbox) if bbox else None,
                json.dumps(uia_info) if uia_info else None,
                snap_path,
            )
        )
        conn.commit()
        print(f"[TRAIN] step {self.order_idx}: {action_type} text={text} xy=({x},{y}) snap={snap_path}")
        self.order_idx += 1

    # Listen for mouse clicks to capture click events
    def on_click(self, x, y, button, pressed):
        if not STATE['training']:
            return
        if not pressed:
            return  # on release
        img = grab_screen()
        snap_path = save_snap(img, prefix="click")
        elem = uia_at(x, y)
        props = uia_props(elem)
        bbox = props.get('BoundingRectangle') if props else None
        self.record_step('click', None, int(x), int(y), props, bbox, snap_path)

    # Listen for key presses to capture typed text
    def on_press(self, key):
        # capture typed text for training (simplified; no IME handling)
        if not STATE['training']:
            return
        try:
            if isinstance(key, keyboard.KeyCode) and key.char is not None:
                self.buffer_typed.append(key.char)
            elif key == keyboard.Key.enter:
                self.buffer_typed.append('\n')
            elif key == keyboard.Key.space:
                self.buffer_typed.append(' ')
        except Exception:
            pass

    # On key release, if Enter or Tab, consider word completed and save
    def on_release(self, key):
        if not STATE['training']:
            return
        # flush buffer if we think a word was completed
        if key in (keyboard.Key.enter, keyboard.Key.tab):
            text = ''.join(self.buffer_typed).strip()
            self.buffer_typed.clear()
            if text:
                img = grab_screen()
                snap_path = save_snap(img, prefix="type")
                self.record_step('type', text, None, None, None, None, snap_path)

    # Start mouse and keyboard listeners
    def run(self):
        self.kb_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.ms_listener = mouse.Listener(on_click=self.on_click)
        self.kb_listener.start(); self.ms_listener.start()

recorder = Recorder()

# ---- Runner ----


def center_of_bbox(bbox):
    if not bbox:
        return None
    L, T, R, B = bbox if len(bbox) == 4 else [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
    x = int((L + R) / 2)
    y = int((T + B) / 2)
    return x, y


def click_at(x: int, y: int):
    pyautogui.moveTo(x, y, duration=0.15)
    pyautogui.click(); time.sleep(SLEEP_AFTER_CLICK)


def type_text(txt: str):
    pyautogui.typewrite(txt, interval=SLEEP_AFTER_TYPE)


def run_session(session_id: int):
    cur = conn.cursor()
    cur.execute("SELECT id, order_idx, action_type, text, x, y, bbox, uia, snap_path FROM steps WHERE session_id=? ORDER BY order_idx ASC", (session_id,))
    steps = cur.fetchall()
    if not steps:
        print("[RUN] No steps recorded for session", session_id)
        return

    for (step_id, order_idx, action_type, text, x, y, bbox_json, uia_json, snap_path) in steps:
        if STATE['stopping']:
            print("[RUN] Stopped by user")
            return
        while STATE['paused']:
            time.sleep(0.2)

        bbox = json.loads(bbox_json) if bbox_json else None
        uinfo = json.loads(uia_json) if uia_json else None

        if action_type == 'click':
            # Try UIA by remembered props (Name + ControlType); fallback to OCR by remembered Name
            target_name = (uinfo or {}).get('Name') if uinfo else None
            target_type = (uinfo or {}).get('ControlType') if uinfo else None

            # 1) UIA search (whole desktop)
            elem, score = (None, -1)
            if target_name:
                elem, score = uia_find_best(None, target_name, None)
            if elem and elem.BoundingRectangle:
                cx, cy = center_of_bbox(list(elem.BoundingRectangle))
                if cx and cy:
                    print(f"[RUN] UIA click '{target_name}' score={score} @ ({cx},{cy})")
                    click_at(cx, cy)
                    time.sleep(PAUSE_BETWEEN_STEPS)
                    continue

            # 2) OCR fallback on full screen
            img = grab_screen()
            if target_name:
                box, oscore = ocr_find_best(img, target_name)
                if box:
                    bx, by, bw, bh = box['bbox']
                    cx = int(bx + bw/2); cy = int(by + bh/2)
                    print(f"[RUN] OCR click '{target_name}' score={oscore} @ ({cx},{cy})")
                    click_at(cx, cy)
                    time.sleep(PAUSE_BETWEEN_STEPS)
                    continue

            # 3) Last resort: recorded coordinates
            if x is not None and y is not None:
                print(f"[RUN] Fallback click recorded coords @ ({x},{y})")
                click_at(int(x), int(y))
                time.sleep(PAUSE_BETWEEN_STEPS)
                continue

            print(f"[RUN][WARN] Could not resolve click target for step {order_idx}")

        elif action_type == 'type':
            print(f"[RUN] type: {text!r}")
            type_text(text or '')
            time.sleep(PAUSE_BETWEEN_STEPS)

        else:
            print(f"[RUN] Unknown action_type: {action_type}")


# ---------- Hotkey handler ----------

def describe_state():
    flags = []
    for k in ('training','running','paused'):
        if STATE[k]: flags.append(k)
    if not flags:
        flags = ['idle']
    return "+".join(flags)


def toggle_train():
    print("[HOTKEY] Ctrl+Alt+T pressed - toggling TRAIN mode")
    if STATE['training']:
        STATE['training'] = False
        recorder.end_session()
        print("[HOTKEY] TRAIN mode stopped")
    else:
        if STATE['running']:
            print("[HOTKEY] Cannot start TRAIN while RUNNING.")
            return
        STATE['training'] = True
        recorder.start_session()
        print("[HOTKEY] TRAIN mode started")


def do_run():
    print("[HOTKEY] Ctrl+Alt+R pressed - starting RUN mode")
    if STATE['training']:
        print("[HOTKEY] Stop TRAIN first.")
        return
    if STATE['running']:
        print("[HOTKEY] Already running…")
        return

    # pick latest session
    cur = conn.cursor()
    cur.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        print("[RUN] No sessions to run.")
        return
    sid = row[0]

    def _runner():
        try:
            STATE['running'] = True
            print(f"[RUN] Starting session #{sid}")
            run_session(sid)
        finally:
            STATE['running'] = False
            print("[RUN] Finished.")

    threading.Thread(target=_runner, daemon=True).start()


def toggle_pause():
    print("[HOTKEY] Ctrl+Alt+P pressed - toggling PAUSE mode")
    STATE['paused'] = not STATE['paused']
    print("[STATE]", describe_state())


def do_stop():
    print("[HOTKEY] Ctrl+Alt+S pressed - stopping program")
    STATE['stopping'] = True
    print("[STATE] stopping=true")
    print("[MAIN] Exiting program...")
    # Exit the program
    import sys
    sys.exit(0)


# Hotkey registration functions
def register_hotkeys():
    """Register all hotkeys using keyboard.add_hotkey()"""
    try:
        keyboard.add_hotkey(HOTKEY_TRAIN, toggle_train)
        keyboard.add_hotkey(HOTKEY_RUN, do_run)
        keyboard.add_hotkey(HOTKEY_PAUSE, toggle_pause)
        keyboard.add_hotkey(HOTKEY_STOP, do_stop)
        print("[HOTKEYS] All hotkeys registered successfully")
    except Exception as e:
        print(f"[HOTKEYS] Error registering hotkeys: {e}")


# -------------- Main ---------------

def main():
    print("\nRPA Learner MVP — Controls:")
    print("\n  Ctrl+Alt+T  Start/Stop TRAIN")
    print("\n  Ctrl+Alt+R  RUN latest session")
    print("\n  Ctrl+Alt+P  Pause/Resume while running")
    print("\n  Ctrl+Alt+S  STOP program\n")
    
    # Start the recorder (for capturing mouse clicks and typed text during training)
    recorder.run()
    
    # Register hotkeys
    register_hotkeys()
    
    # Keep the program running and wait for hotkeys
    try:
        print("[MAIN] Program running. Press Ctrl+Alt+S to stop.")
        keyboard.wait('esc')  # Wait for ESC key to exit, or use Ctrl+Alt+S hotkey
    except KeyboardInterrupt:
        print("\n[MAIN] Program interrupted by user")
    finally:
        print("[MAIN] Shutting down...")



if __name__ == '__main__':
    main()
