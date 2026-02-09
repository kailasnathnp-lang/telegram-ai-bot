import os
import io
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

import requests
from flask import Flask, request, abort
from openai import OpenAI

# Optional: Excel support
try:
    import pandas as pd
except Exception:
    pd = None

# ------------------- ENV -------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "").strip()

# Extra Telegram webhook protection (recommended)
TELEGRAM_SECRET_TOKEN = os.environ.get("TELEGRAM_SECRET_TOKEN", "").strip()

# Chat model (TEXT MODELS only)
MODEL = os.environ.get("MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct").strip()

# Whisper model (AUDIO ONLY)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3-turbo").strip()

TZ = ZoneInfo("Asia/Kolkata")
DB_PATH = os.environ.get("DB_PATH", "bot_memory.db").strip()

if not BOT_TOKEN or not GROQ_API_KEY or not WEBHOOK_SECRET:
    raise RuntimeError("Missing required env vars: BOT_TOKEN, GROQ_API_KEY, WEBHOOK_SECRET")

SYSTEM_PROMPT = """
You are Kailas's personal AI friend.
Be warm, casual, and human. Use simple words.
Keep replies short unless he asks for details.
Ask one helpful follow-up question when needed.
Remember facts he shares (preferences, goals, work, habits).
Use the provided current date/time for "today/tomorrow/yesterday".
If something is unclear, ask a short clarifying question.
""".strip()

# Groq using OpenAI-compatible client
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
app = Flask(__name__)

# ------------------- DB -------------------
def db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_memory (
                user_id TEXT PRIMARY KEY,
                memory TEXT DEFAULT ''
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                role TEXT,
                content TEXT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_notes (
                user_id TEXT,
                date TEXT,
                note TEXT,
                PRIMARY KEY (user_id, date)
            )
        """)
        conn.commit()

def get_memory(user_id: str) -> str:
    with db_conn() as conn:
        row = conn.execute("SELECT memory FROM user_memory WHERE user_id=?", (user_id,)).fetchone()
        return row[0] if row else ""

def set_memory(user_id: str, memory: str):
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO user_memory (user_id, memory)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET memory=excluded.memory
        """, (user_id, memory))
        conn.commit()

def save_history(user_id: str, role: str, content: str):
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        conn.commit()

def get_recent_history(user_id: str, limit: int = 12):
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT role, content FROM chat_history
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
    rows.reverse()
    return [{"role": r, "content": c} for r, c in rows]

def set_daily_note(user_id: str, date: str, note: str):
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO daily_notes (user_id, date, note)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, date) DO UPDATE SET note=excluded.note
        """, (user_id, date, note))
        conn.commit()

def get_daily_note(user_id: str, date: str) -> str:
    with db_conn() as conn:
        row = conn.execute("SELECT note FROM daily_notes WHERE user_id=? AND date=?", (user_id, date)).fetchone()
        return row[0] if row else ""

# ------------------- Telegram helpers -------------------
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

def tg_request(method: str, payload=None, timeout=25):
    url = f"{TG_API}/{method}"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def send_message(chat_id, text, reply_to_message_id=None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to_message_id:
        payload["reply_to_message_id"] = reply_to_message_id
    try:
        tg_request("sendMessage", payload=payload, timeout=15)
    except Exception:
        pass

def get_file_bytes(file_id: str) -> bytes:
    info = tg_request("getFile", payload={"file_id": file_id})
    file_path = info["result"]["file_path"]
    file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    r = requests.get(file_url, timeout=40)
    r.raise_for_status()
    return r.content

# ------------------- Extractors -------------------
def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import pdfplumber
    except Exception:
        return "⚠️ pdfplumber not installed. Install: pip install pdfplumber"

    out = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages[:20], start=1):
                text = page.extract_text() or ""
                if text.strip():
                    out.append(f"[Page {i}]\n{text.strip()}")
        return "\n\n".join(out).strip() or "⚠️ PDF opened but no text found (might be scanned image PDF)."
    except Exception:
        return "⚠️ Couldn’t read this PDF (corrupted/encrypted?)."

def extract_excel_text(file_bytes: bytes, filename: str) -> str:
    """
    Reads Excel/CSV into text.
    - For .xlsx/.xls: reads ALL sheets
    - For .csv: reads as one table
    """
    if pd is None:
        return "⚠️ Excel support missing. Install: pip install pandas openpyxl"

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
            df = df.fillna("")
            return "[CSV]\n" + df.to_string(index=False)

        # Excel
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        output = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df = df.fillna("")
            # limit huge sheets to avoid token overflow
            df_head = df.head(200)
            output.append(f"[Sheet: {sheet_name}]\n{df_head.to_string(index=False)}")

        return "\n\n".join(output).strip()

    except Exception as e:
        return f"⚠️ Failed to read Excel: {e}"

def ocr_image(image_bytes: bytes) -> str:
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return "⚠️ OCR not available. Install: pip install pillow pytesseract and install Tesseract."

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img)
        return text.strip() or "⚠️ OCR got no readable text."
    except Exception:
        return "⚠️ OCR failed on this image."

def transcribe_voice(audio_bytes: bytes) -> str:
    try:
        f = io.BytesIO(audio_bytes)
        f.name = "voice.ogg"
        res = client.audio.transcriptions.create(model=WHISPER_MODEL, file=f)
        return getattr(res, "text", "") or ""
    except Exception:
        return ""

# ------------------- Commands -------------------
def maybe_store_long_memory(user_text: str):
    lower = user_text.lower().strip()
    if lower.startswith("remember:"):
        return user_text.split(":", 1)[1].strip()
    if lower.startswith("remember that"):
        return user_text.split("that", 1)[1].strip()
    return None

def maybe_store_daily(user_text: str):
    lower = user_text.strip().lower()
    if lower.startswith("today:"):
        return user_text.split(":", 1)[1].strip()
    return None

# ------------------- LLM -------------------
def build_messages(user_id: str, user_text: str):
    now = datetime.now(TZ)
    today = now.strftime("%Y-%m-%d")

    time_context = f"Current date/time: {now.strftime('%A, %d %B %Y, %I:%M %p')} (Asia/Kolkata)."
    memory = get_memory(user_id)
    daily = get_daily_note(user_id, today)
    history = get_recent_history(user_id, limit=10)

    memory_block = f"User memory: {memory}" if memory else "User memory: (none yet)"
    daily_block = f"Today's note ({today}): {daily}" if daily else f"Today's note ({today}): (none)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": time_context},
        {"role": "system", "content": memory_block},
        {"role": "system", "content": daily_block},
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages

def chat_reply(user_id: str, user_text: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=build_messages(user_id, user_text),
        temperature=0.7,
        max_tokens=450,
    )
    return resp.choices[0].message.content.strip()

# ------------------- Security -------------------
def verify_telegram_secret():
    if not TELEGRAM_SECRET_TOKEN:
        return True
    header = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    return header == TELEGRAM_SECRET_TOKEN

# ------------------- Routes -------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post(f"/webhook/{WEBHOOK_SECRET}")
def webhook():
    if not verify_telegram_secret():
        abort(403)

    data = request.get_json(silent=True) or {}
    msg = data.get("message") or data.get("edited_message")
    if not msg:
        return "ok"

    chat_id = msg.get("chat", {}).get("id")
    user_id = str(msg.get("from", {}).get("id", chat_id or "unknown"))
    message_id = msg.get("message_id")

    now = datetime.now(TZ)
    today = now.strftime("%Y-%m-%d")

    # -------- TEXT --------
    if "text" in msg and msg["text"]:
        user_text = msg["text"].strip()
        lower = user_text.lower()

        daily_note = maybe_store_daily(user_text)
        if daily_note is not None:
            set_daily_note(user_id, today, daily_note)
            send_message(chat_id, f"✅ Saved for today ({today}).", reply_to_message_id=message_id)
            return "ok"

        if lower in ["today", "today?", "what is today", "what is today?"]:
            note = get_daily_note(user_id, today)
            send_message(chat_id, note if note else "📌 Nothing saved for today yet. Use: today: ...", reply_to_message_id=message_id)
            return "ok"

        mem_to_store = maybe_store_long_memory(user_text)
        if mem_to_store:
            existing = get_memory(user_id)
            new_mem = (existing + "\n" + mem_to_store).strip() if existing else mem_to_store
            set_memory(user_id, new_mem)
            send_message(chat_id, "✅ Got it — I’ll remember that.", reply_to_message_id=message_id)
            return "ok"

        try:
            save_history(user_id, "user", user_text)
            reply = chat_reply(user_id, user_text)
            save_history(user_id, "assistant", reply)
        except Exception:
            reply = "Hmm, I had a glitch. Try again?"
        send_message(chat_id, reply, reply_to_message_id=message_id)
        return "ok"

    # -------- VOICE --------
    if "voice" in msg:
        file_id = msg["voice"].get("file_id")
        if not file_id:
            send_message(chat_id, "I got your voice message, but couldn’t access it.", reply_to_message_id=message_id)
            return "ok"

        audio_bytes = get_file_bytes(file_id)
        text = transcribe_voice(audio_bytes)
        if not text.strip():
            send_message(chat_id, "I couldn’t transcribe that voice note.", reply_to_message_id=message_id)
            return "ok"

        user_text = f"(Voice transcript)\n{text.strip()}"
        try:
            save_history(user_id, "user", user_text)
            reply = chat_reply(user_id, user_text)
            save_history(user_id, "assistant", reply)
        except Exception:
            reply = "Voice handled, but AI reply failed. Try again?"
        send_message(chat_id, reply, reply_to_message_id=message_id)
        return "ok"

    # -------- DOCUMENTS (PDF/EXCEL) --------
    if "document" in msg:
        doc = msg["document"]
        file_id = doc.get("file_id")
        file_name = (doc.get("file_name") or "").lower()

        if not file_id:
            send_message(chat_id, "I got the file, but couldn’t access it.", reply_to_message_id=message_id)
            return "ok"

        file_bytes = get_file_bytes(file_id)

        if file_name.endswith(".pdf"):
            extracted = extract_pdf_text(file_bytes)
        elif file_name.endswith((".xlsx", ".xls", ".csv")):
            extracted = extract_excel_text(file_bytes, file_name)
        else:
            send_message(chat_id, "I support PDF, Excel (.xlsx/.xls), and CSV.", reply_to_message_id=message_id)
            return "ok"

        user_text = (
            "Here is data extracted from a file.\n\n"
            f"{extracted}\n\n"
            "Now: summarize it and answer any questions you can."
        )

        try:
            save_history(user_id, "user", user_text)
            reply = chat_reply(user_id, user_text)
            save_history(user_id, "assistant", reply)
        except Exception:
            reply = "I extracted the file, but the AI reply failed. Try again?"
        send_message(chat_id, reply, reply_to_message_id=message_id)
        return "ok"

    # -------- PHOTO (OCR) --------
    if "photo" in msg and msg["photo"]:
        file_id = msg["photo"][-1].get("file_id")
        if not file_id:
            send_message(chat_id, "I got the image, but couldn’t access it.", reply_to_message_id=message_id)
            return "ok"

        img_bytes = get_file_bytes(file_id)
        ocr_text = ocr_image(img_bytes)

        user_text = f"Text extracted from image (OCR):\n\n{ocr_text}\n\nExplain what it says and answer the user."
        try:
            save_history(user_id, "user", user_text)
            reply = chat_reply(user_id, user_text)
            save_history(user_id, "assistant", reply)
        except Exception:
            reply = "I read the image text, but AI reply failed. Try again?"
        send_message(chat_id, reply, reply_to_message_id=message_id)
        return "ok"

    send_message(chat_id, "Send text, a voice note, a PDF, an Excel/CSV, or an image 🙂", reply_to_message_id=message_id)
    return "ok"

# ------------------- Main -------------------
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
