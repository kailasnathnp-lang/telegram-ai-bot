# Telegram AI Bot using Groq + Flask 🤖

A cloud-deployed Telegram AI assistant that can chat naturally, remember user context,
read documents, and transcribe voice messages.

## 🚀 Features
- Natural conversation via Telegram
- Persistent memory & chat history (SQLite)
- Daily notes per user
- PDF document reading & summarization
- Excel / CSV file processing
- Voice message transcription (Whisper)
- Secure webhook-based cloud deployment

## 🧠 Tech Stack
- Python
- Flask
- Telegram Bot API
- Groq LLM (OpenAI-compatible)
- SQLite
- Gunicorn
- Render Cloud

## 🏗️ Architecture
Telegram → Webhook → Flask API → Groq LLM → SQLite → Telegram

## 🖼️ Demo Screenshots
Screenshots are available in the `/assets` folder:
- Telegram chat interaction
- File upload & response
- Render deployment status

## ⚙️ Setup (optional)
```bash
pip install -r requirements.txt
python app.py
