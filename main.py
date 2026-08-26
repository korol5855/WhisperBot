import os
import asyncio
import logging
import tempfile
import html

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)
from groq import AsyncGroq

# Налаштування логування
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("WhisperBot")

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not BOT_TOKEN or not GROQ_API_KEY:
    raise RuntimeError("❌ Не задано BOT_TOKEN або GROQ_API_KEY!")

client = AsyncGroq(api_key=GROQ_API_KEY, timeout=120.0)
WHISPER_MODEL = "whisper-large-v3"

transcription_semaphore = asyncio.Semaphore(3)
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text(
            "🎙️ WhisperBot готовий до роботи. Надішли голосове повідомлення.",
        )

async def transcribe_voice(file_path: str) -> str:
    async with transcription_semaphore:
        with open(file_path, "rb") as audio_file:
            result = await client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=("voice.ogg", audio_file),
                response_format="json",
                temperature=0.0,
            )
            return result.text.strip()

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.voice:
        return

    if message.voice.file_size and message.voice.file_size > MAX_FILE_SIZE:
        await message.reply_text(
            "❌ Голосове занадто велике (максимум 25 МБ).",
            reply_to_message_id=message.message_id,
        )
        return

    voice_path = None
    try:
        telegram_file = await context.bot.get_file(message.voice.file_id)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
            voice_path = temp_file.name

        await telegram_file.download_to_drive(custom_path=voice_path)

        if os.path.getsize(voice_path) > MAX_FILE_SIZE:
            await message.reply_text(
                "❌ Голосове занадто велике.",
                reply_to_message_id=message.message_id,
            )
            return

        text = await transcribe_voice(voice_path)

        if not text:
            await message.reply_text(
                "🤷 Не зміг розібрати текст.",
                reply_to_message_id=message.message_id,
            )
            return

        safe_text = html.escape(text)
        result_text = f"<blockquote expandable>{safe_text}</blockquote>"

        await message.reply_text(
            result_text,
            parse_mode="HTML",
            reply_to_message_id=message.message_id,
        )

    except Exception as e:
        logger.exception(f"Помилка обробки: {e}")
        try:
            await message.reply_text(
                "❌ Помилка розпізнавання.",
                reply_to_message_id=message.message_id,
            )
        except:
            pass
    finally:
        if voice_path and os.path.exists(voice_path):
            try:
                os.remove(voice_path)
            except:
                pass

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception(f"Telegram Error: {context.error}")

def main():
    logger.info("🚀 Запуск WhisperBot...")
    app = ApplicationBuilder().token(BOT_TOKEN).concurrent_updates(True).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_error_handler(error_handler)

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
    
