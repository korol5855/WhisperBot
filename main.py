import os
import asyncio
import logging
import tempfile
import html

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import TelegramError

from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)

from groq import AsyncGroq


# =========================================================
# НАЛАШТУВАННЯ
# =========================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("❌ Не задано BOT_TOKEN!")

if not GROQ_API_KEY:
    raise RuntimeError("❌ Не задано GROQ_API_KEY!")


# =========================================================
# МОДЕЛЬ ТА ЛІМІТИ
# =========================================================

WHISPER_MODEL = "whisper-large-v3"

TRANSCRIPTION_LIMIT = 3
transcription_semaphore = asyncio.Semaphore(TRANSCRIPTION_LIMIT)

MAX_FILE_SIZE = 25 * 1024 * 1024
MAX_MESSAGE_LENGTH = 3800


# =========================================================
# ЛОГУВАННЯ
# =========================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("WhisperBot")


# =========================================================
# GROQ CLIENT
# =========================================================

client = AsyncGroq(
    api_key=GROQ_API_KEY,
    timeout=120.0,
)


# =========================================================
# /START
# =========================================================

async def cmd_start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.message:
        return

    await update.message.reply_text(
        "🎙️ <b>WhisperBot готовий до роботи.</b>\n\n"
        "Надішли голосове або аудіо — я перетворю його "
        "на текст дослівно (українська, російська, суржик, "
        "сленг і матюки без перекладу).",
        parse_mode="HTML",
    )


# =========================================================
# ПЕРЕВІРКА НА ЛАТИНСЬКИЙ ТРАНСЛІТ
# =========================================================

def looks_like_latin_translit(text: str) -> bool:
    if not text:
        return False

    latin_letters = sum(
        1 for char in text if char.lower() in "abcdefghijklmnopqrstuvwxyz"
    )
    cyrillic_letters = sum(
        1 for char in text if char.lower() in "абвгдежзийклмнопрстуфхцчшщьюяіїєґ"
    )

    if latin_letters >= 3 and cyrillic_letters == 0:
        logger.warning("⚠️ Виявлено можливий латинський трансліт: %s", text)
        return True

    return False


# =========================================================
# ПЕРЕВІРКА НА ЯВНІ ГАЛЮЦИНАЦІЇ
# =========================================================

HALLUCINATION_PHRASES = [
    "thanks for watching",
    "thank you for watching",
    "like and subscribe",
    "thank you for listening",
    "дякую за перегляд",
    "підписуйтесь на канал",
    "підписуйся на канал",
    "субтитры сделал",
    "subtitles by",
    "продолжение следует",
    "до свидания",
    "спасибо за просмотр",
    "ставьте лайки",
    "благодарю за просмотр",
]


def looks_like_hallucination(text: str) -> bool:
    if not text:
        return False

    text_lower = text.lower().strip()
    for phrase in HALLUCINATION_PHRASES:
        if phrase in text_lower:
            logger.warning("⚠️ Виявлено можливу галюцинацію Whisper: %s", phrase)
            return True

    return False


def result_is_suspicious(text: str) -> bool:
    if not text:
        return True
    if looks_like_latin_translit(text):
        return True
    if looks_like_hallucination(text):
        return True
    return False


# =========================================================
# ОСНОВНА ТРАНСКРИПЦІЯ
# =========================================================

async def transcribe_audio(
    file_path: str,
    filename: str,
) -> str:
    async with transcription_semaphore:
        logger.info("🎙️ Починаю транскрипцію: %s", filename)

        # Спроба №1 — AUTO
        logger.info("🔎 Спроба №1: AUTO")
        text_auto = await _request_whisper(file_path, filename, language=None)
        logger.info("📝 AUTO результат: %s", text_auto)

        if text_auto and not result_is_suspicious(text_auto):
            return text_auto

        # Спроба №2 — UK
        logger.warning("⚠️ AUTO дав підозрілий результат. Пробую language='uk'.")
        text_uk = await _request_whisper(file_path, filename, language="uk")
        logger.info("📝 UK результат: %s", text_uk)

        if text_uk and not result_is_suspicious(text_uk):
            return text_uk

        # Спроба №3 — RU
        logger.warning("⚠️ UK дав підозрілий результат. Пробую language='ru'.")
        text_ru = await _request_whisper(file_path, filename, language="ru")
        logger.info("📝 RU результат: %s", text_ru)

        if text_ru and not result_is_suspicious(text_ru):
            return text_ru

        # Fallbacks
        if text_auto:
            return text_auto
        if text_uk:
            return text_uk
        if text_ru:
            return text_ru

        return ""


# =========================================================
# ЗАПИТ ДО WHISPER
# =========================================================

async def _request_whisper(
    file_path: str,
    filename: str,
    language: str | None = None,
) -> str:
    with open(file_path, "rb") as audio_file:
        kwargs = {
            "model": WHISPER_MODEL,
            "file": (filename, audio_file),
            "response_format": "json",
            "temperature": 0.0,
            "prompt": (
                "Це жива розмова українською, російською або суржиком. "
                "Транскрибуй дослівно те, що реально говорить людина. "
                "Не перекладай. Не перефразовуй. Не виправляй граматику. "
                "Не замінюй російські слова українськими. "
                "Не замінюй українські слова російськими. "
                "Не виправляй суржик на літературну мову. "
                "Зберігай сленг, матюки, скорочення, розмовні слова та імена. "
                "Українську, російську та суржик записуй кирилицею. "
                "Якщо слово нерозбірливе, не вигадуй його."
            ),
        }

        if language:
            kwargs["language"] = language

        result = await client.audio.transcriptions.create(**kwargs)

    text = getattr(result, "text", "") or ""
    return text.strip()


# =========================================================
# РОЗБИВАННЯ ДОВГОГО ТЕКСТУ
# =========================================================

def split_text(
    text: str,
    max_length: int = MAX_MESSAGE_LENGTH,
) -> list[str]:
    if len(text) <= max_length:
        return [text]

    parts = []
    current = ""

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        while len(paragraph) > max_length:
            cut = paragraph.rfind(" ", 0, max_length)
            if cut <= 0:
                cut = max_length
            parts.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()

        if not paragraph:
            continue

        if not current:
            current = paragraph
        elif len(current) + 1 + len(paragraph) <= max_length:
            current += "\n" + paragraph
        else:
            parts.append(current)
            current = paragraph

    if current:
        parts.append(current)

    return parts


# =========================================================
# ВІДПРАВКА ТРАНСКРИПЦІЇ
# =========================================================

async def send_transcription(
    message,
    text: str,
):
    parts = split_text(text)

    for index, part in enumerate(parts):
        safe_text = html.escape(part)
        result_text = f"<blockquote expandable>{safe_text}</blockquote>"

        if index == 0:
            await message.reply_text(
                result_text,
                parse_mode="HTML",
                reply_to_message_id=message.message_id,
            )
        else:
            await message.reply_text(result_text, parse_mode="HTML")


# =========================================================
# ОБРОБКА VOICE / AUDIO
# =========================================================

async def handle_audio(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.message
    if not message:
        return

    if message.voice:
        telegram_media = message.voice
        filename = "voice.ogg"
    elif message.audio:
        telegram_media = message.audio
        filename = message.audio.file_name or "audio.mp3"
    else:
        return

    if telegram_media.file_size and telegram_media.file_size > MAX_FILE_SIZE:
        await message.reply_text(
            "❌ Файл занадто великий (ліміт 25 МБ).",
            reply_to_message_id=message.message_id,
        )
        return

    temp_path = None

    try:
        # Показуємо статус друку/завантаження в Telegram (опціонально)
        await context.bot.send_chat_action(
            chat_id=message.chat_id,
            action=ChatAction.TYPING,
        )

        telegram_file = await context.bot.get_file(telegram_media.file_id)

        suffix = os.path.splitext(filename)[1] or ".ogg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name

        await telegram_file.download_to_drive(custom_path=temp_path)

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise RuntimeError("Не вдалося завантажити файл або він порожній.")

        text = await transcribe_audio(temp_path, filename)

        if not text:
            await message.reply_text(
                "🤷 Не вдалося розібрати слова.",
                reply_to_message_id=message.message_id,
            )
            return

        await send_transcription(message, text)
        logger.info("✅ Успішно оброблено: %s", filename)

    except Exception as e:
        logger.exception("❌ Помилка обробки: %s", e)
        try:
            await message.reply_text(
                "❌ Помилка розпізнавання. Спробуй ще раз.",
                reply_to_message_id=message.message_id,
            )
        except Exception:
            pass

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# =========================================================
# ГЛОБАЛЬНИЙ ОБРОБНИК ПОМИЛОК
# =========================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):
    logger.error("Telegram Error: %s", context.error, exc_info=context.error)


# =========================================================
# ЗАПУСК БОТА
# =========================================================

def main():
    logger.info("🚀 Запуск фінального WhisperBot...")

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
    app.add_error_handler(error_handler)

    logger.info("✅ Бот успішно запущений і готовий.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
    
