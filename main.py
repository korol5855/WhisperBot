import os
import asyncio
import logging
import tempfile
import html
import re

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


WHISPER_MODEL = "whisper-large-v3"

# Максимум одночасних запитів до Groq
TRANSCRIPTION_LIMIT = 3

transcription_semaphore = asyncio.Semaphore(
    TRANSCRIPTION_LIMIT
)

# Максимальний розмір файлу
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

# Максимальна довжина одного Telegram-повідомлення
MAX_MESSAGE_LENGTH = 4000


# =========================================================
# ЛОГУВАННЯ
# =========================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("WhisperBot")


# =========================================================
# GROQ
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
        "🎙️ <b>WhisperBot готовий.</b>\n\n"
        "Надішли голосове повідомлення або аудіофайл — "
        "я перетворю його на текст.",
        parse_mode="HTML",
    )


# =========================================================
# ПЕРЕВІРКА НА ПІДОЗРІЛУ МОВУ
# =========================================================

# Польські символи/буквосполучення.
# Це НЕ визначення мови на 100%.
# Це лише сигнал, що Whisper міг піти не в ту мову.

POLISH_PATTERNS = [
    r"\bco\b",
    r"\bże\b",
    r"\bjest\b",
    r"\bnie\b",
    r"\bsię\b",
    r"\bcię\b",
    r"\bczy\b",
    r"\bjak\b",
    r"\bto\b",
    r"\bna\b",
    r"\bdo\b",
    r"\bten\b",
    r"\bta\b",
    r"\btego\b",
    r"\bzaraz\b",
    r"\bmyślał\b",
    r"\bmyśleć\b",
    r"\bcz\b",
    r"\bsz\b",
    r"\bczł\b",
    r"\bą\b",
    r"\bę\b",
    r"\bł\b",
    r"\bś\b",
    r"\bź\b",
    r"\bż\b",
]


def looks_like_wrong_language(text: str) -> bool:

    if not text:
        return False

    text_lower = text.lower()

    matches = 0

    for pattern in POLISH_PATTERNS:

        if re.search(pattern, text_lower):
            matches += 1

    # Якщо знайдено кілька характерних польських ознак
    if matches >= 2:
        return True

    return False


# =========================================================
# ТРАНСКРИПЦІЯ
# =========================================================

async def whisper_request(
    file_path: str,
    filename: str,
    language: str | None = None,
) -> str:

    with open(
        file_path,
        "rb",
    ) as audio_file:

        kwargs = {
            "model": WHISPER_MODEL,

            "file": (
                filename,
                audio_file,
            ),

            "response_format": "json",

            "temperature": 0.0,

            "prompt": (
                "Дослівна транскрипція розмовної мови. "
                "Українська, російська та суржик. "
                "Не перекладай текст. "
                "Не перефразовуй. "
                "Не виправляй граматику. "
                "Зберігай сленг, матюки, "
                "назви, повтори та помилки."
            ),
        }

        # Якщо ми вже знаємо мову — передаємо її Whisper.
        if language:
            kwargs["language"] = language

        result = await client.audio.transcriptions.create(
            **kwargs
        )

    return (result.text or "").strip()


# =========================================================
# ОСНОВНА ТРАНСКРИПЦІЯ
# =========================================================

async def transcribe_audio(
    file_path: str,
    filename: str,
) -> str:

    async with transcription_semaphore:

        logger.info(
            "🎙️ Починаю транскрипцію: %s",
            filename,
        )

        # -------------------------------------------------
        # ПЕРША СПРОБА
        # Без примусової мови.
        # -------------------------------------------------

        text = await whisper_request(
            file_path=file_path,
            filename=filename,
            language=None,
        )

        logger.info(
            "📝 Перша спроба: %s",
            text[:300],
        )

        # -------------------------------------------------
        # ЯКЩО РЕЗУЛЬТАТ СХОЖИЙ НА ПОЛЬСЬКУ
        # -------------------------------------------------

        if looks_like_wrong_language(text):

            logger.warning(
                "⚠️ Схоже, Whisper визначив неправильну мову. "
                "Запускаю повторну транскрипцію."
            )

            # ---------------------------------------------
            # ДРУГА СПРОБА — УКРАЇНСЬКА
            # ---------------------------------------------

            ukrainian_text = await whisper_request(
                file_path=file_path,
                filename=filename,
                language="uk",
            )

            logger.info(
                "🇺🇦 Українська спроба: %s",
                ukrainian_text[:300],
            )

            # ---------------------------------------------
            # ДРУГА СПРОБА — РОСІЙСЬКА
            # ---------------------------------------------

            russian_text = await whisper_request(
                file_path=file_path,
                filename=filename,
                language="ru",
            )

            logger.info(
                "🇷🇺 Російська спроба: %s",
                russian_text[:300],
            )

            # ---------------------------------------------
            # ВИБІР РЕЗУЛЬТАТУ
            # ---------------------------------------------
            #
            # Якщо один варіант очевидно не польський,
            # віддаємо перевагу йому.
            #
            # Якщо обидва нормальні — залишаємо перший
            # автоматичний результат, бо він міг краще
            # відповідати реальній мові.
            # ---------------------------------------------

            uk_wrong = looks_like_wrong_language(
                ukrainian_text
            )

            ru_wrong = looks_like_wrong_language(
                russian_text
            )

            if not uk_wrong and ru_wrong:

                text = ukrainian_text

            elif not ru_wrong and uk_wrong:

                text = russian_text

            elif ukrainian_text and russian_text:

                # Для змішаного суржику автоматичний результат
                # часто кращий за примусову мову.
                #
                # Тому тут залишаємо першу спробу, якщо вона
                # взагалі є.
                if text:
                    text = text

                else:
                    text = ukrainian_text

        # -------------------------------------------------
        # РЕЗУЛЬТАТ
        # -------------------------------------------------

        logger.info(
            "✅ Фінальна транскрипція: %d символів",
            len(text),
        )

        return text


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

    paragraphs = text.split("\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        while len(paragraph) > max_length:

            cut = paragraph.rfind(
                " ",
                0,
                max_length,
            )

            if cut <= 0:
                cut = max_length

            chunk = paragraph[:cut].strip()

            if chunk:
                parts.append(chunk)

            paragraph = paragraph[cut:].strip()

        if not paragraph:
            continue

        if not current:

            current = paragraph

        elif (
            len(current)
            + 1
            + len(paragraph)
            <= max_length
        ):

            current += "\n" + paragraph

        else:

            parts.append(current)
            current = paragraph

    if current:
        parts.append(current)

    return parts


# =========================================================
# ВІДПРАВКА РЕЗУЛЬТАТУ
# =========================================================

async def send_transcription(
    message,
    text: str,
):

    parts = split_text(text)

    for index, part in enumerate(parts):

        safe_text = html.escape(part)

        result_text = (
            "<blockquote expandable>"
            f"{safe_text}"
            "</blockquote>"
        )

        if index == 0:

            await message.reply_text(
                result_text,
                parse_mode="HTML",
                reply_to_message_id=message.message_id,
            )

        else:

            await message.reply_text(
                result_text,
                parse_mode="HTML",
            )


# =========================================================
# ОБРОБКА AUDIO / VOICE
# =========================================================

async def handle_audio(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    message = update.message

    if not message:
        return


    # -----------------------------------------------------
    # ВИЗНАЧАЄМО ФАЙЛ
    # -----------------------------------------------------

    if message.voice:

        telegram_media = message.voice
        filename = "voice.ogg"

    elif message.audio:

        telegram_media = message.audio

        filename = (
            message.audio.file_name
            or "audio.mp3"
        )

    else:

        return


    # -----------------------------------------------------
    # ПЕРЕВІРКА РОЗМІРУ
    # -----------------------------------------------------

    if (
        telegram_media.file_size
        and telegram_media.file_size > MAX_FILE_SIZE
    ):

        await message.reply_text(
            "❌ Аудіо занадто велике.\n"
            "Максимальний розмір — 25 МБ.",
            reply_to_message_id=message.message_id,
        )

        return


    temp_path = None
    status_message = None


    try:

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        await context.bot.send_chat_action(
            chat_id=message.chat_id,
            action=ChatAction.TYPING,
        )

        status_message = await message.reply_text(
            "🎙️ <i>Розпізнаю...</i>",
            parse_mode="HTML",
            reply_to_message_id=message.message_id,
        )


        # -------------------------------------------------
        # TELEGRAM FILE
        # -------------------------------------------------

        telegram_file = await context.bot.get_file(
            telegram_media.file_id
        )


        # -------------------------------------------------
        # TEMP FILE
        # -------------------------------------------------

        suffix = os.path.splitext(filename)[1]

        if not suffix:
            suffix = ".ogg"


        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name


        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        await telegram_file.download_to_drive(
            custom_path=temp_path
        )


        # -------------------------------------------------
        # CHECK FILE
        # -------------------------------------------------

        if not os.path.exists(temp_path):

            raise RuntimeError(
                "Файл не був завантажений."
            )


        actual_size = os.path.getsize(
            temp_path
        )


        if actual_size <= 0:

            raise RuntimeError(
                "Файл порожній."
            )


        if actual_size > MAX_FILE_SIZE:

            await status_message.edit_text(
                "❌ Аудіо занадто велике.\n"
                "Максимальний розмір — 25 МБ."
            )

            return


        logger.info(
            "📥 Файл: %s | %.2f MB",
            filename,
            actual_size / 1024 / 1024,
        )


        # -------------------------------------------------
        # WHISPER
        # -------------------------------------------------

        text = await transcribe_audio(
            file_path=temp_path,
            filename=filename,
        )


        # -------------------------------------------------
        # EMPTY
        # -------------------------------------------------

        if not text:

            await status_message.edit_text(
                "🤷 Не зміг розібрати текст."
            )

            return


        # -------------------------------------------------
        # DELETE STATUS
        # -------------------------------------------------

        try:

            await status_message.delete()

        except TelegramError:

            pass


        # -------------------------------------------------
        # SEND RESULT
        # -------------------------------------------------

        await send_transcription(
            message,
            text,
        )


        logger.info(
            "✅ Готово: %s",
            filename,
        )


    # =====================================================
    # ERROR
    # =====================================================

    except Exception as e:

        logger.exception(
            "❌ Помилка обробки: %s",
            e,
        )

        try:

            if status_message:

                await status_message.edit_text(
                    "❌ <b>Помилка розпізнавання.</b>\n"
                    "Спробуй ще раз.",
                    parse_mode="HTML",
                )

            else:

                await message.reply_text(
                    "❌ Помилка розпізнавання.",
                    reply_to_message_id=message.message_id,
                )

        except Exception:

            pass


    # =====================================================
    # CLEANUP
    # =====================================================

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            try:

                os.remove(temp_path)

            except Exception:

                logger.warning(
                    "⚠️ Не вдалося видалити файл: %s",
                    temp_path,
                )


# =========================================================
# GLOBAL ERROR
# =========================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):

    logger.error(
        "Telegram Error: %s",
        context.error,
        exc_info=context.error,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    logger.info(
        "🚀 Запуск WhisperBot..."
    )


    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .concurrent_updates(True)
        .build()
    )


    # /start
    app.add_handler(
        CommandHandler(
            "start",
            cmd_start,
        )
    )


    # Voice + Audio
    app.add_handler(
        MessageHandler(
            filters.VOICE | filters.AUDIO,
            handle_audio,
        )
    )


    # Errors
    app.add_error_handler(
        error_handler
    )


    logger.info(
        "✅ WhisperBot запущений."
    )


    app.run_polling(
        drop_pending_updates=True
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()
