import os
import asyncio
import logging
import tempfile
import html

from telegram import Update
from telegram.constants import ChatAction
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

# Одночасно максимум 3 транскрипції
TRANSCRIPTION_LIMIT = 3
transcription_semaphore = asyncio.Semaphore(TRANSCRIPTION_LIMIT)

# Максимальний розмір файлу
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


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
# ТРАНСКРИПЦІЯ
# =========================================================

async def transcribe_audio(
    file_path: str,
    filename: str,
) -> str:

    async with transcription_semaphore:

        logger.info(
            "Починаю транскрипцію: %s",
            filename,
        )

        with open(file_path, "rb") as audio_file:

            result = await client.audio.transcriptions.create(
                model=WHISPER_MODEL,

                file=(
                    filename,
                    audio_file,
                ),

                response_format="json",

                temperature=0.0,

                initial_prompt=(
                    "Українська, російська мова та суржик. "
                    "Транскрибуй максимально дослівно. "
                    "Зберігай слова, сленг, мат, помилки, "
                    "повтори та особливості мовлення. "
                    "Не перекладай. "
                    "Не перефразовуй. "
                    "Не роби текст красивішим. "
                    "Не виправляй граматику. "
                    "Передавай саме те, що було сказано."
                ),
            )

        text = result.text.strip()

        logger.info(
            "Транскрипцію завершено. Символів: %d",
            len(text),
        )

        return text


# =========================================================
# ОБРОБКА АУДІО
# =========================================================

async def handle_audio(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.message

    if not message:
        return

    # -----------------------------------------------------
    # Визначаємо тип повідомлення
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
    # Перевірка розміру
    # -----------------------------------------------------

    if (
        telegram_media.file_size
        and telegram_media.file_size > MAX_FILE_SIZE
    ):
        await message.reply_text(
            "❌ Файл занадто великий.\n"
            "Максимальний розмір — 25 МБ.",
            reply_to_message_id=message.message_id,
        )
        return

    temp_path = None
    status_message = None

    try:

        # -------------------------------------------------
        # Статус
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
        # Отримуємо файл Telegram
        # -------------------------------------------------

        telegram_file = await context.bot.get_file(
            telegram_media.file_id
        )

        # -------------------------------------------------
        # Тимчасовий файл
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
        # Завантаження
        # -------------------------------------------------

        await telegram_file.download_to_drive(
            custom_path=temp_path
        )

        # -------------------------------------------------
        # Повторна перевірка реального розміру
        # -------------------------------------------------

        actual_size = os.path.getsize(temp_path)

        if actual_size > MAX_FILE_SIZE:

            await status_message.edit_text(
                "❌ Файл занадто великий.\n"
                "Максимальний розмір — 25 МБ."
            )

            return

        # -------------------------------------------------
        # Whisper
        # -------------------------------------------------

        text = await transcribe_audio(
            temp_path,
            filename,
        )

        # -------------------------------------------------
        # Порожній результат
        # -------------------------------------------------

        if not text:

            await status_message.edit_text(
                "🤷 Не зміг розібрати текст."
            )

            return

        # -------------------------------------------------
        # Безпечний HTML
        # -------------------------------------------------

        safe_text = html.escape(text)

        result_text = (
            f"<blockquote expandable>"
            f"{safe_text}"
            f"</blockquote>"
        )

        # -------------------------------------------------
        # Результат
        # -------------------------------------------------

        await status_message.edit_text(
            result_text,
            parse_mode="HTML",
        )

        logger.info(
            "Готово: %s",
            filename,
        )

    # =====================================================
    # ПОМИЛКА
    # =====================================================

    except Exception as e:

        logger.exception(
            "Помилка обробки аудіо: %s",
            e,
        )

        try:

            if status_message:

                await status_message.edit_text(
                    "❌ Помилка під час розпізнавання.\n"
                    "Спробуй ще раз."
                )

            else:

                await message.reply_text(
                    "❌ Помилка під час розпізнавання.",
                    reply_to_message_id=message.message_id,
                )

        except Exception:
            pass

    # =====================================================
    # ВИДАЛЕННЯ ТИМЧАСОВОГО ФАЙЛУ
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
                    "Не вдалося видалити: %s",
                    temp_path,
                )


# =========================================================
# GLOBAL ERROR HANDLER
# =========================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):

    logger.exception(
        "Telegram Error: %s",
        context.error,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    logger.info("🚀 Запуск WhisperBot...")

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

    # Голосові + аудіофайли
    app.add_handler(
        MessageHandler(
            filters.VOICE | filters.AUDIO,
            handle_audio,
        )
    )

    # Глобальні помилки
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
