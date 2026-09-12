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


# Модель Whisper
WHISPER_MODEL = "whisper-large-v3"


# Максимальна кількість одночасних транскрипцій
TRANSCRIPTION_LIMIT = 3

transcription_semaphore = asyncio.Semaphore(
    TRANSCRIPTION_LIMIT
)


# Максимальний розмір файлу
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


# Telegram обмежує довжину повідомлення.
# Беремо запас, щоб HTML точно влазив.
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

        # Якщо окремий абзац довший за ліміт
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

        # Додаємо до поточної частини
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
# WHISPER
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

        with open(
            file_path,
            "rb",
        ) as audio_file:

            result = await client.audio.transcriptions.create(

                # -------------------------------------------------
                # МОДЕЛЬ
                # -------------------------------------------------

                model=WHISPER_MODEL,


                # -------------------------------------------------
                # ФАЙЛ
                # -------------------------------------------------

                file=(
                    filename,
                    audio_file,
                ),


                # -------------------------------------------------
                # ФОРМАТ
                # -------------------------------------------------

                response_format="json",


                # -------------------------------------------------
                # МІНІМАЛЬНА ВАРІАТИВНІСТЬ
                # -------------------------------------------------

                temperature=0.0,


                # -------------------------------------------------
                # ВАЖЛИВО:
                # НЕ ВКАЗУЄМО language
                #
                # Бо людина може говорити українською,
                # російською та суржиком в одному голосовому.
                # -------------------------------------------------


                # -------------------------------------------------
                # КОРОТКИЙ КОНТЕКСТ
                # -------------------------------------------------

                initial_prompt=(
                    "Дослівна транскрипція розмовної мови. "
                    "Українська, російська та суржик. "
                    "Не перекладай. "
                    "Не перефразовуй. "
                    "Не виправляй слова або граматику. "
                    "Зберігай сленг, матюки, повтори та помилки."
                ),
            )

        text = (result.text or "").strip()

        logger.info(
            "✅ Транскрипцію завершено. Символів: %d",
            len(text),
        )

        return text


# =========================================================
# ВІДПРАВКА ТРАНСКРИПЦІЇ
# =========================================================

async def send_transcription(
    message,
    text: str,
):

    parts = split_text(text)

    for index, part in enumerate(parts):

        # Безпечний HTML
        safe_text = html.escape(part)

        result_text = (
            "<blockquote expandable>"
            f"{safe_text}"
            "</blockquote>"
        )

        # Перша частина — відповідь на голосове
        if index == 0:

            await message.reply_text(
                result_text,
                parse_mode="HTML",
                reply_to_message_id=message.message_id,
            )

        # Наступні частини — звичайними повідомленнями
        else:

            await message.reply_text(
                result_text,
                parse_mode="HTML",
            )


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


    # =====================================================
    # ВИЗНАЧАЄМО ТИП АУДІО
    # =====================================================

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


    # =====================================================
    # ПЕРЕВІРКА РОЗМІРУ
    # =====================================================

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

        # =================================================
        # ПОКАЗУЄМО, ЩО БОТ ПРАЦЮЄ
        # =================================================

        await context.bot.send_chat_action(
            chat_id=message.chat_id,
            action=ChatAction.TYPING,
        )


        status_message = await message.reply_text(
            "🎙️ <i>Розпізнаю...</i>",
            parse_mode="HTML",
            reply_to_message_id=message.message_id,
        )


        # =================================================
        # ОТРИМУЄМО ФАЙЛ TELEGRAM
        # =================================================

        telegram_file = await context.bot.get_file(
            telegram_media.file_id
        )


        # =================================================
        # СТВОРЮЄМО ТИМЧАСОВИЙ ФАЙЛ
        # =================================================

        suffix = os.path.splitext(filename)[1]

        if not suffix:
            suffix = ".ogg"


        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name


        # =================================================
        # ЗАВАНТАЖУЄМО АУДІО
        # =================================================

        await telegram_file.download_to_drive(
            custom_path=temp_path
        )


        # =================================================
        # ПЕРЕВІРКА ФАЙЛУ
        # =================================================

        if not os.path.exists(temp_path):

            raise RuntimeError(
                "Файл не був завантажений."
            )


        actual_size = os.path.getsize(
            temp_path
        )


        if actual_size <= 0:

            raise RuntimeError(
                "Завантажений файл порожній."
            )


        if actual_size > MAX_FILE_SIZE:

            await status_message.edit_text(
                "❌ Аудіо занадто велике.\n"
                "Максимальний розмір — 25 МБ."
            )

            return


        logger.info(
            "📥 Файл завантажено: %s | %.2f MB",
            filename,
            actual_size / 1024 / 1024,
        )


        # =================================================
        # ТРАНСКРИПЦІЯ
        # =================================================

        text = await transcribe_audio(
            temp_path,
            filename,
        )


        # =================================================
        # НІЧОГО НЕ РОЗІБРАНО
        # =================================================

        if not text:

            await status_message.edit_text(
                "🤷 Не зміг розібрати текст."
            )

            return


        # =================================================
        # ВИДАЛЯЄМО "РОЗПІЗНАЮ..."
        # =================================================

        try:

            await status_message.delete()

        except TelegramError:

            pass


        # =================================================
        # ВІДПРАВЛЯЄМО РЕЗУЛЬТАТ
        # =================================================

        await send_transcription(
            message,
            text,
        )


        logger.info(
            "✅ Готово: %s",
            filename,
        )


    # =====================================================
    # ПОМИЛКА
    # =====================================================

    except Exception as e:

        logger.exception(
            "❌ Помилка обробки аудіо: %s",
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
    # ОЧИЩЕННЯ
    # =====================================================

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            try:

                os.remove(temp_path)

                logger.info(
                    "🗑️ Тимчасовий файл видалено."
                )

            except Exception:

                logger.warning(
                    "⚠️ Не вдалося видалити "
                    "тимчасовий файл: %s",
                    temp_path,
                )


# =========================================================
# GLOBAL ERROR HANDLER
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


    # =====================================================
    # /START
    # =====================================================

    app.add_handler(
        CommandHandler(
            "start",
            cmd_start,
        )
    )


    # =====================================================
    # VOICE + AUDIO
    # =====================================================

    app.add_handler(
        MessageHandler(
            filters.VOICE | filters.AUDIO,
            handle_audio,
        )
    )


    # =====================================================
    # ПОМИЛКИ
    # =====================================================

    app.add_error_handler(
        error_handler
    )


    logger.info(
        "✅ WhisperBot запущений."
    )


    # =====================================================
    # POLLING
    # =====================================================

    app.run_polling(
        drop_pending_updates=True
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()
