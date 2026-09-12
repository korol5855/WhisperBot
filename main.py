import os
import asyncio
import logging
import tempfile
import html
import statistics

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


# Найточніша модель Whisper у Грут для нашої задачі
WHISPER_MODEL = "whisper-large-v3"

# Максимум одночасних транскрипцій
TRANSCRIPTION_LIMIT = 3

transcription_semaphore = asyncio.Semaphore(
    TRANSCRIPTION_LIMIT
)

# Максимальний розмір аудіо
MAX_FILE_SIZE = 25 * 1024 * 1024

# Telegram
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
# ДОПОМІЖНІ ФУНКЦІЇ
# =========================================================

def get_value(obj, name, default=None):
    """
    Працює і з Pydantic-об'єктами Groq,
    і зі звичайними dict.
    """

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def calculate_confidence(result) -> float:
    """
    Рахуємо середній avg_logprob по сегментах.

    Чим ближче значення до 0 — тим впевненіше Whisper.
    Наприклад:
        -0.15  -> дуже добре
        -0.40  -> нормально
        -0.80  -> підозріло
        -1.20  -> дуже підозріло
    """

    segments = get_value(result, "segments", []) or []

    values = []

    for segment in segments:
        avg_logprob = get_value(
            segment,
            "avg_logprob",
            None,
        )

        if avg_logprob is None:
            continue

        try:
            values.append(float(avg_logprob))
        except (TypeError, ValueError):
            pass

    if not values:
        return -999.0

    return statistics.mean(values)


def calculate_no_speech(result) -> float:
    """
    Середня ймовірність тиші.
    """

    segments = get_value(result, "segments", []) or []

    values = []

    for segment in segments:
        no_speech_prob = get_value(
            segment,
            "no_speech_prob",
            None,
        )

        if no_speech_prob is None:
            continue

        try:
            values.append(float(no_speech_prob))
        except (TypeError, ValueError):
            pass

    if not values:
        return 0.0

    return statistics.mean(values)


def calculate_compression_ratio(result) -> float:
    """
    Високий compression_ratio може бути ознакою
    повторів або галюцинацій.
    """

    segments = get_value(result, "segments", []) or []

    values = []

    for segment in segments:
        ratio = get_value(
            segment,
            "compression_ratio",
            None,
        )

        if ratio is None:
            continue

        try:
            values.append(float(ratio))
        except (TypeError, ValueError):
            pass

    if not values:
        return 0.0

    return statistics.mean(values)


# =========================================================
# ОЦІНКА РЕЗУЛЬТАТУ
# =========================================================

def score_result(
    text: str,
    confidence: float,
    no_speech: float,
    compression_ratio: float,
) -> float:

    if not text:
        return -1000.0

    score = 0.0

    # ---------------------------------------------
    # Впевненість Whisper
    # ---------------------------------------------

    if confidence > -0.25:
        score += 100

    elif confidence > -0.40:
        score += 80

    elif confidence > -0.60:
        score += 50

    elif confidence > -0.80:
        score += 20

    elif confidence > -1.00:
        score -= 20

    else:
        score -= 60


    # ---------------------------------------------
    # Надмірна компресія
    # ---------------------------------------------

    if compression_ratio > 3.5:
        score -= 40

    elif compression_ratio > 2.8:
        score -= 15


    # ---------------------------------------------
    # Ймовірність тиші
    # ---------------------------------------------

    if no_speech > 0.85 and len(text) > 20:
        score -= 50


    # ---------------------------------------------
    # Дуже короткий результат
    # ---------------------------------------------

    if len(text) < 3:
        score -= 50


    # ---------------------------------------------
    # Повтори одного й того самого
    # ---------------------------------------------

    words = text.lower().split()

    if len(words) >= 8:

        unique_words = len(set(words))
        total_words = len(words)

        repetition_ratio = (
            unique_words / total_words
        )

        if repetition_ratio < 0.35:
            score -= 40

        elif repetition_ratio < 0.50:
            score -= 15


    return score


# =========================================================
# WHISPER ЗАПИТ
# =========================================================

async def whisper_request(
    file_path: str,
    filename: str,
    language: str | None = None,
):
    """
    Виконує один запит до Whisper.

    language:
        None -> автоматичне визначення
        uk   -> українська
        ru   -> російська
    """

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

            # verbose_json потрібен для confidence
            "response_format": "verbose_json",

            # Для транскрипції краще 0
            "temperature": 0.0,

            # Короткий prompt.
            # Не намагаємося командувати Whisper
            # на пів сторінки.
            "prompt": (
                "Розмовна українська та російська мова, "
                "суржик. Дослівна транскрипція."
            ),
        }

        if language:
            kwargs["language"] = language

        result = await client.audio.transcriptions.create(
            **kwargs
        )

    text = (
        get_value(result, "text", "")
        or ""
    ).strip()

    confidence = calculate_confidence(
        result
    )

    no_speech = calculate_no_speech(
        result
    )

    compression_ratio = calculate_compression_ratio(
        result
    )

    score = score_result(
        text=text,
        confidence=confidence,
        no_speech=no_speech,
        compression_ratio=compression_ratio,
    )

    return {
        "text": text,
        "confidence": confidence,
        "no_speech": no_speech,
        "compression_ratio": compression_ratio,
        "score": score,
        "language": language or "auto",
    }


# =========================================================
# ПЕРЕВІРКА НА ПІДОЗРІЛИЙ РЕЗУЛЬТАТ
# =========================================================

def looks_suspicious(result) -> bool:

    text = result["text"]

    if not text:
        return True

    confidence = result["confidence"]
    score = result["score"]

    # Дуже низька впевненість
    if confidence < -0.75:
        return True

    # Поганий загальний score
    if score < 20:
        return True

    # Дуже дивний compression ratio
    if result["compression_ratio"] > 3.5:
        return True

    return False


# =========================================================
# ВИБІР НАЙКРАЩОГО РЕЗУЛЬТАТУ
# =========================================================

def choose_best_result(results):

    valid_results = [
        result
        for result in results
        if result["text"]
    ]

    if not valid_results:
        return {
            "text": "",
            "score": -1000,
            "language": "none",
        }

    # Вибираємо не найдовший текст,
    # а результат із найкращою оцінкою.
    best = max(
        valid_results,
        key=lambda x: x["score"],
    )

    return best


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

        # =================================================
        # 1. ОСНОВНА СПРОБА — АВТОВИЗНАЧЕННЯ
        # =================================================

        auto_result = await whisper_request(
            file_path=file_path,
            filename=filename,
            language=None,
        )

        logger.info(
            "📝 AUTO | score=%.2f | confidence=%.3f | %s",
            auto_result["score"],
            auto_result["confidence"],
            auto_result["text"][:300],
        )


        # =================================================
        # Якщо результат нормальний —
        # НЕ робимо зайвих запитів.
        #
        # Це важливо для змішаного українсько-
        # російського мовлення.
        # =================================================

        if not looks_suspicious(auto_result):

            logger.info(
                "✅ AUTO результат визнано нормальним."
            )

            return auto_result["text"]


        # =================================================
        # 2. РЕЗУЛЬТАТ ПІДОЗРІЛИЙ
        #
        # Робимо контрольні uk + ru.
        # =================================================

        logger.warning(
            "⚠️ AUTO результат підозрілий. "
            "Запускаю контрольні uk/ru."
        )


        ukrainian_task = whisper_request(
            file_path=file_path,
            filename=filename,
            language="uk",
        )

        russian_task = whisper_request(
            file_path=file_path,
            filename=filename,
            language="ru",
        )

        ukrainian_result, russian_result = (
            await asyncio.gather(
                ukrainian_task,
                russian_task,
            )
        )


        logger.info(
            "🇺🇦 UK | score=%.2f | confidence=%.3f | %s",
            ukrainian_result["score"],
            ukrainian_result["confidence"],
            ukrainian_result["text"][:300],
        )

        logger.info(
            "🇷🇺 RU | score=%.2f | confidence=%.3f | %s",
            russian_result["score"],
            russian_result["confidence"],
            russian_result["text"][:300],
        )


        # =================================================
        # 3. ВИБИРАЄМО НАЙКРАЩИЙ
        # =================================================

        best = choose_best_result([
            auto_result,
            ukrainian_result,
            russian_result,
        ])


        logger.info(
            "🏆 Обрано: %s | score=%.2f",
            best["language"],
            best["score"],
        )


        return best["text"]


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
# ВІДПРАВКА ТЕКСТУ
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
# AUDIO / VOICE
# =========================================================

async def handle_audio(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    message = update.message

    if not message:
        return


    # =====================================================
    # ВИЗНАЧАЄМО ТИП
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
        and telegram_media.file_size
        > MAX_FILE_SIZE
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
        # TELEGRAM STATUS
        # =================================================

        await context.bot.send_chat_action(
            chat_id=message.chat_id,
            action=ChatAction.TYPING,
        )


        status_message = (
            await message.reply_text(
                "🎙️ <i>Розпізнаю...</i>",
                parse_mode="HTML",
                reply_to_message_id=message.message_id,
            )
        )


        # =================================================
        # ЗАВАНТАЖЕННЯ ФАЙЛУ
        # =================================================

        telegram_file = (
            await context.bot.get_file(
                telegram_media.file_id
            )
        )


        suffix = os.path.splitext(
            filename
        )[1]

        if not suffix:
            suffix = ".ogg"


        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name


        await telegram_file.download_to_drive(
            custom_path=temp_path
        )


        # =================================================
        # ПЕРЕВІРКИ
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


        # =================================================
        # ТРАНСКРИПЦІЯ
        # =================================================

        text = await transcribe_audio(
            file_path=temp_path,
            filename=filename,
        )


        # =================================================
        # ПОРОЖНІЙ РЕЗУЛЬТАТ
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
    # ОЧИЩЕННЯ TEMP
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

    app.add_handler(
        CommandHandler(
            "start",
            cmd_start,
        )
    )

    app.add_handler(
        MessageHandler(
            filters.VOICE | filters.AUDIO,
            handle_audio,
        )
    )

    app.add_error_handler(
        error_handler
    )

    logger.info(
        "✅ WhisperBot запущений."
    )

    app.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
