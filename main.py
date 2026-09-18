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

# Максимум одночасних транскрипцій
TRANSCRIPTION_LIMIT = 3
transcription_semaphore = asyncio.Semaphore(TRANSCRIPTION_LIMIT)

# Максимальний розмір файлу
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

# Telegram має ліміт близько 4096 символів.
# Залишаємо запас для HTML.
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
        "Надішли голосове або аудіо — я перетворю його в текст.",
        parse_mode="HTML",
    )


# =========================================================
# ПЕРЕВІРКА НА ПОЛЬСЬКУ / НЕПРАВИЛЬНУ МОВУ
# =========================================================

POLISH_WORD_PATTERNS = [
    r"\bco\b",
    r"\bże\b",
    r"\bjest\b",
    r"\bsię\b",
    r"\bcię\b",
    r"\bczy\b",
    r"\bjak\b",
    r"\bto\b",
    r"\bna\b",
    r"\bdo\b",
    r"\bzaraz\b",
    r"\bmyślał\b",
    r"\bmyśleć\b",
    r"\bmyślę\b",
    r"\bczemu\b",
    r"\bdlaczego\b",
    r"\bteraz\b",
    r"\bjeszcze\b",
    r"\bmoże\b",
    r"\bbyć\b",
    r"\bniech\b",
]

# Польські літери, які дуже добре сигналізують,
# що Whisper пішов не в ту мову.
POLISH_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

# Типові польські закінчення/фрагменти.
POLISH_FRAGMENTS = [
    "cz",
    "sz",
    "rz",
    "dz",
    "ści",
    "śmy",
    "ście",
    "owej",
    "owego",
    "ami",
    "ach",
]


def looks_like_wrong_language(text: str) -> bool:
    """
    Перевіряє, чи результат Whisper підозріло схожий
    на польську або іншу неправильну мову.
    """

    if not text:
        return False

    text_lower = text.lower().strip()

    # -----------------------------------------------------
    # 1. Польські специфічні символи
    # -----------------------------------------------------

    polish_char_count = sum(
        1 for char in text if char in POLISH_CHARS
    )

    # Якщо є хоча б кілька польських специфічних символів
    # у короткому тексті — це вже сильний сигнал.
    if polish_char_count >= 2:
        logger.warning(
            "⚠️ Виявлено польські специфічні символи: %s",
            polish_char_count,
        )
        return True

    # -----------------------------------------------------
    # 2. Польські службові слова
    # -----------------------------------------------------

    word_matches = 0

    for pattern in POLISH_WORD_PATTERNS:
        if re.search(pattern, text_lower):
            word_matches += 1

    # 2+ характерних слова — вже підозріло.
    if word_matches >= 2:
        logger.warning(
            "⚠️ Виявлено польські слова: %s",
            word_matches,
        )
        return True

    # -----------------------------------------------------
    # 3. Польські характерні фрагменти
    # -----------------------------------------------------

    fragment_matches = 0

    for fragment in POLISH_FRAGMENTS:
        if fragment in text_lower:
            fragment_matches += 1

    # Фрагменти самі по собі слабкий сигнал,
    # тому вимагаємо декілька.
    if fragment_matches >= 4:
        logger.warning(
            "⚠️ Виявлено багато польських фрагментів: %s",
            fragment_matches,
        )
        return True

    # -----------------------------------------------------
    # 4. Дуже короткий дивний результат
    # -----------------------------------------------------

    words = text_lower.split()

    # Наприклад, голосове на 20 секунд,
    # а Whisper повернув 1-2 дивних слова.
    if len(words) <= 2:
        suspicious_words = [
            "co",
            "że",
            "jest",
            "się",
            "myślał",
            "myśleć",
        ]

        if any(word in suspicious_words for word in words):
            logger.warning(
                "⚠️ Дуже короткий підозрілий результат: %s",
                text,
            )
            return True

    return False


# =========================================================
# ПЕРЕВІРКА НА WHISPER-ГАЛЮЦИНАЦІЇ
# =========================================================

HALLUCINATION_PHRASES = [
    "thanks for watching",
    "thank you for watching",
    "subscribe",
    "like and subscribe",
    "thank you",
    "дякую за перегляд",
    "підписуйтесь на канал",
    "підписуйся на канал",
    "www.",
    "http://",
    "https://",
]


def looks_like_hallucination(text: str) -> bool:
    if not text:
        return False

    text_lower = text.lower().strip()

    for phrase in HALLUCINATION_PHRASES:
        if phrase in text_lower:
            logger.warning(
                "⚠️ Виявлено можливу галюцинацію Whisper: %s",
                phrase,
            )
            return True

    return False


# =========================================================
# ОЦІНКА РЕЗУЛЬТАТУ
# =========================================================

def result_is_suspicious(text: str) -> bool:
    if not text:
        return True

    if looks_like_wrong_language(text):
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

        logger.info(
            "🎙️ Починаю транскрипцію: %s",
            filename,
        )

        # =================================================
        # СПРОБА №1
        # Українська як основна мова
        # =================================================

        text_uk = await _request_whisper(
            file_path=file_path,
            filename=filename,
            language="uk",
        )

        logger.info(
            "📝 Результат uk: %s",
            text_uk,
        )

        # Якщо українська вийшла нормально —
        # нічого більше не робимо.
        if text_uk and not result_is_suspicious(text_uk):
            return text_uk

        # =================================================
        # СПРОБА №2
        # Автовизначення мови
        # =================================================

        logger.warning(
            "⚠️ Результат uk підозрілий. "
            "Запускаю повторне розпізнавання з auto-detect."
        )

        text_auto = await _request_whisper(
            file_path=file_path,
            filename=filename,
            language=None,
        )

        logger.info(
            "📝 Результат auto: %s",
            text_auto,
        )

        if text_auto and not result_is_suspicious(text_auto):
            return text_auto

        # =================================================
        # СПРОБА №3
        # Примусова російська
        # =================================================

        logger.warning(
            "⚠️ Auto результат теж підозрілий. "
            "Запускаю повторне розпізнавання з language='ru'."
        )

        text_ru = await _request_whisper(
            file_path=file_path,
            filename=filename,
            language="ru",
        )

        logger.info(
            "📝 Результат ru: %s",
            text_ru,
        )

        # =================================================
        # ВИБІР РЕЗУЛЬТАТУ
        # =================================================

        candidates = [
            ("uk", text_uk),
            ("auto", text_auto),
            ("ru", text_ru),
        ]

        valid_candidates = [
            (name, text)
            for name, text in candidates
            if text and not result_is_suspicious(text)
        ]

        if valid_candidates:
            # Перевага українському результату.
            for name, text in valid_candidates:
                if name == "uk":
                    return text

            return valid_candidates[0][1]

        # Якщо всі результати підозрілі,
        # краще повернути український, якщо він є.
        if text_uk:
            return text_uk

        if text_auto:
            return text_auto

        return text_ru


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

            "file": (
                filename,
                audio_file,
            ),

            "response_format": "json",

            # Мінімізуємо випадкові вигадки.
            "temperature": 0.0,

            "prompt": (
                "Це реальна розмовна мова. "
                "Основна мова — українська. "
                "Також можлива російська мова та суржик. "
                "Потрібна дослівна транскрипція того, що реально чути. "
                "Не перекладай текст. "
                "Не вигадуй слова. "
                "Не змінюй мову мовця. "
                "Зберігай сленг, матюки, імена, назви та авторську лексику."
            ),
        }

        if language:
            kwargs["language"] = language

        result = await client.audio.transcriptions.create(
            **kwargs
        )

    text = getattr(result, "text", "") or ""

    return text.strip()


# =========================================================
# РОЗБИВАННЯ ТЕКСТУ ДЛЯ TELEGRAM
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

            cut = paragraph.rfind(
                " ",
                0,
                max_length,
            )

            if cut <= 0:
                cut = max_length

            parts.append(
                paragraph[:cut].strip()
            )

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
# ВІДПРАВКА ТРАНСКРИПЦІЇ
# =========================================================

async def send_transcription(
    message,
    text: str,
):

    parts = split_text(text)

    for index, part in enumerate(parts):

        safe_text = html.escape(part)

        result_text = (
            f"<blockquote expandable>"
            f"{safe_text}"
            f"</blockquote>"
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
# ОБРОБКА АУДІО / VOICE
# =========================================================

async def handle_audio(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    message = update.message

    if not message:
        return

    # =====================================================
    # ВИЗНАЧАЄМО ТИП ФАЙЛУ
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
            "❌ Файл занадто великий "
            "(ліміт 25 МБ).",
            reply_to_message_id=message.message_id,
        )

        return

    temp_path = None
    status_message = None

    try:

        # =================================================
        # TYPING
        # =================================================

        await context.bot.send_chat_action(
            chat_id=message.chat_id,
            action=ChatAction.TYPING,
        )

        # =================================================
        # СТАТУС
        # =================================================

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
        # ТИМЧАСОВИЙ ФАЙЛ
        # =================================================

        suffix = (
            os.path.splitext(filename)[1]
            or ".ogg"
        )

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name

        # =================================================
        # ЗАВАНТАЖЕННЯ
        # =================================================

        await telegram_file.download_to_drive(
            custom_path=temp_path
        )

        if (
            not os.path.exists(temp_path)
            or os.path.getsize(temp_path) == 0
        ):
            raise RuntimeError(
                "Не вдалося завантажити файл "
                "або він порожній."
            )

        # =================================================
        # ТРАНСКРИПЦІЯ
        # =================================================

        text = await transcribe_audio(
            temp_path,
            filename,
        )

        # =================================================
        # НЕМАЄ ТЕКСТУ
        # =================================================

        if not text:

            await status_message.edit_text(
                "🤷 Не вдалося розібрати слова."
            )

            return

        # =================================================
        # ВИДАЛЯЄМО СТАТУС
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
            "✅ Успішно оброблено: %s",
            filename,
        )

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

    finally:

        # =================================================
        # ВИДАЛЯЄМО ТИМЧАСОВИЙ ФАЙЛ
        # =================================================

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            try:

                os.remove(temp_path)

            except Exception:

                pass


# =========================================================
# ОБРОБКА ПОМИЛОК
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
# ЗАПУСК
# =========================================================

def main():

    logger.info(
        "🚀 Запуск покращеного WhisperBot..."
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

    # Помилки
    app.add_error_handler(
        error_handler
    )

    logger.info(
        "✅ Бот успішно запущений "
        "і готовий до роботи."
    )

    app.run_polling(
        drop_pending_updates=True
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
