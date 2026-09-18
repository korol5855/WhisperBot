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


# =========================================================
# МОДЕЛЬ
# =========================================================

WHISPER_MODEL = "whisper-large-v3"


# =========================================================
# ЛІМІТИ
# =========================================================

# Максимум одночасних транскрипцій
TRANSCRIPTION_LIMIT = 3

transcription_semaphore = asyncio.Semaphore(
    TRANSCRIPTION_LIMIT
)


# Максимальний розмір аудіо
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


# Запас до Telegram-ліміту повідомлення
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
        "на текст без перекладу та виправлення мови.",
        parse_mode="HTML",
    )


# =========================================================
# ПЕРЕВІРКА НА ЛАТИНСЬКИЙ ТРАНСЛІТ
# =========================================================

def looks_like_latin_translit(text: str) -> bool:
    """
    Виявляє ситуацію, коли Whisper замість кирилиці
    раптом повернув щось на кшталт:

        A to e havaryu

    Важливо:
    ми НЕ виправляємо такий текст самостійно.
    Просто робимо повторний запит до Whisper.
    """

    if not text:
        return False

    latin_letters = sum(
        1
        for char in text
        if char.lower() in "abcdefghijklmnopqrstuvwxyz"
    )

    cyrillic_letters = sum(
        1
        for char in text
        if char.lower()
        in (
            "абвгдежзийклмнопрстуфхцчшщьюя"
            "іїєґ"
        )
    )

    # Якщо є достатньо латинських літер,
    # але немає кирилиці — дуже схоже на трансліт.
    if latin_letters >= 4 and cyrillic_letters == 0:

        logger.warning(
            "⚠️ Whisper повернув латиницю замість кирилиці: %s",
            text,
        )

        return True

    return False


# =========================================================
# ПЕРЕВІРКА НА ЯВНУ ГАЛЮЦИНАЦІЮ
# =========================================================

HALLUCINATION_PHRASES = [
    "thanks for watching",
    "thank you for watching",
    "like and subscribe",
    "subscribe",
    "thank you for listening",
    "дякую за перегляд",
    "підписуйтесь на канал",
    "підписуйся на канал",
]


def looks_like_hallucination(text: str) -> bool:
    """
    Ловимо тільки очевидні шаблони галюцинацій.

    НЕ перевіряємо польську.
    НЕ перевіряємо окремі слова.
    НЕ намагаємося визначати мову через regex.
    """

    if not text:
        return False

    text_lower = text.lower().strip()

    for phrase in HALLUCINATION_PHRASES:

        if phrase in text_lower:

            logger.warning(
                "⚠️ Можлива галюцинація Whisper: %s",
                phrase,
            )

            return True

    return False


# =========================================================
# ЗАГАЛЬНА ПЕРЕВІРКА РЕЗУЛЬТАТУ
# =========================================================

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

        logger.info(
            "🎙️ Починаю транскрипцію: %s",
            filename,
        )

        # =================================================
        # СПРОБА №1
        # Українська як основна мова
        # =================================================

        text = await _request_whisper(
            file_path=file_path,
            filename=filename,
            language="uk",
        )

        logger.info(
            "📝 Основний результат: %s",
            text,
        )

        # Якщо все нормально —
        # одразу повертаємо результат.
        if text and not result_is_suspicious(text):

            return text


        # =================================================
        # СПРОБА №2
        # AUTO-DETECT
        # =================================================

        if looks_like_latin_translit(text):

            logger.warning(
                "⚠️ Виявлено латинський трансліт."
                " Повторюю через auto-detect."
            )

        elif looks_like_hallucination(text):

            logger.warning(
                "⚠️ Виявлено можливу галюцинацію."
                " Повторюю через auto-detect."
            )

        else:

            logger.warning(
                "⚠️ Основний результат порожній."
                " Повторюю через auto-detect."
            )


        text_auto = await _request_whisper(
            file_path=file_path,
            filename=filename,
            language=None,
        )

        logger.info(
            "📝 Auto результат: %s",
            text_auto,
        )


        # Якщо auto дав нормальний результат —
        # використовуємо його.
        if text_auto and not result_is_suspicious(text_auto):

            return text_auto


        # =================================================
        # ЗАПОБІЖНИК
        # =================================================

        # Якщо повтор теж дивний,
        # не робимо третій запит.
        #
        # Це важливо:
        # не витрачаємо API на нескінченні спроби
        # та не починаємо "виправляти" слова самостійно.

        if text:
            return text

        return text_auto


# =========================================================
# ЗАПИТ ДО WHISPER
# =========================================================

async def _request_whisper(
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

            # Максимально стабільний режим
            "temperature": 0.0,

            # =================================================
            # ГОЛОВНИЙ PROMPT
            # =================================================
            #
            # Тут НЕ кажемо "переклади".
            #
            # Навпаки:
            # просто записати те, що сказано.
            #
            "prompt": (
                "Це жива розмовна українська мова. "
                "У розмові можуть бути російські слова, "
                "російські фрази та суржик. "
                "Транскрибуй дослівно те, що говорить людина. "
                "Не перекладай. "
                "Не перефразовуй. "
                "Не виправляй граматику. "
                "Не замінюй російські слова українськими. "
                "Не замінюй українські слова російськими. "
                "Не виправляй суржик на літературну мову. "
                "Зберігай сленг, матюки, скорочення, "
                "розмовні слова та імена. "
                "Записуй текст кирилицею."
            ),
        }


        # =================================================
        # LANGUAGE
        # =================================================

        if language:

            kwargs["language"] = language


        # =================================================
        # ЗАПИТ
        # =================================================

        result = await client.audio.transcriptions.create(
            **kwargs
        )


    # =====================================================
    # ОТРИМУЄМО ТЕКСТ
    # =====================================================

    text = getattr(
        result,
        "text",
        "",
    ) or ""


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


            paragraph = paragraph[
                cut:
            ].strip()


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

            current += (
                "\n"
                + paragraph
            )


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

        safe_text = html.escape(
            part
        )


        result_text = (
            "<blockquote expandable>"
            f"{safe_text}"
            "</blockquote>"
        )


        # Перше повідомлення —
        # відповідь саме на голосове.
        if index == 0:

            await message.reply_text(
                result_text,
                parse_mode="HTML",
                reply_to_message_id=(
                    message.message_id
                ),
            )


        # Наступні частини —
        # звичайними повідомленнями.
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
        and
        telegram_media.file_size
        > MAX_FILE_SIZE
    ):

        await message.reply_text(
            "❌ Файл занадто великий "
            "(ліміт 25 МБ).",
            reply_to_message_id=(
                message.message_id
            ),
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

        status_message = (
            await message.reply_text(
                "🎙️ <i>Розпізнаю...</i>",
                parse_mode="HTML",
                reply_to_message_id=(
                    message.message_id
                ),
            )
        )


        # =================================================
        # ОТРИМУЄМО ФАЙЛ TELEGRAM
        # =================================================

        telegram_file = (
            await context.bot.get_file(
                telegram_media.file_id
            )
        )


        # =================================================
        # ТИМЧАСОВИЙ ФАЙЛ
        # =================================================

        suffix = (
            os.path.splitext(
                filename
            )[1]
            or ".ogg"
        )


        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_path = (
                temp_file.name
            )


        # =================================================
        # ЗАВАНТАЖУЄМО
        # =================================================

        await telegram_file.download_to_drive(
            custom_path=temp_path
        )


        # =================================================
        # ПЕРЕВІРЯЄМО ФАЙЛ
        # =================================================

        if (
            not os.path.exists(
                temp_path
            )
            or
            os.path.getsize(
                temp_path
            ) == 0
        ):

            raise RuntimeError(
                "Не вдалося завантажити "
                "файл або він порожній."
            )


        # =================================================
        # ТРАНСКРИПЦІЯ
        # =================================================

        text = await transcribe_audio(
            file_path=temp_path,
            filename=filename,
        )


        # =================================================
        # ЯКЩО ТЕКСТУ НЕМАЄ
        # =================================================

        if not text:

            await status_message.edit_text(
                "🤷 Не вдалося розібрати слова."
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
        # ВІДПРАВЛЯЄМО ТЕКСТ
        # =================================================

        await send_transcription(
            message,
            text,
        )


        logger.info(
            "✅ Успішно оброблено: %s",
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
                    reply_to_message_id=(
                        message.message_id
                    ),
                )


        except Exception:

            pass


    # =====================================================
    # ВИДАЛЕННЯ ТИМЧАСОВОГО ФАЙЛУ
    # =====================================================

    finally:

        if (
            temp_path
            and
            os.path.exists(
                temp_path
            )
        ):

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass


# =========================================================
# ГЛОБАЛЬНИЙ ОБРОБНИК ПОМИЛОК
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
# ЗАПУСК БОТА
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
        "✅ Бот успішно запущений "
        "і готовий до роботи."
    )


    # =====================================================
    # POLLING
    # =====================================================

    app.run_polling(
        drop_pending_updates=True
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
