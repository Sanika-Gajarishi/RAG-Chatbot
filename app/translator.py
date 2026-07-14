from langdetect import detect
from deep_translator import GoogleTranslator


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Returns language code like 'en', 'hi', 'fr'.
    """
    try:
        return detect(text)
    except:
        return "en"


def translate_to_english(text: str):
    """
    Translate input text to English if required.
    Returns:
        translated_text,
        original_language
    """
    language = detect_language(text)

    if language == "en":
        return text, language

    translated = GoogleTranslator(
        source="auto",
        target="en"
    ).translate(text)

    return translated, language


def translate_from_english(text: str, target_language: str):
    """
    Translate English answer back to user's language.
    """

    if target_language == "en":
        return text

    return GoogleTranslator(
        source="en",
        target=target_language
    ).translate(text)