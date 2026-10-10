LANG_NAMES = {"en": "English", "hi": "Hindi"}

SYSTEM_PROMPT = """You are {name}, a friendly voice assistant. Everything you write is spoken aloud, so:
- Keep replies short: one to three sentences.
- Use spoken language. No markdown, bullets, emojis, code, or URLs.
- Reply in the user's language. For Hindi, always use Devanagari, never romanized Hindi.
- If an earlier reply ends with "…", you were interrupted there. Do not repeat it."""


def build_messages(history, lang, max_turns, name):
    system = SYSTEM_PROMPT.format(name=name)
    if lang in LANG_NAMES:
        system += f"\n\nThe user speaks {LANG_NAMES[lang]}. Reply in {LANG_NAMES[lang]}."
    recent = history[-max_turns * 2:] if max_turns > 0 else []
    return [{"role": "system", "content": system}, *recent]

