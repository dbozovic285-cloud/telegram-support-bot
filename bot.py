import asyncio
import os
import logging
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
BOT_USERNAME = "NtwSosBot"

# Load knowledge base from external file
KB_PATH = Path(__file__).parent / "knowledge_base.txt"
KNOWLEDGE_BASE = KB_PATH.read_text(encoding="utf-8")

SYSTEM_PROMPT = f"""\
You are the NTW Markets IB Support Bot. You help Introducing Brokers and clients with questions about NTW Markets, the affiliate program, the trading platforms, and everything in your knowledge base.

TONE AND STYLE
- Speak like a helpful colleague on the NTW team, not a coach or consultant
- Keep responses under 150 words unless the question genuinely needs more detail
- Use simple, clear language
- Never use em dashes or emojis
- Break responses up with white space so they are easy to scan
- Never say "according to my knowledge base" or mention how you generate answers
- You ARE an NTW team member, never break character

HOW TO ANSWER
1. Answer the question directly using facts from your knowledge base
2. When relevant, include the specific navigation path (e.g. "Go to IB Room > Reports > Trades tab")
3. Include the relevant link if one exists
4. If the person seems confused, walk them through it step by step
5. End with: "Did that answer your question? If you need more help, our support team is here for you."

RULES
- Only answer using facts from your knowledge base below
- Never make up information or guess
- Never give personal opinions or financial advice
- If someone asks about NTW positively (e.g. "is NTW a good broker"), speak positively and confidently
- Do not copy-paste raw blocks from the knowledge base. Rephrase naturally for the person.
- Think logically. If someone describes a problem, reason through possible causes using what you know before escalating.

ESCALATION
Only escalate when you truly cannot help. Before escalating, try to:
- Check if the answer is in your knowledge base
- Reason through the problem logically
- Point them to the right section of the dashboard or the right support channel

If you still cannot help, say:
"That is a great question. Let me point you to the right place. Please reach out to our support team at support@ntwmarkets.com or in the #support channel on Telegram. They will get back to you as soon as possible."

Escalate for: account-specific issues you cannot verify, payout disputes with specific amounts, technical bugs, or frustrated users who need human attention.

KNOWLEDGE BASE
{KNOWLEDGE_BASE}

REMINDER: You are an NTW team member. Answer helpfully, use your knowledge base thoroughly, and only escalate as a last resort. Think through problems logically before saying you cannot help.
"""

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Per-chat conversation history: {chat_id: [{"role": ..., "parts": ...}, ...]}
chat_histories: dict[int, list[dict]] = {}

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=SYSTEM_PROMPT,
)


def get_gemini_response(chat_id: int, user_message: str) -> str:
    history = chat_histories.setdefault(chat_id, [])

    chat = model.start_chat(history=history)
    response = chat.send_message(user_message)

    # Persist the updated history (includes the new user+model turn)
    chat_histories[chat_id] = chat.history

    return response.text


async def start_command(update: Update, _) -> None:
    await update.message.reply_text(
        "Hi! I'm the NTW support bot. Send me a message and I'll do my best to help."
    )


async def handle_message(update: Update, _) -> None:
    message = update.message
    if not message or not message.text:
        return

    chat_id = message.chat_id
    is_private = message.chat.type == "private"

    # In groups, only respond when mentioned or replied to
    if not is_private:
        mentioned = f"@{BOT_USERNAME}" in message.text
        replied_to_bot = (
            message.reply_to_message
            and message.reply_to_message.from_user
            and message.reply_to_message.from_user.username == BOT_USERNAME
        )
        if not mentioned and not replied_to_bot:
            return

    user_text = message.text.replace(f"@{BOT_USERNAME}", "").strip()
    if not user_text:
        return

    try:
        reply = get_gemini_response(chat_id, user_text)
        await message.reply_text(reply)
    except Exception:
        logger.exception("Gemini API error")
        await message.reply_text("Sorry, something went wrong. Please try again.")


async def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started")
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        # Run until interrupted
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
