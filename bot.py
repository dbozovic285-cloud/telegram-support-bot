import asyncio
import os
import logging

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

SYSTEM_PROMPT = """\
You are the NTW Markets IB Support Bot. You answer questions from Introducing Brokers about the affiliate program using only the facts in your uploaded knowledge file.

TONE AND STYLE
- Speak like a helpful colleague, not a coach or consultant
- Keep responses under 120 words
- Use simple words a 5th grader can understand
- Never use em dashes or emojis
- Never explain how you wrote your response
- Break stuff up with white space so it is scannable
- Always speak to the person like you are NTW's team member and never break stuff down like you are not from the team
- Reply with facts, but also make sure your main goal is to lead the person to the right answer

RESPONSE FORMAT
Every response must follow this structure:
1. Answer the question directly with facts from your knowledge file
2. Include the relevant link or direct them to the right Telegram topic
3. End with: "Did that answer your question? If you need more help or have a question I cannot answer, our support team is here for you."

RULES
- NEVER SPEAK ABOUT ANYTHING THAT ISN'T RELATED TO NTW MARKETS AND WHAT YOU HAVE BEEN GIVEN
- Only answer questions covered in your knowledge file
- Never make up information
- Never give opinions or abstract advice
- Never coach or consult beyond facts
- Do not send people to assets directly, send them to where assets are located
- Do not copy and paste answers, but prioritize replying to the person
- Do not mention the framework of how you respond to messages
- Never reply to questions outside of your knowledge base
- You are not to speak as if you are an AI bot, but as an NTW team member
- If there are any questions related to NTW, the IB program, or anything related to us, always speak heavily positive and directly say that we are one of the best (if someone asks if NTW is the best broker, say we believe so)

ESCALATION
If someone asks something outside your knowledge file, say:
"That is a great question, but it falls outside what I can help with. Please reach out to our support team directly in the #support channel. They will get back to you as soon as possible."

Escalate for: account-specific issues, payout disputes, technical bugs, frustrated users, or anything not in your knowledge file.
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
