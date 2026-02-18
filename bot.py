import asyncio
import os
import re
import logging
from datetime import datetime, timezone
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
SUPPORT_GROUP_ID = os.environ.get("SUPPORT_GROUP_ID")  # Optional until configured
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

If you still cannot help, respond with ONLY this tag and nothing else:
[ESCALATE:category]

Categories:
- commission: Commission disputes, missing payouts, wrong amounts
- technical: Platform bugs, errors, things not working
- account: Account access, verification issues, locked accounts
- copy_trading: Copy trading or PAMM issues that need human help
- general: Anything else that needs human support

Example: If someone says their commissions are wrong and you cannot resolve it, respond with only:
[ESCALATE:commission]

Do NOT add any other text before or after the tag. Just the tag.

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

# --- Ticket system ---

ESCALATE_PATTERN = re.compile(r"\[ESCALATE:(\w+)\]")

CATEGORY_LABELS = {
    "commission": "Commission Issue",
    "technical": "Technical Issue",
    "account": "Account Issue",
    "copy_trading": "Copy Trading Issue",
    "general": "General Support",
}

QUALIFICATION_QUESTIONS = {
    "commission": [
        "What is the email address on your IB account?",
        "Which client or date range is this about?",
        "What amount did you expect vs what you see in your dashboard?",
    ],
    "technical": [
        "What page or feature were you using when the issue happened?",
        "What exactly happened? (error message, blank screen, etc.)",
    ],
    "account": [
        "What is the email address on your account?",
        "Describe what you need help with in one message.",
    ],
    "copy_trading": [
        "Are you trying to follow a trader or set up as a Master?",
        "What specifically is the issue? (can't follow, error, no results, etc.)",
    ],
    "general": [
        "Give me a quick summary of what you need help with so I can pass it to our team.",
    ],
}

# {chat_id: {"category", "questions", "answers", "current_q", "original_query",
#             "user_id", "username", "first_name", "context"}}
ticket_states: dict[int, dict] = {}


def start_qualification(chat_id: int, category: str, original_query: str,
                        user, context_lines: list[str]) -> str:
    """Begin the qualification flow and return the first question."""
    questions = QUALIFICATION_QUESTIONS.get(category, QUALIFICATION_QUESTIONS["general"])
    ticket_states[chat_id] = {
        "category": category,
        "questions": questions,
        "answers": [],
        "current_q": 0,
        "original_query": original_query,
        "user_id": user.id,
        "username": user.username or "",
        "first_name": user.first_name or "",
        "context": context_lines,
    }
    intro = (
        "I want to make sure our support team has everything they need to help you quickly. "
        "Let me ask a couple of questions.\n\n"
    )
    return intro + questions[0]


def handle_ticket_response(chat_id: int, user_text: str) -> str | None:
    """Process a user's answer during qualification. Returns the next message or None if done."""
    state = ticket_states.get(chat_id)
    if not state:
        return None

    # Cancel keywords
    if user_text.lower().strip() in ("cancel", "nevermind", "never mind", "stop", "exit"):
        del ticket_states[chat_id]
        return "No problem, ticket cancelled. You can ask me anything else or just start over."

    # Record answer
    state["answers"].append(user_text)
    state["current_q"] += 1

    # More questions?
    if state["current_q"] < len(state["questions"]):
        return state["questions"][state["current_q"]]

    # All questions answered - ticket is ready to send
    return None


def build_ticket_message(state: dict) -> str:
    """Format the ticket message for the support group."""
    category_label = CATEGORY_LABELS.get(state["category"], state["category"])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    username_display = f"@{state['username']}" if state["username"] else "No username"
    name_display = state["first_name"] or "Unknown"

    # Build details from Q&A
    details_lines = []
    for q, a in zip(state["questions"], state["answers"]):
        details_lines.append(f"  Q: {q}\n  A: {a}")
    details = "\n\n".join(details_lines)

    # Recent conversation context
    context_section = ""
    if state.get("context"):
        context_section = "\nRecent conversation:\n" + "\n".join(state["context"]) + "\n"

    ticket = (
        f"NEW SUPPORT TICKET\n"
        f"{'=' * 30}\n\n"
        f"From: {username_display} ({name_display})\n"
        f"Telegram ID: {state['user_id']}\n"
        f"Category: {category_label}\n"
        f"Time: {now}\n\n"
        f"Original question:\n\"{state['original_query']}\"\n\n"
        f"Details collected:\n{details}\n"
        f"{context_section}\n"
        f"Click {username_display} to message them directly."
    )
    return ticket


async def send_ticket_to_support(bot, state: dict) -> bool:
    """Send formatted ticket to the support group. Returns True on success."""
    if not SUPPORT_GROUP_ID:
        logger.warning("SUPPORT_GROUP_ID not set, cannot send ticket")
        return False

    ticket_text = build_ticket_message(state)
    try:
        await bot.send_message(chat_id=int(SUPPORT_GROUP_ID), text=ticket_text)
        return True
    except Exception:
        logger.exception("Failed to send ticket to support group")
        return False


def get_recent_context(chat_id: int, max_turns: int = 3) -> list[str]:
    """Extract the last few exchanges from chat history for ticket context."""
    history = chat_histories.get(chat_id, [])
    lines = []
    # history is list of {"role": "user"/"model", "parts": [...]}
    for entry in history[-(max_turns * 2):]:
        role = entry.get("role", "unknown")
        parts = entry.get("parts", [])
        text = ""
        for part in parts:
            if isinstance(part, str):
                text = part
            elif hasattr(part, "text"):
                text = part.text
        if text:
            label = "User" if role == "user" else "Bot"
            # Truncate long messages
            if len(text) > 200:
                text = text[:200] + "..."
            lines.append(f"  {label}: {text}")
    return lines


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

    # --- Ticket qualification flow (active) ---
    if chat_id in ticket_states:
        if not is_private:
            await message.reply_text(
                "Please DM me directly to continue with your support ticket. "
                "Tap my name and hit 'Message' to open a private chat."
            )
            return

        next_msg = handle_ticket_response(chat_id, user_text)
        if next_msg:
            # More questions to ask
            await message.reply_text(next_msg)
        else:
            # Qualification complete - send ticket
            state = ticket_states.pop(chat_id)
            sent = await send_ticket_to_support(message.get_bot(), state)
            if sent:
                await message.reply_text(
                    "Got it! I have passed your details to our support team. "
                    "Someone will reach out to you directly on Telegram shortly.\n\n"
                    "If you do not hear back within a few hours, you can also "
                    "email support@ntwmarkets.com."
                )
            else:
                await message.reply_text(
                    "I was not able to submit your ticket right now. "
                    "Please reach out directly to support@ntwmarkets.com "
                    "and include the details you just shared with me."
                )
        return

    # --- Normal Gemini flow ---
    try:
        reply = get_gemini_response(chat_id, user_text)
    except Exception:
        logger.exception("Gemini API error")
        await message.reply_text("Sorry, something went wrong. Please try again.")
        return

    # --- Check for escalation tag ---
    match = ESCALATE_PATTERN.search(reply)
    if match and SUPPORT_GROUP_ID:
        category = match.group(1).lower()
        if category not in QUALIFICATION_QUESTIONS:
            category = "general"

        if not is_private:
            # In groups, direct user to DM the bot
            await message.reply_text(
                "I think our support team can help with this. "
                "Please send me a direct message so I can gather your details "
                "and create a ticket. Tap my name and hit 'Message'."
            )
            return

        # Start qualification in private chat
        context_lines = get_recent_context(chat_id)
        first_question = start_qualification(
            chat_id, category, user_text, message.from_user, context_lines
        )
        await message.reply_text(first_question)
    elif match and not SUPPORT_GROUP_ID:
        # Escalation triggered but no support group configured - fall back
        await message.reply_text(
            "I think our support team can help with this. "
            "Please reach out to support@ntwmarkets.com or the #support "
            "channel on Telegram and they will get back to you."
        )
    else:
        # Normal response
        await message.reply_text(reply)


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
