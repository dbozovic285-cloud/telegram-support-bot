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
5. End with: "Did that answer your question?"

RULES
- Only answer using facts from your knowledge base below
- Never make up information or guess
- Never give personal opinions or financial advice
- If someone asks about NTW positively (e.g. "is NTW a good broker"), speak positively and confidently
- Do not copy-paste raw blocks from the knowledge base. Rephrase naturally for the person.
- Think logically. If someone describes a problem, reason through possible causes using what you know before escalating.

ESCALATION
Try to answer using your knowledge base first. Escalate when:
- User explicitly asks to speak with a human, be forwarded to someone, or contact the support team
- Account-specific disputes you cannot verify (wrong commissions, missing payouts with specific amounts)
- Technical bugs you cannot resolve
- Account access issues (locked out, login problems)
- Frustrated users who have already tried everything you suggested

1. If the user is just asking for a human, you do not need to add anything else — just output the tag
2. Otherwise provide whatever helpful info you can, then on the very last line add the tag: [ESCALATE:category]

Categories:
- commission: Commission disputes, missing payouts, wrong amounts
- technical: Platform bugs, errors, things not working
- account: Account access, verification issues, locked accounts
- copy_trading: Copy trading or PAMM issues that need human help
- general: Anything else that needs human support

The [ESCALATE:category] tag must be the very last line, alone, with no text after it.
Do NOT mention support@ntwmarkets.com or any support channels. The bot handles routing to support.

Example:
"Here are some things to check in IB Room > Reports...
[ESCALATE:commission]"

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
        "What is the email address on your account?",
        "Describe exactly what you need help with. Include any error messages, steps you have already tried, and what you expected to happen.",
    ],
}

# {chat_id: {"category", "questions", "answers", "current_q", "stage",
#             "additional_info", "original_query", "user_id", "username",
#             "first_name", "context"}}
ticket_states: dict[int, dict] = {}

# Stores escalation context while waiting for user to type "Ticket"
# {chat_id: {"category": "commission", "original_query": "..."}}
pending_escalations: dict[int, dict] = {}


def start_qualification(chat_id: int, category: str, original_query: str,
                        user, context_lines: list[str]) -> str:
    """Begin the qualification flow and return the first question."""
    questions = QUALIFICATION_QUESTIONS.get(category, QUALIFICATION_QUESTIONS["general"])
    ticket_states[chat_id] = {
        "category": category,
        "questions": questions,
        "answers": [],
        "current_q": 0,
        "stage": "questions",
        "additional_info": "",
        "original_query": original_query,
        "user_id": user.id,
        "username": user.username or "",
        "first_name": user.first_name or "",
        "context": context_lines,
    }
    intro = (
        "I'll create a support ticket for you. Type Exit at any time to cancel.\n\n"
    )
    return intro + questions[0]


def handle_ticket_response(chat_id: int, user_text: str) -> str | None:
    """Process a user response during ticket flow. Returns next message, or None to send ticket."""
    state = ticket_states.get(chat_id)
    if not state:
        return None

    stage = state.get("stage", "questions")
    text = user_text.strip()

    # Only "Exit" cancels - works at any stage
    if text.lower() == "exit":
        del ticket_states[chat_id]
        return "Ticket cancelled. Feel free to ask me anything else."

    if stage == "questions":
        state["answers"].append(text)
        state["current_q"] += 1

        # More category questions?
        if state["current_q"] < len(state["questions"]):
            return state["questions"][state["current_q"]]

        # All category questions done - ask for additional info
        state["stage"] = "additional_info"
        return (
            "Is there anything else you'd like to add that could help our team?\n\n"
            "Add it here, or type Submit to send your ticket now."
        )

    elif stage == "additional_info":
        if text.lower() == "submit":
            # Send with no additional info
            return None
        # Save additional info, ask for submit confirmation
        state["additional_info"] = text
        state["stage"] = "await_submit"
        return "Got it. Type Submit to send your ticket to the team."

    elif stage == "await_submit":
        if text.lower() == "submit":
            return None  # Send the ticket
        return "Type Submit to send your ticket, or Exit to cancel."

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

    additional_section = ""
    if state.get("additional_info"):
        additional_section = f"\nAdditional info:\n  {state['additional_info']}\n"

    ticket = (
        f"NEW SUPPORT TICKET\n"
        f"{'=' * 30}\n\n"
        f"From: {username_display} ({name_display})\n"
        f"Telegram ID: {state['user_id']}\n"
        f"Category: {category_label}\n"
        f"Time: {now}\n\n"
        f"Original question:\n\"{state['original_query']}\"\n\n"
        f"Details collected:\n{details}\n"
        f"{additional_section}"
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
    # history is list of Content proto objects with .role and .parts attributes
    for entry in history[-(max_turns * 2):]:
        try:
            role = getattr(entry, "role", "unknown")
            parts = getattr(entry, "parts", [])
            text = ""
            for part in parts:
                if isinstance(part, str):
                    text = part
                elif hasattr(part, "text"):
                    text = part.text
            if text:
                label = "User" if role == "user" else "Bot"
                if len(text) > 200:
                    text = text[:200] + "..."
                lines.append(f"  {label}: {text}")
        except Exception:
            continue
    return lines


# Per-chat locks so rapid messages queue up instead of racing to Gemini
chat_locks: dict[int, asyncio.Lock] = {}


def _get_chat_lock(chat_id: int) -> asyncio.Lock:
    if chat_id not in chat_locks:
        chat_locks[chat_id] = asyncio.Lock()
    return chat_locks[chat_id]


def _call_gemini(chat_id: int, user_message: str) -> str:
    """Synchronous Gemini call - runs in a thread via asyncio.to_thread."""
    history = chat_histories.setdefault(chat_id, [])
    chat = model.start_chat(history=history)
    response = chat.send_message(user_message)
    chat_histories[chat_id] = chat.history
    return response.text


async def get_gemini_response(chat_id: int, user_message: str) -> str:
    """Async wrapper with retry + backoff for transient Gemini errors."""
    last_exc = None
    for attempt in range(3):
        try:
            return await asyncio.to_thread(_call_gemini, chat_id, user_message)
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            transient = any(k in msg for k in (
                "429", "resource exhausted", "quota",
                "503", "service unavailable", "deadline exceeded",
                "internal", "unavailable",
            ))
            if transient and attempt < 2:
                wait = (attempt + 1) * 3  # 3s then 6s
                logger.warning("Gemini transient error (attempt %d), retry in %ds: %s",
                               attempt + 1, wait, e)
                await asyncio.sleep(wait)
            else:
                break
    raise last_exc


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

    # --- Ticket flow (active - collecting answers) ---
    if chat_id in ticket_states:
        if not is_private:
            await message.reply_text(
                "Please DM me directly to continue with your support ticket. "
                "Tap my name and hit 'Message' to open a private chat."
            )
            return

        next_msg = handle_ticket_response(chat_id, user_text)
        if next_msg:
            await message.reply_text(next_msg)
        else:
            # Flow complete (user typed Submit) - send ticket
            state = ticket_states.pop(chat_id)
            sent = await send_ticket_to_support(message.get_bot(), state)
            if sent:
                await message.reply_text(
                    "Your ticket has been submitted. Our team will reach out to "
                    "you directly on Telegram shortly."
                )
            else:
                await message.reply_text(
                    "I was not able to submit your ticket right now. "
                    "Please email support@ntwmarkets.com with the details you shared."
                )
        return

    # --- "Ticket" keyword - user wants to create a ticket ---
    if user_text.strip().lower() == "ticket":
        if not is_private:
            await message.reply_text(
                "Please send me a direct message to create a support ticket. "
                "Tap my name and hit 'Message'."
            )
            return
        pending = pending_escalations.pop(chat_id, {
            "category": "general",
            "original_query": "User requested support ticket",
        })
        context_lines = get_recent_context(chat_id)
        first_question = start_qualification(
            chat_id, pending["category"], pending["original_query"],
            message.from_user, context_lines
        )
        await message.reply_text(first_question)
        return

    # --- Normal Gemini flow ---
    try:
        async with _get_chat_lock(chat_id):
            reply = await get_gemini_response(chat_id, user_text)
    except Exception:
        logger.exception("Gemini API error")
        await message.reply_text(
            "I am having trouble connecting right now. Please send your message again in a moment."
        )
        return

    # --- Strip escalation tag and store category if present ---
    match = ESCALATE_PATTERN.search(reply)
    clean_reply = ESCALATE_PATTERN.sub("", reply).strip() if match else reply.strip()

    if match and SUPPORT_GROUP_ID and is_private:
        category = match.group(1).lower()
        if category not in QUALIFICATION_QUESTIONS:
            category = "general"
        pending_escalations[chat_id] = {
            "category": category,
            "original_query": user_text,
        }

    # Every reply gets the ticket CTA — no exceptions
    TICKET_CTA = "\n\nType Ticket to reach our support team."
    await message.reply_text(clean_reply + TICKET_CTA)


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
