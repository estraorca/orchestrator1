import asyncio
import os
from dotenv import load_dotenv
import logging

from google import genai
from groq import Groq
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ========================
# 1. Configurazione
# ========================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

MODELLO_GEMINI = "gemini-3.6-flash"
MODELLO_GROQ = "openai/gpt-oss-120b"

# Telegram ha un limite di 4096 caratteri per messaggio
MAX_LEN = 4000

# ========================
# 2. Funzioni chiamate AI
# ========================

async def chiama_gemini(prompt):
    try:
        response = await gemini_client.aio.models.generate_content(
            model=MODELLO_GEMINI, contents=prompt
        )
        return {"modello": "Gemini", "testo": response.text}
    except Exception as e:
        return {"modello": "Gemini", "errore": str(e)}

async def chiama_groq(prompt):
    try:
        response = groq_client.chat.completions.create(
            model=MODELLO_GROQ,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"modello": "Groq", "testo": response.choices[0].message.content}
    except Exception as e:
        return {"modello": "Groq", "errore": str(e)}

# ========================
# 3. L'Orchestratore
# ========================

async def orchestratore(problema):
    risultati = await asyncio.gather(
        chiama_gemini(problema),
        chiama_groq(problema)
    )
    risposte_valide = [r for r in risultati if "errore" not in r]

    if not risposte_valide:
        errori = "\n".join([f"❌ {r['modello']}: {r['errore'][:100]}" for r in risultati])
        return f"⚠️ Tutti i modelli hanno fallito.\n\n{errori}"

    if len(risposte_valide) == 1:
        r = risposte_valide[0]
        return f"[{r['modello']}]\n\n{r['testo']}"

    gemini = risposte_valide[0]
    groq = risposte_valide[1]

    # Confronto per similarità (parole in comune)
    parole_gemini = set(gemini["testo"].lower().split())
    parole_groq = set(groq["testo"].lower().split())
    if parole_gemini and parole_groq:
        similarita = len(parole_gemini & parole_groq) / max(len(parole_gemini), len(parole_groq))
        if similarita >= 0.7:
            return f"[Gemini + Groq concordi]\n\n{gemini['testo']}"

    # Risposte divergenti: chiedo a Gemini di sintetizzarle
    prompt_sintesi = f"""Ho posto questa domanda: "{problema}"

Due modelli AI hanno risposto così:

Risposta A (Gemini):
{gemini['testo']}

Risposta B (Groq):
{groq['testo']}

Scrivi un'unica risposta finale che integri il meglio di entrambe, risolva eventuali contraddizioni e sia chiara e completa. Rispondi direttamente con la sintesi."""

    sintesi = await chiama_gemini(prompt_sintesi)
    if "errore" in sintesi:
        return f"⚖️ Risposte divergenti:\n\n🔷 [Gemini]\n{gemini['testo']}\n\n🔶 [Groq]\n{groq['testo']}"
    return f"[Sintesi di Gemini + Groq]\n\n{sintesi['testo']}"

# ========================
# 4. Funzione di invio con split
# ========================

async def invia_messaggio_lungo(update: Update, testo: str):
    """Invia un messaggio, spezzandolo se supera il limite di Telegram."""
    if len(testo) <= MAX_LEN:
        await update.message.reply_text(testo)
        return

    # Spezza il testo in blocchi da MAX_LEN caratteri
    for i in range(0, len(testo), MAX_LEN):
        chunk = testo[i:i + MAX_LEN]
        await update.message.reply_text(chunk)

# ========================
# 5. Gestione Bot Telegram
# ========================

logging.basicConfig(level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Ciao! Sono il tuo orchestratore AI.\n"
        "Mandami una domanda e la elaborerò con Gemini 3.6 e Groq."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    await update.message.reply_text("🧠 Sto consultando Gemini e Groq...")
    try:
        risposta = await orchestratore(user_message)
        await invia_messaggio_lungo(update, risposta)
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {str(e)}")

# ========================
# 6. Avvio (Webhook per Render, Polling in locale)
# ========================

def main():
    PORT = int(os.environ.get("PORT", 10000))

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if RENDER_EXTERNAL_URL:
        print(f"🤖 Bot in esecuzione con Webhook su {RENDER_EXTERNAL_URL}")
        webhook_url = f"{RENDER_EXTERNAL_URL}/{TELEGRAM_TOKEN}"
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TELEGRAM_TOKEN,
            webhook_url=webhook_url
        )
    else:
        print("🤖 Bot in esecuzione in locale (polling)...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
