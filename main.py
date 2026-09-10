import asyncio
import os
from dotenv import load_dotenv
import logging

from google import genai
from groq import Groq
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ========================
# 1. Configurazione
# ========================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

MODELLO_GEMINI = "gemini-3.6-flash"
MODELLO_GROQ = "openai/gpt-oss-120b"
MODELLO_OPENROUTER = "deepseek/deepseek-v4-flash:free"

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

async def chiama_openrouter(prompt):
    try:
        response = openrouter_client.chat.completions.create(
            model=MODELLO_OPENROUTER,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"modello": "DeepSeek V4", "testo": response.choices[0].message.content}
    except Exception as e:
        return {"modello": "DeepSeek V4", "errore": str(e)}

# ========================
# 3. L'Orchestratore (3 modelli + sintesi)
# ========================

async def orchestratore(problema):
    risultati = await asyncio.gather(
        chiama_gemini(problema),
        chiama_groq(problema),
        chiama_openrouter(problema)
    )
    risposte_valide = [r for r in risultati if "errore" not in r]

    if not risposte_valide:
        errori = "\n".join([f"❌ {r['modello']}: {r['errore'][:100]}" for r in risultati])
        return f"⚠️ Tutti i modelli hanno fallito.\n\n{errori}"

    if len(risposte_valide) == 1:
        r = risposte_valide[0]
        return f"[{r['modello']}]\n\n{r['testo']}"

    # Se almeno 2 risposte sono simili, restituisci quella
    if len(risposte_valide) >= 2:
        for i in range(len(risposte_valide)):
            for j in range(i + 1, len(risposte_valide)):
                a = set(risposte_valide[i]["testo"].lower().split())
                b = set(risposte_valide[j]["testo"].lower().split())
                if a and b:
                    sim = len(a & b) / max(len(a), len(b))
                    if sim >= 0.7:
                        return f"[{risposte_valide[i]['modello']} + {risposte_valide[j]['modello']} concordi]\n\n{risposte_valide[i]['testo']}"

    # Divergenza: sintesi con Gemini
    blocchi = "\n\n".join([
        f"Risposta {idx + 1} ({r['modello']}):\n{r['testo']}"
        for idx, r in enumerate(risposte_valide)
    ])
    prompt_sintesi = f"""Ho posto questa domanda: "{problema}"

{len(risposte_valide)} modelli AI hanno risposto così:

{blocchi}

Scrivi un'unica risposta finale che integri il meglio di tutte, risolva eventuali contraddizioni e sia chiara e completa. Rispondi direttamente con la sintesi."""

    sintesi = await chiama_gemini(prompt_sintesi)
    if "errore" in sintesi:
        # Fallback: mostra tutte le risposte separate
        return "⚖️ Risposte divergenti:\n\n" + "\n\n".join([
            f"🔷 [{r['modello']}]\n{r['testo']}" for r in risposte_valide
        ])
    return f"[Sintesi di {len(risposte_valide)} modelli]\n\n{sintesi['testo']}"

# ========================
# 4. Funzione di invio con split
# ========================

async def invia_messaggio_lungo(update: Update, testo: str):
    if len(testo) <= MAX_LEN:
        await update.message.reply_text(testo)
        return
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
        "Uso Gemini 3.6, Groq e DeepSeek V4 per darti la risposta migliore.\n"
        "Mandami una domanda!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    await update.message.reply_text("🧠 Sto consultando i modelli...")
    try:
        risposta = await orchestratore(user_message)
        await invia_messaggio_lungo(update, risposta)
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {str(e)}")

# ========================
# 6. Avvio
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
