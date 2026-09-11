import asyncio
import os
import json
import logging

from dotenv import load_dotenv
from google import genai
from groq import Groq
from openai import OpenAI
import redis.asyncio as redis
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
REDIS_URL = os.getenv("REDIS_URL")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
if OPENROUTER_API_KEY:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
else:
    openrouter_client = None

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

MODELLO_GEMINI = "gemini-3.6-flash"
MODELLO_GROQ = "openai/gpt-oss-120b"
MODELLO_OPENROUTER = "openrouter/free"

MAX_LEN = 4000
MAX_HISTORY = 10
TTL_CRONOLOGIA = 604800

# ========================
# 2. Memoria persistente su Redis
# ========================

def chiave_cronologia(chat_id: int) -> str:
    return f"chat:{chat_id}:history"

async def aggiungi_a_cronologia(chat_id: int, ruolo: str, contenuto: str):
    chiave = chiave_cronologia(chat_id)
    messaggio = json.dumps({"role": ruolo, "content": contenuto})
    await redis_client.rpush(chiave, messaggio)
    await redis_client.ltrim(chiave, -MAX_HISTORY, -1)
    await redis_client.expire(chiave, TTL_CRONOLOGIA)

async def recupera_cronologia(chat_id: int) -> list:
    chiave = chiave_cronologia(chat_id)
    messaggi = await redis_client.lrange(chiave, 0, -1)
    return [json.loads(m) for m in messaggi]

async def reset_cronologia(chat_id: int):
    await redis_client.delete(chiave_cronologia(chat_id))

# ========================
# 3. Funzioni chiamate AI (TUTTI con contesto)
# ========================

async def chiama_gemini(chat_id: int, prompt_utente: str):
    """Gemini riceve la cronologia in formato testuale."""
    try:
        history = await recupera_cronologia(chat_id)
        contesto = ""
        if history:
            righe = []
            for msg in history:
                ruolo = "Utente" if msg["role"] == "user" else "Assistente"
                righe.append(f"{ruolo}: {msg['content']}")
            contesto = "Contesto della conversazione:\n" + "\n".join(righe) + "\n\n"

        prompt_completo = f"{contesto}Utente: {prompt_utente}\n\nAssistente:"
        response = await gemini_client.aio.models.generate_content(
            model=MODELLO_GEMINI,
            contents=prompt_completo
        )
        return {"modello": "Gemini", "testo": response.text}
    except Exception as e:
        return {"modello": "Gemini", "errore": str(e)}

async def chiama_groq(chat_id: int, prompt_utente: str):
    """Groq riceve la cronologia in formato messages."""
    try:
        messages = await recupera_cronologia(chat_id)
        messages.append({"role": "user", "content": prompt_utente})

        response = groq_client.chat.completions.create(
            model=MODELLO_GROQ,
            messages=messages
        )
        return {"modello": "Groq", "testo": response.choices[0].message.content}
    except Exception as e:
        return {"modello": "Groq", "errore": str(e)}

async def chiama_openrouter(chat_id: int, prompt_utente: str):
    """OpenRouter riceve la cronologia in formato messages."""
    if openrouter_client is None:
        return {"modello": "OpenRouter", "errore": "OpenRouter non configurato"}
    try:
        messages = await recupera_cronologia(chat_id)
        messages.append({"role": "user", "content": prompt_utente})

        response = openrouter_client.chat.completions.create(
            model=MODELLO_OPENROUTER,
            messages=messages
        )
        return {"modello": "OpenRouter", "testo": response.choices[0].message.content}
    except Exception as e:
        return {"modello": "OpenRouter", "errore": str(e)}

# ========================
# 4. L'Orchestratore
# ========================

async def orchestratore(chat_id: int, problema: str):
    risultati = await asyncio.gather(
        chiama_gemini(chat_id, problema),
        chiama_groq(chat_id, problema),
        chiama_openrouter(chat_id, problema)
    )
    risposte_valide = [r for r in risultati if "errore" not in r]

    if not risposte_valide:
        errori = "\n".join([f"❌ {r['modello']}: {r['errore'][:100]}" for r in risultati])
        return f"⚠️ Tutti i modelli hanno fallito.\n\n{errori}"

    if len(risposte_valide) == 1:
        r = risposte_valide[0]
        return f"[{r['modello']}]\n\n{r['testo']}"

    # Confronto per similarità
    for i in range(len(risposte_valide)):
        for j in range(i + 1, len(risposte_valide)):
            a = set(risposte_valide[i]["testo"].lower().split())
            b = set(risposte_valide[j]["testo"].lower().split())
            if a and b:
                sim = len(a & b) / max(len(a), len(b))
                if sim >= 0.7:
                    return f"[{risposte_valide[i]['modello']} + {risposte_valide[j]['modello']} concordi]\n\n{risposte_valide[i]['testo']}"

    # Sintesi con Gemini
    blocchi = "\n\n".join([
        f"Risposta {idx + 1} ({r['modello']}):\n{r['testo']}"
        for idx, r in enumerate(risposte_valide)
    ])
    prompt_sintesi = f"""Domanda dell'utente: "{problema}"

Tre assistenti AI hanno proposto queste risposte:

{blocchi}

Il tuo compito: scrivi UNA SOLA risposta finale per l'utente, in modo diretto e naturale.

REGOLE:
- NON menzionare i nomi dei modelli, NON spiegare chi ha sbagliato, NON fare commenti meta.
- Rispondi come se fossi un unico assistente che ha semplicemente la risposta.
- Se le risposte sono contraddittorie, scegli quella più corretta e ignora le altre.
- Se una risposta è chiaramente sbagliata, ignorala silenziosamente.
- Vai dritto al punto.

Rispondi direttamente con la sintesi, senza preamboli."""

    sintesi = await chiama_gemini(chat_id, prompt_sintesi)
    if "errore" in sintesi:
        return "⚖️ Risposte divergenti:\n\n" + "\n\n".join([
            f"🔷 [{r['modello']}]\n{r['testo']}" for r in risposte_valide
        ])
    return f"[Sintesi di {len(risposte_valide)} modelli]\n\n{sintesi['testo']}"

# ========================
# 5. Invio messaggi lunghi
# ========================

async def invia_messaggio_lungo(update: Update, testo: str):
    if len(testo) <= MAX_LEN:
        await update.message.reply_text(testo)
        return
    for i in range(0, len(testo), MAX_LEN):
        chunk = testo[i:i + MAX_LEN]
        await update.message.reply_text(chunk)

# ========================
# 6. Gestione Bot Telegram
# ========================

logging.basicConfig(level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await reset_cronologia(chat_id)
    await update.message.reply_text(
        "👋 Ciao! Sono il tuo orchestratore AI.\n"
        "Uso Gemini 3.6, Groq e OpenRouter.\n\n"
        "🧠 Memoria persistente attiva (7 giorni, ultimi 5 scambi).\n"
        "Usa /reset per ricominciare da capo."
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await reset_cronologia(chat_id)
    await update.message.reply_text("🧹 Memoria cancellata! Ricominciamo da zero.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_message = update.message.text

    await update.message.reply_text("🧠 Sto consultando i modelli...")
    try:
        risposta = await orchestratore(chat_id, user_message)
        await aggiungi_a_cronologia(chat_id, "user", user_message)
        await aggiungi_a_cronologia(chat_id, "assistant", risposta)
        await invia_messaggio_lungo(update, risposta)
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {str(e)}")

# ========================
# 7. Avvio
# ========================

def main():
    PORT = int(os.environ.get("PORT", 10000))

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
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
