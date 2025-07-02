from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

# Configura el log para ver errores si algo falla
logging.basicConfig(level=logging.INFO)

# Modelo
MODEL_ID = "deepseek-ai/DeepSeek-R1-type-model-params"
print("Cargando modelo, esto puede tardar un poco la primera vez...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=quantization,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True,
)

print("Modelo cargado. ¡Listo para Telegram!")

# Historial por usuario
user_histories = {}

# Función que genera la respuesta del modelo
def generar_respuesta(user_id, user_input):
    if user_id not in user_histories:
        user_histories[user_id] = []

    USE_HISTORY = True  # Cambia a False para modo stateless

    chat_history = user_histories[user_id] if USE_HISTORY else []

    initial_prompt = (
        "Eres un asistente útil que responde en español con precisión y brevedad. "
        "No expliques tus pensamientos internos ni digas que estás pensando."
    )

prompt = "\n".join(
        [initial_prompt] +
        chat_history + [f"Usuario: {user_input}", "Asistente:"])

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]

    print(user_histories)

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Extraer solo la última respuesta
    if "Asistente:" in response:
        response = response.split("Asistente:")[-1].strip()

    for stop_token in ["Usuario:", "Asistente:", "</think>", "<|im_end|>", "</s>", "Let me think", "Thought:", "Wait:", "Thinking:"]:
        if stop_token in response:
            response = response.split(stop_token)[0].strip()

    response = response.strip().replace("  ", " ")

    print(tokenizer.eos_token_id)

    # Actualiza historial
    chat_history.append(f"Usuario: {user_input}")
    chat_history.append(f"Asistente: {response}")
    user_histories[user_id] = chat_history[-6:]  # Limita historial

    return response

# Función principal que maneja mensajes
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    user_id = update.message.chat_id

    if user_input.lower() == "reset":
        user_histories[user_id] = []
        await update.message.reply_text("Historial reiniciado")
        return

    await update.message.chat.send_action(action="typing")
    respuesta = generar_respuesta(user_id, user_input)
    await update.message.reply_text(respuesta)

# Inicio del bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("¡Hola! Soy tu asistente IA 🤖. Pregúntame lo que quieras.")

# Token de Telegram
TELEGRAM_TOKEN = "id-telegram-example"

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot en marcha... Ctrl+C para detener.")
    app.run_polling()

if __name__ == "__main__":
    main()
