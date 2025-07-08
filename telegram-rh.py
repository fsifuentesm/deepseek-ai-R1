import pandas as pd
import re
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
import logging

# ---------- CONFIGURACIÓN ----------
load_dotenv()
logging.basicConfig(level=logging.INFO)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
TELEGRAM_TOKEN = os.getenv("token")
archivo_csv = "rh_dataset.csv"  # Cambia esto por el archivo real

# ---------- CARGA DE MODELO ----------
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
    local_files_only=True
)

# ---------- CARGA Y PROCESAMIENTO ----------
def cargar_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def generar_resumen(df):
    resumen = {}
    resumen['total_vacantes'] = len(df)
    resumen['por_status'] = df['status'].value_counts().to_dict()
    resumen['por_ciudad'] = df['city'].value_counts().to_dict()
    resumen['sueldo_promedio_por_ciudad'] = (
        df.groupby('city')['salary_max'].mean().dropna().round(2).to_dict()
    )
    resumen['por_departamento'] = df['custom_fields_departamento'].value_counts().to_dict()
    resumen['por_puesto'] = df['position_name'].value_counts().to_dict()
    return resumen

# ---------- INTERPRETADOR DE PREGUNTAS ----------
def interpretar_pregunta(pregunta):
    pregunta = pregunta.lower()
    if "cuántas" in pregunta or "cuantos" in pregunta:
        if "activas" in pregunta or "vigentes" in pregunta:
            return ("conteo_status", "won")
        if "vacantes" in pregunta and "guadalajara" in pregunta:
            return ("conteo_ciudad", "Guadalajara")
    if "sueldo promedio" in pregunta:
        match = re.search(r"en ([a-záéíóúñ]+)", pregunta)
        if match:
            return ("sueldo_ciudad", match.group(1).capitalize())
    return ("no_entendida", None)

# ---------- GENERADOR DE RESPUESTA ----------
def responder(pregunta, resumen):
    tipo, valor = interpretar_pregunta(pregunta)

    if tipo == "conteo_status":
        total = resumen['por_status'].get(valor, 0)
        contenido = f"Actualmente hay {total} vacantes con estatus '{valor}'."
    elif tipo == "conteo_ciudad":
        total = resumen['por_ciudad'].get(valor, 0)
        contenido = f"Hay {total} vacantes registradas en {valor}."
    elif tipo == "sueldo_ciudad":
        sueldo = resumen['sueldo_promedio_por_ciudad'].get(valor)
        if sueldo:
            contenido = f"El sueldo promedio en {valor} es de ${sueldo:,.2f} MXN."
        else:
            contenido = f"No se encontraron datos de sueldo promedio para {valor}."
    else:
        contenido = "Lo siento, no pude entender tu pregunta. Prueba con otra."

    prompt = f"Eres un asistente de Recursos Humanos. Responde en español con base en el siguiente dato:\n" \
             f"{contenido}\nUsuario: {pregunta}\nAsistente:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    return response

# ---------- TELEGRAM BOT ----------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    await update.message.chat.send_action(action="typing")
    respuesta = responder(user_input, resumen)
    await update.message.reply_text(respuesta)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola 👋 Soy tu asistente de Recursos Humanos. Pregúntame sobre vacantes, ciudades o sueldos."
    )

def main():
    global resumen
    df = cargar_csv(archivo_csv)
    resumen = generar_resumen(df)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot de RH en marcha...")
    app.run_polling()

if __name__ == "__main__":
    main()
