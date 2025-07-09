import pandas as pd
import re
import os
from dotenv import load_dotenv
#from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#import torch
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
import logging
import smtplib
from email.message import EmailMessage
import requests
 
# ---------- CONFIGURACIÓN ----------
load_dotenv()
logging.basicConfig(level=logging.INFO)
 
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
TELEGRAM_TOKEN = os.getenv("token")
archivo_csv = "rh_dataset.csv"
 
# ---------- CARGA DE MODELO ----------
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# quantization = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True
# )
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     device_map="auto",
#     quantization_config=quantization,
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
#     local_files_only=True
# )
 
# Historial por usuario
user_histories = {}
user_states = {}
 
client = 'TMXN00'
url_client = 'https://apidarwin.tracsa.com.mx/api/v1/dbsclient/'
url_repairs = 'https://apidarwin.tracsa.com.mx/api/v1/repairs'
url_parts = 'https://apidarwintest.tracsa.com.mx/api/v1/partslist'
token = os.getenv("token_darwin")
headers = {
    "Authorization": f"Bearer {token}",
}
 
# ---------- UTILIDADES ----------
def cargar_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df
 
def generar_resumen(df):
    response = requests.get(url_repairs, headers=headers)
    resumen = response.json()
    return resumen
 
def construir_prompt(historial, resumen):
    contexto = "".join([f"Usuario: {turno['user']}\nAsistente: {turno['bot']}\n" for turno in historial])
    return f"Eres un cotizador experto. Con base en el siguiente resumen de tipos de reparación:{resumen} Responde en español lo que el usuario pregunte. {contexto}"
 
def obtener_respuesta_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return limpiar_respuesta(response)
 
def limpiar_respuesta(respuesta):
    for stop_token in ["Usuario:", "Asistente:", "</think>", "<|im_end|>", "</s>", "Let me think", "Thought:", "Wait:", "Thinking:"]:
        if stop_token in respuesta:
            respuesta = respuesta.split(stop_token)[0].strip()
    return respuesta
 
def enviar_correo_office365(destinatario, asunto, cuerpo):
    user = os.getenv("OFFICE365_USER")
    password = os.getenv("OFFICE365_PASS")
    if not user or not password:
        print("Faltan credenciales de Office 365 en variables de entorno.")
        return False
    msg = EmailMessage()
    msg["Subject"] = asunto
    msg["From"] = user
    msg["To"] = destinatario
    msg.set_content(str(cuerpo))
    try:
        with smtplib.SMTP("smtp.office365.com", 587) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
        print("Correo enviado correctamente.")
        return True
    except Exception as e:
        print(f"Error al enviar correo: {e}")
        return False
 
# ---------- TELEGRAM BOT ----------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text.strip()
 
    if text.lower() == "reset":
        user_states.pop(user_id, None)
        user_histories.pop(user_id, None)
        await update.message.reply_text("✅ Conversación reiniciada. Escribe /start para comenzar de nuevo.")
        print(f"[RESET] Usuario {user_id} reinició la conversación.")
        return
 
    if user_id not in user_states:
        user_states[user_id] = {'step': None, 'data': {}}
 
    if user_id not in user_histories:
        user_histories[user_id] = []
 
    resumen_texto = context.application.bot_data.get("resumen", "")
 
    # Si ya se completó el flujo, permitir preguntas libres sobre el resumen
    if user_states[user_id]['step'] is None:
        historial = user_histories[user_id]
        historial.append({"user": text, "bot": "..."})
        if len(historial) > 10:
            historial.pop(0)
 
        prompt = construir_prompt(historial, resumen_texto)
        #respuesta = obtener_respuesta_llm(prompt)
        respuesta = ''
 
        historial[-1]['bot'] = respuesta
 
        await update.message.reply_text(respuesta)
        print(f"[DIALOGO LIBRE] Usuario: {user_id}\nPregunta: {text}\nRespuesta: {respuesta}\n")
        return
 
    step = user_states[user_id]['step']
    data = user_states[user_id]['data']
 
    if step == 0:
        await update.message.reply_text("¿Cuál es el código de cliente?")
        user_states[user_id]['step'] += 1

    elif step == 1:
        data['cliente'] = text
        try:
            response = requests.get(
                f'{url_client}{text}',
                headers=headers,
                timeout=5
            )
            print(response.json())
            if response.status_code == 200 and response.json():
                data_json = response.json()
                modelos = set(item["MODELO"].strip() for item in data_json.get("data", []) if "MODELO" in item)
                if modelos:
                    modelos_lista = "\n".join(f"- {m}" for m in sorted(modelos))
                    print(modelos_lista)
                    await update.message.reply_text(
                        f"✅ Cliente válido.\nModelos disponibles:\n{modelos_lista}\n\n¿Cuál modelo necesitas?"
                    )
                else:
                    await update.message.reply_text("✅ Cliente válido, pero no se encontraron modelos registrados.")
                user_states[user_id]['step'] += 1

            else:
                await update.message.reply_text("❌ El código de cliente no es válido. Intenta de nuevo.")
                print(f"[CLIENTE NO VÁLIDO] Usuario: {user_id} ingresó: {text}")

        except Exception as e:
            await update.message.reply_text("⚠️ Error al validar el cliente. Intenta más tarde.")
            print(f"[ERROR CLIENTE] {e}")

    elif step == 2:
        data['modelo'] = text.upper()
        try:
            response = requests.get(
                f"{url_repairs}/?model={data['modelo']}",
                headers=headers,
                timeout=5
            )
            if response.status_code == 200:
                resultados = response.json().get("results", [])
                tipos = set(r["repair_type"] for r in resultados if "repair_type" in r)
                if tipos:
                    lista_tipos = "\n".join(f"- {t}" for t in sorted(tipos))
                    await update.message.reply_text(
                        f"🔧 Reparaciones disponibles para el modelo {data['modelo']}:\n{lista_tipos}\n\n¿Qué tipo de reparación necesitas?"
                    )

                else:
                    await update.message.reply_text(
                        f"❌ No se encontraron reparaciones para el modelo {data['modelo']}.\nPor favor, intenta con otro modelo."
                    )
                    user_states[user_id]['step'] = 2
                    return
            else:
                await update.message.reply_text("No fue posible obtener las reparaciones disponibles.")

            user_states[user_id]['step'] += 1
    
        except Exception as e:
            await update.message.reply_text("⚠️ Error al consultar reparaciones del modelo. Intenta más tarde.")
            print(f"[ERROR MODELO] {e}")
 
    elif step == 3:
        data['reparacion'] = text
        await update.message.reply_text("¡Gracias! Procesando solicitud...")
 
        historial = user_histories[user_id]
        historial.append({"user": str(data), "bot": "Procesando solicitud..."})
        if len(historial) > 10:
            historial.pop(0)
 
        prompt = construir_prompt(historial, resumen_texto)
        #respuesta = obtener_respuesta_llm(prompt)
        respuesta = 'aaaa'
 
        await update.message.reply_text(respuesta)
 
        historial[-1]["bot"] = respuesta
 
        enviar_correo_office365(
            destinatario=os.getenv("destinatarios"),
            asunto="Nueva cotización generada",
            cuerpo=respuesta
        )
 
        print(f"[COTIZACIÓN] Usuario: {user_id}\nDatos: {data}\nRespuesta: {respuesta}\n")
 
        user_states[user_id]['step'] = None  # Permite preguntas libres a partir de aquí
 
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola 👋 Soy tu asistente de cotizaciones. Vamos a recolectar algunos datos para generar una cotización."
    )
    user_id = update.message.from_user.id
    user_states[user_id] = {'step': 0, 'data': {}}
    await handle_message(update, context)
 
def main():
    df = cargar_csv(archivo_csv)
    resumen = generar_resumen(df)
 
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.bot_data["resumen"] = resumen
 
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
 
    print("Bot de TRACSA en marcha...")
    app.run_polling()
 
if __name__ == "__main__":
    main()
