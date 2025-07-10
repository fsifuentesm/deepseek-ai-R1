import pandas as pd
import re
import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, CallbackQueryHandler, filters
import logging
import smtplib
from email.message import EmailMessage
import requests

# ---------- CONFIGURACIÓN ----------
load_dotenv()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("token")

# Historial por usuario
user_histories = {}
user_states = {}

url_client = 'https://apidarwin.tracsa.com.mx/api/v1/dbsclient/'
url_repairs = 'https://apidarwin.tracsa.com.mx/api/v1/repairs'
url_parts = 'https://apidarwin.tracsa.com.mx/api/v1/partslist/'
token = os.getenv("token_darwin")

headers = {
    "Authorization": f"Bearer {token}",
}

# ---------- UTILIDADES ----------
def limpiar_respuesta(respuesta):
    for stop_token in ["Usuario:", "Asistente:", "</think>", "<|im_end|>", "</s>", "Let me think", "Thought:", "Wait:", "Thinking:"]:
        if stop_token in respuesta:
            respuesta = respuesta.split(stop_token)[0].strip()
    return respuesta

def enviar_correo(destinatarios, asunto, cuerpo):
    try:
        smtp_user = os.getenv("OFFICE365_USER")
        smtp_pass = os.getenv("OFFICE365_PASS")

        if not smtp_user or not smtp_pass:
            raise ValueError("Credenciales de Office 365 no configuradas.")

        # Si es una sola dirección, conviértela en lista
        if isinstance(destinatarios, str):
            destinatarios = [destinatarios]

        msg = EmailMessage()
        msg["From"] = smtp_user
        msg["To"] = ", ".join(destinatarios)  # Esto es solo para que el header se vea bien
        msg["Subject"] = asunto
        msg.set_content(cuerpo)

        smtp_server = "smtp.office365.com"
        smtp_port = 587

        with smtplib.SMTP(smtp_server, smtp_port) as smtp:
            smtp.starttls()
            smtp.login(smtp_user, smtp_pass)
            smtp.send_message(msg, to_addrs=destinatarios)  # Esto asegura que los reciba quien debe

        print("✅ Correo enviado con éxito.")
        return True

    except Exception as e:
        print(f"❌ Error al enviar correo: {e}")
        return False


# ---------- REPARACIONES ----------
async def mostrar_reparaciones(update, context, user_id, modelo):
    try:
        response = requests.get(
            f"{url_repairs}/?model={modelo}",
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            resultados = response.json().get("results", [])
            tipos = sorted(set(r["repair_type"] for r in resultados if "repair_type" in r))

            if tipos:
                keyboard = [[InlineKeyboardButton(t, callback_data=f"repair_{t}")] for t in tipos]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await context.bot.send_message(
                    chat_id=user_id,
                    text=f"🔧 Reparaciones disponibles para el modelo {modelo}:\nSelecciona una opción:",
                    reply_markup=reply_markup
                )
                user_states[user_id]['step'] = 3
            else:
                await context.bot.send_message(chat_id=user_id, text=f"❌ No se encontraron reparaciones para {modelo}.")
                user_states[user_id]['step'] = 2
        else:
            await context.bot.send_message(chat_id=user_id, text="⚠️ No fue posible consultar las reparaciones.")
    except Exception as e:
        print(f"[ERROR MODELO desde botón] {e}")
        await context.bot.send_message(chat_id=user_id, text="⚠️ Error al consultar reparaciones.")

# ---------- TELEGRAM BOT ----------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text.strip()

    if text.lower() == "reset":
        user_states.pop(user_id, None)
        user_histories.pop(user_id, None)
        await update.message.reply_text("✅ Conversación reiniciada. Escribe /start para comenzar de nuevo.")
        return

    if user_id not in user_states:
        user_states[user_id] = {'step': None, 'data': {}}

    if user_id not in user_histories:
        user_histories[user_id] = []

    if user_states[user_id]['step'] is None:
        historial = user_histories[user_id]
        historial.append({"user": text, "bot": "Estoy listo para ayudarte. Escribe /start para comenzar."})
        if len(historial) > 10:
            historial.pop(0)
        await update.message.reply_text(historial[-1]['bot'])
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
            if response.status_code == 200 and response.json():
                data_json = response.json()
                modelos = sorted(set(item["MODELO"].strip() for item in data_json.get("data", []) if "MODELO" in item))

                if modelos:
                    keyboard = [[InlineKeyboardButton(m, callback_data=f"modelo_{m}")] for m in modelos]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await update.message.reply_text("✅ Cliente válido. Selecciona un modelo:", reply_markup=reply_markup)
                    user_states[user_id]['step'] = 2
                else:
                    await update.message.reply_text("✅ Cliente válido, pero no se encontraron modelos registrados.")
            else:
                await update.message.reply_text("❌ El código de cliente no es válido. Intenta de nuevo.")
        except Exception as e:
            await update.message.reply_text("⚠️ Error al validar el cliente. Intenta más tarde.")
            print(f"[ERROR CLIENTE] {e}")

    elif step == 2:
        data['modelo'] = text.upper()
        await mostrar_reparaciones(update, context, user_id, data['modelo'])

    elif step == 3:
        data['reparacion'] = text
        await update.message.reply_text("¡Gracias! Procesando solicitud...")

        historial = user_histories[user_id]
        historial.append({"user": str(data), "bot": "Procesando solicitud..."})
        if len(historial) > 10:
            historial.pop(0)

        await finalizar_cotizacion(user_id, context, data)



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola 👋 Soy tu asistente de cotizaciones. Vamos a recolectar algunos datos para generar una cotización."
    )
    user_id = update.message.from_user.id
    user_states[user_id] = {'step': 0, 'data': {}}
    await handle_message(update, context)


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = user_states[user_id]['data']

    if query.data.startswith("modelo_"):
        modelo = query.data.replace("modelo_", "")
        user_states[user_id]['data']['modelo'] = modelo
        await query.edit_message_text(f"✅ Modelo seleccionado: {modelo}")
        await mostrar_reparaciones(update, context, user_id, modelo)

    elif query.data.startswith("repair_"):
        tipo_reparacion = query.data.replace("repair_", "")
        data["reparacion"] = tipo_reparacion

        await query.edit_message_text(
            f"✅ Tipo de reparación seleccionada: {tipo_reparacion}\n¡Gracias! Procesando solicitud..."
        )

        await finalizar_cotizacion(user_id, context, data)



async def finalizar_cotizacion(user_id, context, data):
    # Aquí puedes construir el JSON que quieras enviar
    json_payload = {
        "client": data.get("cliente"),
        "model": data.get("modelo"),
        "parts": {},
        "repair_type": data.get("reparacion"),
        "series": "N/A",
    }

    # Enviar a otro endpoint
    try:
        response = requests.post(f"{url_parts}", json=json_payload, timeout=5, headers=headers)
        if response.status_code in [200, 201]:
            print("✅ Cotización enviada correctamente al endpoint externo.")
        else:
            print(f"❌ Error al enviar cotización: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[ERROR ENVÍO EXTERNO] {e}")

    # Generar resumen
    respuesta = f"""✅ Cotización generada:

📌 Cliente: {data.get("cliente")}
🛠 Modelo: {data.get("modelo")}
🔧 Reparación: {data.get("reparacion")}
    """

    # Enviar al usuario
    await context.bot.send_message(chat_id=user_id, text=respuesta)

    destinatarios = os.getenv("destinatarios", "")
    lista_destinatarios = [d.strip() for d in destinatarios.split(",") if d.strip()]

    # Enviar correo
    enviar_correo(
        lista_destinatarios,
        "Nueva cotización generada",
        respuesta
    )

    # Actualizar historial
    historial = user_histories[user_id]
    historial.append({"user": str(data), "bot": respuesta})
    if len(historial) > 10:
        historial.pop(0)

    user_states[user_id]['step'] = None  # Fin del flujo



def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CallbackQueryHandler(handle_button))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot de TRACSA en marcha...")
    app.run_polling()

if __name__ == "__main__":
    main()
