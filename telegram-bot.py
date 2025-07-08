import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
import logging
import psycopg2
from dotenv import load_dotenv
 
# cargar variables de entorno
load_dotenv()
 
# logging
logging.basicConfig(level=logging.INFO)
 
# modelo
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
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
 
# historial
user_histories = {}
 
# embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")
 
# conectar a Postgres
db_host = os.getenv("db_host")
db_port = os.getenv("db_port")
db_name = os.getenv("db_name")
db_user = os.getenv("db_user")
db_pass = os.getenv("db_pass")
 
# indexar la tabla inicialmente
conn = psycopg2.connect(
    host=db_host, port=db_port,
    dbname=db_name, user=db_user,
    password=db_pass
)
cur = conn.cursor()
cur.execute("""SELECT id, title, ss."branchOffice" FROM sitsa_sitsafieldreport ss""")
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
dataset = [dict(zip(colnames, row)) for row in rows]
cur.close()
conn.close()
 
texts = [json.dumps(record, ensure_ascii=False) for record in dataset]
 
# generar embeddings si no existen
if not os.path.exists("embeddings.pkl"):
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    with open("embeddings.pkl", "wb") as f:
        pickle.dump((texts, embeddings), f)
else:
    with open("embeddings.pkl", "rb") as f:
        texts, embeddings = pickle.load(f)
embeddings = np.array(embeddings)
 
def buscar_contexto(user_input):
    user_emb = embedder.encode(user_input)
    sims = cosine_similarity([user_emb], embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:50]
    contextos = [texts[i] for i in top_idx]
    return "\n".join(contextos)
 
def generar_sql(user_input, contexto):
    prompt = (
        "Genera únicamente la instrucción SQL en PostgreSQL para responder la pregunta, "
        "basada en la tabla sitsa_sitsafieldreport. "
        "Los nombres de columna van exactamente así: id, title, \"branchOffice\" (con comillas dobles), "
        "y LIMIT 50 si aplica. "
        "NO incluyas texto de razonamiento, ni etiquetas de pensamiento, ni comentarios, SOLO la instrucción SQL limpia."
        f"\nContexto:\n{contexto}\n"
        f"Pregunta del usuario: {user_input}\n"
        "Instrucción SQL:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    sql_instruction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # limpia tokens de razonamiento
    for stop_token in ["</think>", "Usuario:", "Asistente:", "<|im_end|>", "</s>", "Thought:", "Thinking:"]:
        if stop_token in sql_instruction:
            sql_instruction = sql_instruction.split(stop_token)[0].strip()
    
    print(f"SQL limpio generado por IA: {sql_instruction}")
    return sql_instruction

 
def ejecutar_sql(sql_instruction):
    try:
        conn = psycopg2.connect(
            host=db_host, port=db_port,
            dbname=db_name, user=db_user,
            password=db_pass
        )
        cur = conn.cursor()
        cur.execute(sql_instruction)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        results = [dict(zip(colnames, row)) for row in rows]
        cur.close()
        conn.close()
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        print(f"Error ejecutando SQL: {e}")
        return "Error ejecutando la consulta generada."
 
def generar_respuesta(user_id, user_input, resultados_sql):
    if user_id not in user_histories:
        user_histories[user_id] = []
    chat_history = user_histories[user_id]
    initial_prompt = (
        "Eres un asistente útil y preciso en español. "
        "Responde únicamente usando los datos proporcionados, "
        "y si no los conoces, di 'No tengo datos suficientes'."
    )
    prompt = "\n".join(
        [initial_prompt] + chat_history + [f"Datos de la consulta:\n{resultados_sql}", f"Usuario: {user_input}", "Asistente:"]
    )
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
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Extraer solo la última respuesta
    if "Asistente:" in response:
        response = response.split("Asistente:")[-1].strip()

    for stop_token in ["Usuario:", "Asistente:", "</think>", "<|im_end|>", "</s>", "Let me think", "Thought:", "Wait:", "Thinking:"]:
        if stop_token in response:
            response = response.split(stop_token)[0].strip()

    chat_history.append(f"Usuario: {user_input}")
    chat_history.append(f"Asistente: {response}")
    user_histories[user_id] = chat_history[-6:]
    return response
 
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    user_id = update.message.chat_id
 
    if user_input.lower() == "reset":
        user_histories[user_id] = []
        await update.message.reply_text("Historial reiniciado")
        return
 
    await update.message.chat.send_action(action="typing")
    contexto = buscar_contexto(user_input)
    sql_sugerido = generar_sql(user_input, contexto)
 
    # ejecuta el SQL y usa resultado como contexto
    resultados_sql = ejecutar_sql(sql_sugerido)
 
    # genera respuesta final
    respuesta = generar_respuesta(user_id, user_input, resultados_sql)
    await update.message.reply_text(respuesta)
 
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "¡Hola! Soy tu asistente IA con generación de SQL automática. Pregúntame algo sobre los reportes, "
        "y generaré la consulta además de explicarte los resultados."
    )
 
TELEGRAM_TOKEN = os.getenv("token")
 
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot en marcha... Ctrl+C para detener.")
    app.run_polling()
 
if __name__ == "__main__":
    main()
