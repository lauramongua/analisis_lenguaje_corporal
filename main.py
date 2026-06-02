import cv2
from ultralytics import YOLO
import time 
import telebot
import ollama
import threading
import os

# ==================== CONFIGURACIONES GLOBALES ====================
TELEGRAM_TOKEN = "8650188996:AAFuiH2q3bDEgJO8vwV7oG4LNNQj_YKNQVw"
CHAT_ID = "7941168412"
CONTEXTO_ENTREVISTA = "entrevista de trabajo"

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Variables de control de tiempo y estado
tiempo_inicio = time.time()
gesto_anterior = "Normal"
accion_ejecutada = False  # Indica si la IA está procesando actualmente

def enviar_mesaje_telegram(postura, texto_informe):
    try:
        print("Enviando reporte y captura a Telegram...")
        texto_telegram = f"⚠️ *ALERTA DE LENGUAJE NO VERBAL*\n\nPostura detectada: *{postura}*\n\n{texto_informe}"

        with open("captura_corregida.jpg", "rb") as foto:
            bot.send_photo(CHAT_ID, foto, caption=texto_telegram[:1024], parse_mode="Markdown")
            
        if len(texto_telegram) > 1024:
            bot.send_message(CHAT_ID, texto_telegram, parse_mode="Markdown")
            
        print("✅ ¡Todo enviado a Telegram correctamente!")
    except Exception as error_tel:
        print(f"❌ Error al enviar el mensaje de Telegram: {error_tel}")

# ==================== FUNCIÓN DE IA ====================
def enviar_a_ollama_local(postura_detectada):
    global accion_ejecutada, tiempo_inicio
    
    print(f"🧠 [HILO] Iniciando análisis con Qwen2.5-VL para: {postura_detectada}...")

    prompt_conductual = (
        f"Actúa como un experto en psicología organizacional y comunicación no verbal.\n"
        f"Analiza la situación de un candidato en una {CONTEXTO_ENTREVISTA}.\n"
        f"Nuestro sistema de visión artificial local ha detectado geométricamente la postura: '{postura_detectada}'.\n"
        f"Evalúa la imagen adjunta. Analiza sutilmente la tensión y ofrécele 3 consejos prácticos de comunicación no verbal."
    )

    try:

        # Petición local al modelo multimodal Qwen
        response = ollama.chat(
            model='qwen2.5vl:3b', 
            messages=[{
                'role': 'user',
                'content': prompt_conductual,
                'images': ['captura_corregida.jpg']
            }]
        )
        
        texto_ia = response['message']['content']
        
        print("\n================ 🦙 INFORME CONDUCTUAL DE QWEN LOCAL ================")
        print(texto_ia)
        print("======================================================================\n")
        
        # --- ENVIAR ALERTA POR TELEGRAM ---
        print("🚀 [HILO] Enviando reporte y captura a Telegram...")
        texto_telegram = f"⚠️ *ALERTA DE LENGUAJE NO VERBAL*\n\nPostura detectada: *{postura_detectada}*\n\n{texto_ia}"
        
        with open("captura_corregida.jpg", "rb") as foto:
            bot.send_photo(CHAT_ID, foto, caption=texto_telegram[:1024], parse_mode="Markdown")
            
        if len(texto_telegram) > 1024:
            bot.send_message(CHAT_ID, texto_telegram, parse_mode="Markdown")
            
        print("✅ [HILO] ¡Todo enviado a Telegram correctamente!")

    except Exception as e:
        # IMPORTANTE: Si Telegram o Ollama fallan, este print te dirá el motivo exacto en la consola
        print(f"❌ [HILO] Error crítico en el procesamiento: {e}")
    
    # Pausa de seguridad antes de permitir que el sistema vuelva a enviar otra foto
    print("⏱️ [HILO] Iniciando pausa de seguridad de 10 segundos...")
    time.sleep(10)
    
    # Al terminar la pausa, reiniciamos el reloj y liberamos el candado
    tiempo_inicio = time.time()
    accion_ejecutada = False
    print("🔓[HILO] Sistema liberado. Listo para detectar nuevas posturas.")


# ==================== INICIALIZACIÓN DE MODELOS ====================
model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede acceder a la webcam")
    exit()

print("Buscando webcam... Presiona la tecla 'q' para salir. ")

# ==================== BUCLE PRINCIPAL DE LA CÁMARA ====================
while cap.isOpened():
    succes, frame = cap.read()
    if not succes:
        print("Error al leer el fotograma de la webcam")
        break
    
    frame = cv2.flip(frame, 1)
    results = model(frame, verbose=False)

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        puntos = results[0].keypoints.xy[0]

        if len(puntos) == 17:
            # Extracción de puntos clave
            ojo_izq_y     = int(puntos[1][1])
            ojo_der_y     = int(puntos[2][1])
            oreja_izq_x   = int(puntos[3][0])
            oreja_izq_y   = int(puntos[3][1])
            oreja_der_x   = int(puntos[4][0])  
            oreja_der_y   = int(puntos[4][1])
            hombro_izq_x  = int(puntos[5][0])
            hombro_izq_y  = int(puntos[5][1])
            hombro_der_x  = int(puntos[6][0])
            hombro_der_y  = int(puntos[6][1])
            muneca_izq_x  = int(puntos[9][0])
            muneca_izq_y  = int(puntos[9][1])
            muneca_der_x  = int(puntos[10][0])
            muneca_der_y  = int(puntos[10][1])

            # Métricas
            distancia_manos = abs(muneca_izq_x - muneca_der_x)
            ancho_hombros   = abs(hombro_izq_x - hombro_der_x)
            distancia_entre_orejas = abs(oreja_der_x - oreja_izq_x)
            escala_referencia = distancia_entre_orejas if distancia_entre_orejas > 0 else (ancho_hombros * 0.3)

            distancia_oreja_hombro_izq = hombro_izq_y - oreja_izq_y
            distancia_oreja_hombro_der = hombro_der_y - oreja_der_y

            # Árbol de decisión de posturas
            postura_actual = "Normal"

            if (ojo_izq_y < muneca_izq_y < hombro_izq_y) or (ojo_der_y < muneca_der_y < hombro_der_y):
                postura_actual = "Contacto de manos en la cara o cuello (Ansiedad sutil)"
            elif (distancia_oreja_hombro_izq < escala_referencia * 0.7) or (distancia_oreja_hombro_der < escala_referencia * 0.7):
                postura_actual = "Hombros levemente elevados y rígidos (Tensión)"
            elif distancia_manos < (escala_referencia * 0.8):
                if muneca_izq_y < (hombro_izq_y + 40):
                    postura_actual = "Brazos cruzados en el pecho (Defensivo)"
                else:
                    postura_actual = "Manos entrelazadas o manipulación de dedos (Ansiedad)"
            elif distancia_manos > (ancho_hombros * 1.2):
                postura_actual = "Postura Abierta / Gestos Expresivos (Confianza)"

            # Dibujar en pantalla
            color = (0, 255, 0) if postura_actual == "Normal" else (0, 0, 255)
            cv2.putText(frame, f"Postura: {postura_actual}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            # Círculos visuales
            cv2.circle(frame, (int(puntos[1][0]), ojo_izq_y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (int(puntos[2][0]), ojo_der_y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (int(puntos[3][0]), oreja_izq_y), 5, (255, 0, 255), -1)
            cv2.circle(frame, (int(puntos[4][0]), oreja_der_y), 5, (255, 0, 255), -1)
            cv2.circle(frame, (hombro_izq_x, hombro_izq_y), 6, (0, 255, 255), -1)
            cv2.circle(frame, (hombro_der_x, hombro_der_y), 6, (0, 255, 255), -1)

            # --- CONTROL DE DISPARO CON HILOS ---
            if postura_actual == gesto_anterior:
                # Solo contamos el tiempo si el hilo secundario no está trabajando
                if not accion_ejecutada:
                    tiempo_transcurrido = time.time() - tiempo_inicio

                    if (tiempo_transcurrido >= 2.0) and (postura_actual != "Normal"):
                        print(f"📸 ¡Postura '{postura_actual}' mantenida por 2s! Guardando captura...")
                        cv2.imwrite("captura_corregida.jpg", frame)
                        
                        # Activamos el candado inmediatamente para que el 'while' no cree 1000 hilos seguidos
                        accion_ejecutada = True 
                        
                        # Lanzamos el proceso en segundo plano
                        hilo_ia = threading.Thread(target=enviar_a_ollama_local, args=(postura_actual,))
                        hilo_ia.start()
            
            elif postura_actual != gesto_anterior:
                # Si cambias de postura y el sistema no está procesando, reiniciamos el cronómetro
                if not accion_ejecutada:
                    tiempo_inicio = time.time()
                    gesto_anterior = postura_actual

    cv2.imshow('Lector de Postura', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
print("Programa cerrado correctamente")