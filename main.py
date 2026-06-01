import cv2
from ultralytics import YOLO
import time 
from google import genai

#Se conecta con la IA en la nube
def enviar_a_gemini():
    print("Conectando con Gemini para el análisis psicológico...")

    client = genai.Client

    #prompt 
    prompt_conductual = ("")

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)

tiempo_inicio = 0.0
gesto_anterior = "Normal"
accion_ejecutada = False

#comprobar si la camara se ha abierto correctamente
if not cap.isOpened():
    print("Error_ No se puede acceder a la webcam")
    exit()

print("Buscando webcam... Preciona la tecla 'q' para salir. ")

#Bucle infinito para leer el video fotograma a fotograma
while cap.isOpened():
    #Lee un fotograma de la camara
    succes, frame = cap.read()

    if not succes:
        print("Error al leer el fotograma de la webcam")
        break
    
    #Da vuelta a le fotograma efecto espejo
    frame = cv2.flip(frame, 1)

    #====================== IA LOCAL (YOLO) ============================
    # Pasar el fotograma actual a nuestro modelo de IA.
    results = model(frame, verbose=False) # Uso 'verbose=False' para que no llene la consola inferior con textos innecesarios.

    #verifica que se encontro al menos una persona en la cámara
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:

        #Extrae la lista de puntos de la primera persona detectada
        puntos = results[0].keypoints.xy[0]

        if len(puntos) > 10:
            hombro_izq_x = int(puntos[5][0])
            hombro_izq_y = int(puntos[5][1])

            hombro_der_x = int(puntos[6][0])
            hombro_der_y = int(puntos[6][1])

            muneca_izq_x = int(puntos[9][0])
            muneca_der_x = int(puntos[10][0])

            oreja_izq_y = int(puntos[3][1])
            oreja_der_y = int(puntos[4][1])
            muneca_izq_y = int(puntos[9][1])
            muneca_der_y = int(puntos[10][1])
            ojo_izq_y = int(puntos[1][1])
            ojo_der_y = int(puntos[2][1])

            distancia_manos = abs(muneca_izq_x - muneca_der_x)
            #mide la distancia vetical para ver si hay tensión en los hombros
            distancia_oreja_hombro_izq = hombro_izq_y - oreja_izq_y
            distancia_oreja_hombro_der = hombro_der_y - oreja_der_y

            postura_actual = "Normal"

            #=============POSTURA================

            # Regla 1: Si subes las manos por encima de los ojos -> Ansiedad
            # Regla 2: Si los hombros se acercan demasiado a las orejas -> Tensión
            # Regla 3: Si las manos se juntan en el pecho -> Brazos Cruzados

            if(muneca_izq_y < ojo_izq_y and muneca_izq_y > 0) or (muneca_der_y < ojo_der_y and muneca_der_y >0):
                postura_actual = "Manos en la Cara (Ansiedad)"
            elif (distancia_oreja_hombro_izq < 45 and distancia_oreja_hombro_izq > 0) or \
                 (distancia_oreja_hombro_der < 45 and distancia_oreja_hombro_der > 0):
                postura_actual = "Hombros Encogidos (Tensión)"
            elif distancia_manos < 60 and muneca_izq_x > 0 and muneca_der_x > 0:
                postura_actual = "Brazos Cruzados"


            # linea de color
            color = (0, 255, 0) if postura_actual == "Normal" else (0, 0, 255)
            cv2.putText(frame, f"Postura: {postura_actual}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # ==========DIBUJAR PUNTOS DE ANÁLISIS EN PANTALLA
            # Dibujamos círculos en los Ojos (Color Azul)
            cv2.circle(frame, (int(puntos[1][0]), ojo_izq_y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (int(puntos[2][0]), ojo_der_y), 5, (255, 0, 0), -1)
            
            # Dibujamos círculos en las Orejas (Color Rosa)
            cv2.circle(frame, (int(puntos[3][0]), oreja_izq_y), 5, (255, 0, 255), -1)
            cv2.circle(frame, (int(puntos[4][0]), oreja_der_y), 5, (255, 0, 255), -1)
            
            # Dibujamos círculos y textos en los Hombros (Color Amarillo)
            cv2.circle(frame, (hombro_izq_x, hombro_izq_y), 6, (0, 255, 255), -1)
            cv2.putText(frame, f"H_Izq (Y:{hombro_izq_y})", (hombro_izq_x + 10, hombro_izq_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.circle(frame, (hombro_der_x, hombro_der_y), 6, (0, 255, 255), -1)
            cv2.putText(frame, f"H_Der (Y:{hombro_der_y})", (hombro_der_x - 110, hombro_der_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Dibujamos círculos y textos en las Muñecas (Color Naranja)
            cv2.circle(frame, (muneca_izq_x, muneca_izq_y), 6, (0, 165, 255), -1)
            cv2.putText(frame, f"M_Izq (Y:{muneca_izq_y})", (muneca_izq_x + 10, muneca_izq_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            cv2.circle(frame, (muneca_der_x, muneca_der_y), 6, (0, 165, 255), -1)
            cv2.putText(frame, f"M_Der (Y:{muneca_der_y})", (muneca_der_x - 110, muneca_der_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            #========================== 
            
            #si mantiene la misma postura que en el fotograma anterior
            if postura_actual == gesto_anterior:
                tiempo_transcurrido = time.time() - tiempo_inicio

                # Se activa con cualquier postura de riesgo que dure 2 segundos
                if (tiempo_transcurrido >= 2.0) and (accion_ejecutada == False) and (postura_actual != "Normal"):
                   
                    cv2.imwrite("captura_corregida.jpg", frame)
                    print("Imagen 'captura_corregida.jpg' guardada con éxito.")
            
            #si cambio la postura reinicio el reloj al tiempo actual
            elif postura_actual != gesto_anterior:
                tiempo_inicio = time.time()

                gesto_anterior = postura_actual

                accion_ejecutada = False
    #Mostrar el video en una ventana llamada mi lector de postura
    cv2.imshow('Lector de Postura', frame)

    #ROMPRE el buble si al pulsar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
print("Programa cerrado correctamente")