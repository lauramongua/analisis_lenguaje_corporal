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

        if len(puntos) > 6:
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
            distancia_oreja_hombro_izq = hombro_izq_y - oreja_izq_y
            distancia_oreja_hombro_der = hombro_der_y - oreja_der_y

            postura_actual = "Normal"

            #pinta la postura
            if distancia_manos < 60 and muneca_izq_x > 0 and muneca_der_x > 0:
                postura_actual = "Brazos Cruzados"

            color = (0, 0, 255) if postura_actual == "Brazos Cruzados" else (0, 255, 0)
            cv2.putText(frame, f"Postura: {postura_actual}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            #si mantiene la misma postura que en el fotograma anterior
            if postura_actual == gesto_anterior:
                tiempo_transcurrido = time.time() - tiempo_inicio

                #si lleva mas de 2seg con los brazos cruzadps 
                if(tiempo_transcurrido >= 2.0) and (accion_ejecutada == False) and (postura_actual == "Brazos Cruzados"):
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