# chequear_camaras.py
import cv2

MAX_CAMERAS = 5  # Asumimos que no tienes más de 5 cámaras conectadas

print("--- Herramienta de Verificación de Índices de Cámara ---")

for i in range(MAX_CAMERAS):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Índice {i}: CÁMARA DETECTADA. Presiona ESPACIO para capturar o ESC para siguiente.")
        
        # Muestra la ventana de la cámara
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.putText(frame, f"INDICE: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f"Camara Indice {i}", frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC - Salir de esta cámara e ir a la siguiente
                break
            elif key == 32: # SPACE - Captura y confirma
                 print(f"Confirmado: Esta es la cámara con índice {i}.")
                 break
        
        cap.release()
        cv2.destroyWindow(f"Camara Indice {i}")
    else:
        print(f"❌ Índice {i}: No se pudo abrir la cámara.")
        
print("--- Fin de la verificación ---")