# agente_seguridad/emotion_detector.py

import cv2
import numpy as np
# from tensorflow.keras.models import load_model # Asumimos que usas Keras o similar

class EmotionDetector:
    def __init__(self, modelo_path, haarcascade_path):
        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)
        if self.face_cascade.empty():
            print(f"❌ ADVERTENCIA: Haar Cascade no cargado. Ruta: {haarcascade_path}")
        self.modelo_emocion = None 
        self.etiquetas_emocion = ['Enojado', 'Disgusto', 'Miedo', 'Feliz', 'Triste', 'Sorpresa', 'Neutro']
        
    def detectar_y_analizar(self, frame_rostro, bbox_rostro):
        """
        Detecta rostro, infiere emoción y dibuja la caja.
        GARANTIZA RETORNAR UNA TUPLA DE 2 ELEMENTOS (emocion, frame_procesado).
        """
        # 🚨 Inicializar valores de retorno garantizados 🚨
        emocion_detectada = 'NEUTRO'
        frame_con_procesamiento = frame_rostro.copy() if frame_rostro is not None and frame_rostro.size > 0 else None 
        
        if frame_con_procesamiento is None:
            # 🚨 Si el frame está vacío, retornamos los valores por defecto 🚨
            return emocion_detectada, frame_con_procesamiento

        try:
            # 1. Detección de Rostro
            rostros = []
            if bbox_rostro is not None and bbox_rostro[2] > 0: # Usamos la BBox del Broker si existe
                rostros = [bbox_rostro]
            elif not self.face_cascade.empty(): # Usamos Haar Cascade si el Broker no dio BBox y el detector cargó
                # Conversión a escala de grises para el detector
                gray_frame = cv2.cvtColor(frame_rostro, cv2.COLOR_BGR2GRAY) 
                # Ejecutar detectMultiScale SOLO si self.face_cascade NO está vacío
                rostros = self.face_cascade.detectMultiScale(
                    gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

            # 2. Análisis de Emoción
            if len(rostros) > 0:
                # Solo procesamos el primer rostro detectado (asumiendo que es el rostro principal)
                x, y, w, h = rostros[0] 
                
                y_start = max(0, y)
                y_end = min(frame_rostro.shape[0], y + h)
                x_start = max(0, x)
                x_end = min(frame_rostro.shape[1], x + w)
                face_crop = frame_rostro[y_start:y_end, x_start:x_end]

                if face_crop.size > 0:
                    # 🚨 Aquí iría tu código de predicción con self.modelo_emocion.predict(face_crop) 🚨
                    emocion_detectada = 'NEUTRO' # Manteniendo el valor de prueba
                    
                    # Dibujar la BBox y la emoción
                    cv2.rectangle(frame_con_procesamiento, (x, y), (x + w, y + h), (255, 100, 0), 2)
                    cv2.putText(frame_con_procesamiento, emocion_detectada, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)
            
                 # 🚨 RETORNO GARANTIZADO: Siempre devuelve 2 valores 🚨
            return emocion_detectada, frame_con_procesamiento

        except Exception as e:
            # Si ocurre CUALQUIER error (ej. cv2.cvtColor falló, un índice salió mal)
            print(f"Error en EmotionDetector: {e}")
            # 🚨 RETORNO GARANTIZADO EN CASO DE FALLO INTERNO 🚨
            return "ERROR_EMOCION", frame_con_procesamiento