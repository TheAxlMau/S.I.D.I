import cv2
import threading
import time
import numpy as np

# Asegúrate de que estos módulos existen en tu carpeta 'agente_seguridad/'
from agente_seguridad.reid_broker import ReIDBroker
from agente_seguridad.agente_central import AgenteCentral
from agente_seguridad.emotion_detector import EmotionDetector
from agente_seguridad.percepcion import Percepcion 
from agente_seguridad.constantes import (
    MODELO_EMOCION_PATH, HAARCASCADE_PATH, 
    UMBRAL_ROJO, UMBRAL_AMARILLO 
) 

# ----------------------------------------------------
# CLASE PARA EJECUTAR CADA CÁMARA EN UN HILO SEPARADO
# ----------------------------------------------------

class CameraWorker(threading.Thread):
    def __init__(self, camera_id, camera_index, broker, es_camara_rostro=False):
        super().__init__()
        self.camera_id = camera_id
        # 🚨 CORRECCIÓN: Pasar el flag de cámara rostro a Percepcion 🚨
        self.perceptor = Percepcion(camera_index, es_camara_rostro=es_camara_rostro) 
        self.broker = broker
        self.es_camara_rostro = es_camara_rostro
        self.ultima_deteccion_yolo = None 
        self.ultimo_frame_raw = None
        self._stop_event = threading.Event() 

    def run(self):
        """Captura frames y envía al Broker para su procesamiento."""
        print(f"DEBUG: Hilo {self.camera_id} iniciado.")
        
        while not self._stop_event.is_set():
            ret, frame = self.perceptor.cap.read()
            
            if not ret or frame is None:
                # Si la cámara no está lista o falla, espera y continúa
                time.sleep(0.1)
                continue
            
            # 1. Procesamiento YOLO / Detección
            # frame_procesado es el frame con las cajas dibujadas por YOLO
            # lista_detecciones es la lista de datos limpios
            frame_procesado, lista_detecciones = self.perceptor.procesar_frame(
                frame, 
                self.camera_id
            )
            
            # 2. Envío al Broker
            self.broker.procesar_detecciones_y_unificar( 
                self.camera_id, 
                lista_detecciones, 
                frame, 
                self.es_camara_rostro
            )
            
            # 3. Guardar resultados para la interfaz visual
            # 🚨 Se asegura que solo se copien frames válidos 🚨
            if frame_procesado is not None:
                self.ultima_deteccion_yolo = frame_procesado.copy() 
            if frame is not None:
                self.ultimo_frame_raw = frame.copy() 
                
            time.sleep(1/30) # Control de velocidad

        if self.perceptor.cap.isOpened():
            self.perceptor.cap.release()
            print(f"DEBUG: Hilo {self.camera_id} detenido y cámara liberada.")


    def stop(self):
        self._stop_event.set()


# ----------------------------------------------------
# MAIN: INICIO DE HILOS Y CICLO DE DECISIÓN
# ----------------------------------------------------
def main():
    print("🤖 Iniciando Agente Inteligente Multi-Modal...")
    
    # --- Configuración Inicial ---
    
    broker = ReIDBroker()
    detector_emociones = EmotionDetector(MODELO_EMOCION_PATH, HAARCASCADE_PATH)
    agente_central = AgenteCentral(broker, detector_emociones)

    # Índices de Cámaras
    # 🚨 NOTA: Si solo tienes una cámara, usa (0, 0) para debug. 🚨
    INDEX_CAMARA_ALTA = 1
    INDEX_CAMARA_ROSTRO = 0

    camara_alta = CameraWorker("CAM_ALTA", INDEX_CAMARA_ALTA, broker, es_camara_rostro=False)
    camara_rostro = CameraWorker("CAM_ROSTRO", INDEX_CAMARA_ROSTRO, broker, es_camara_rostro=True)
    
    camara_alta.start()
    camara_rostro.start()
    
    # --- Configuración de la Interfaz Visual ---
    W, H = 640, 360 
    BANNER_H = 40 
    FRAME_H_TOTAL = H + BANNER_H
    CANVAS_H = 180 
    
    # Colores BGR
    COLOR_VERDE = (0, 255, 0)
    COLOR_AMARILLO = (0, 255, 255)
    COLOR_ROJO = (0, 0, 255)
    COLOR_BLANCO = (255, 255, 255)
    COLOR_GRIS_OSCURO = (30, 30, 30)
    COLOR_AZUL_MARINO = (100, 50, 20) 

    alerta = "INICIANDO"
    
    frame_visual_alta = np.zeros((H, W, 3), dtype=np.uint8) 
    frame_visual_rostro = np.zeros((H, W, 3), dtype=np.uint8)
    
    workers = [camara_alta, camara_rostro]
    
    # Asignación de umbrales desde constantes
    try:
        umbral_rojo = UMBRAL_ROJO
        umbral_amarillo = UMBRAL_AMARILLO
    except NameError:
        umbral_rojo = 70
        umbral_amarillo = 30
        print("ADVERTENCIA: Usando umbrales por defecto (Rojo=70, Amarillo=30).")


    while True:
        # 1. Obtener datos unificados y frames para el Agente Central
        datos_unificados = broker.obtener_estado_unificado() 
        
        try:
            # 2. PROCESO DE DECISIÓN Y ACTUACIÓN
            # 🚨 USO DE frames_raw y frames_procesados (Asegura que el Agent Central tenga los frames) 🚨
            alerta, frame_visual_alta_proc, frame_visual_rostro_proc = agente_central.procesar_y_actuar(
                datos_unificados, 
                camara_alta.ultima_deteccion_yolo, # El frame ya dibujado por YOLO
                camara_rostro.ultimo_frame_raw     # El frame raw para la detección de rostro
            )
            
            # 🚨 Es vital que solo se asignen frames si no son None 🚨
            if frame_visual_alta_proc is not None:
                frame_visual_alta = frame_visual_alta_proc
            if frame_visual_rostro_proc is not None:
                frame_visual_rostro = frame_visual_rostro_proc

        except Exception as e:
            # 🚨 Este except evita que todo el programa colapse si el Agente Central falla 🚨
            print(f"Error en el Agente Central (bucle principal): {e}")
            alerta = "ERROR CRITICO"

        
        # --- 1. PREPARACIÓN DE PANELES DE CÁMARA (Con Banners) ---
        
        frame_alta_resized = cv2.resize(frame_visual_alta, (W, H))
        frame_rostro_resized = cv2.resize(frame_visual_rostro, (W, H))

        # Panel para Cámara Alta (Total: 640x400)
        panel_alta = np.zeros((FRAME_H_TOTAL, W, 3), dtype=np.uint8)
        panel_alta[BANNER_H:FRAME_H_TOTAL, 0:W] = frame_alta_resized
        cv2.rectangle(panel_alta, (0, 0), (W, BANNER_H), COLOR_AZUL_MARINO, -1)
        cv2.putText(panel_alta, "CAMARA ALTA | VISION GENERAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BLANCO, 2)
        cv2.line(panel_alta, (0, BANNER_H - 1), (W, BANNER_H - 1), COLOR_AMARILLO, 1)

        # Panel para Cámara Rostro (Total: 640x400)
        panel_rostro = np.zeros((FRAME_H_TOTAL, W, 3), dtype=np.uint8)
        panel_rostro[BANNER_H:FRAME_H_TOTAL, 0:W] = frame_rostro_resized
        cv2.rectangle(panel_rostro, (0, 0), (W, BANNER_H), COLOR_AZUL_MARINO, -1)
        cv2.putText(panel_rostro, "CAMARA ROSTRO | ANALISIS FACIAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BLANCO, 2)
        cv2.line(panel_rostro, (0, BANNER_H - 1), (W, BANNER_H - 1), COLOR_AMARILLO, 1)


        # --- 2. PANEL DE DETALLES CON ESTÉTICA MEJORADA ---

        canvas_details = np.zeros((CANVAS_H, W * 2, 3), dtype=np.uint8)
        cv2.rectangle(canvas_details, (0, 0), (W * 2, CANVAS_H), COLOR_GRIS_OSCURO, -1)
        
        # 2.1 Definir colores y textos según la alerta
        if alerta == "NORMAL":
            color_fondo = COLOR_VERDE
            color_texto = (0, 0, 0) 
            texto_status = "OPERACIÓN NORMAL"
        elif alerta == "VIGILANCIA_ACTIVA":
            color_fondo = COLOR_AMARILLO
            color_texto = (0, 0, 0) 
            texto_status = "ATENCIÓN REQUERIDA"
        elif alerta == "PELIGRO_INMINENTE":
            color_fondo = COLOR_ROJO
            color_texto = COLOR_BLANCO
            texto_status = "ALERTA MAXIMA"
        else: # ERROR CRITICO o ERROR_LOGICA
            color_fondo = COLOR_ROJO
            color_texto = COLOR_BLANCO
            texto_status = "ERROR DE SISTEMA"

        # 2.2 Dibujar Panel de ESTADO GENERAL (Mitad izquierda)
        ALERTA_W = W - 20 
        ALERTA_H_FINAL = 90
        cv2.rectangle(canvas_details, (10, 10), (ALERTA_W, ALERTA_H_FINAL), color_fondo, -1)
        
        cv2.putText(canvas_details, "ESTADO GENERAL", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        cv2.putText(canvas_details, texto_status, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_texto, 3)
        
        cv2.putText(canvas_details, "Latencia: X ms", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLANCO, 1)
        
        # 2.3 Dibujar Panel de DETALLES DE PERSONAS (Mitad derecha)
        x_start_person = W + 10
        cv2.putText(canvas_details, "PERSONAS DETECTADAS (ID | Puntaje | Emoción)", 
                    (x_start_person, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BLANCO, 2)
        
        y_offset = 65
        # Separador visual
        cv2.line(canvas_details, (W + 5, 45), (W * 2 - 10, 45), COLOR_AMARILLO, 1)
        
        
        for id_m, data in datos_unificados.items():
            puntaje = data.get('puntaje', 0)
            emocion = data.get('emocion', 'N/A')
            merodeo = "SÍ" if data.get('merodeo_activo', False) else "NO"
            
            # Formateo de los detalles
            detail_text = f"ID: R_{id_m} | P: {puntaje:02d} | Emoción: {emocion: <8} | Merodeo: {merodeo}"
            
            # Color del texto según el puntaje de cada persona
            color_puntaje_texto = COLOR_VERDE
            if puntaje >= umbral_rojo:
                color_puntaje_texto = COLOR_ROJO
            elif puntaje >= umbral_amarillo:
                color_puntaje_texto = COLOR_AMARILLO
                
            cv2.putText(canvas_details, detail_text, (x_start_person, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_puntaje_texto, 2)
            y_offset += 25 

        # --- 3. CONCATENACIÓN FINAL ---
        frames_concatenados = cv2.hconcat([panel_alta, panel_rostro])
        interfaz_final = cv2.vconcat([frames_concatenados, canvas_details])

        cv2.imshow('Interfaz de Vigilancia Centralizada', interfaz_final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for w in workers:
        w.stop()
        w.join()
        
    cv2.destroyAllWindows()
    print("✅ Agente de Seguridad detenido.")

if __name__ == "__main__":
    main()