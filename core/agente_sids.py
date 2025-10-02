# core/agente_sids.py

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from core.anomaly_analyzer import AnomalyAnalyzer
import config.config_sids as cfg
import multiprocessing as mp_
import time

# Renombramos la clase AgenteSIDS a SIDSWorker para reflejar su rol de procesador pesado
class SIDSWorker:
    def __init__(self, input_queue, output_queue):
        # Colas para comunicación entre procesos
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        # Módulos de Percepción (Solo se inicializan una vez)
        self.model = YOLO(cfg.MODELO_YOLO_PATH) 
        self.mp_pose = mp.solutions.pose
        self.pose_model = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Módulo de Decisión (Mantendrá el estado entre detecciones)
        self.analyzer = AnomalyAnalyzer()
        self.width = cfg.TARGET_WIDTH
        self.height = cfg.TARGET_HEIGHT
        
        # ID para mantener el estado del tracking
        self.yolo_track_id_state = None 
        
        print(f"✅ Agente S.I.D.S. Worker inicializado. Usando modelo: {cfg.MODELO_YOLO_PATH}")

    def _analizar_mirada(self, pose_landmarks):
        # (La implementación de _analizar_mirada es la misma que la versión anterior)
        if not pose_landmarks:
            return False, (255, 255, 255) 

        landmarks = pose_landmarks.landmark
        nariz = np.array([landmarks[self.mp_pose.PoseLandmark.NOSE].x, landmarks[self.mp_pose.PoseLandmark.NOSE].y])
        hombro_izq = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        hombro_der = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
                               
        centro_cuerpo = (hombro_izq + hombro_der) / 2
        desviacion_horizontal = abs(nariz[0] - centro_cuerpo[0])
        
        if desviacion_horizontal < cfg.UMBRAL_MIRADA_FRONTAL:
            return True, (0, 165, 255) # Naranja/Azul claro para Mirada
        
        return False, (255, 255, 255)

    def percibir_y_decidir(self, frame, tracking_only=False):
        """
        PERCIBIR Y DECIDIR: Ejecuta el pipeline completo de IA.
        Si tracking_only=True, solo corre YOLO track para actualizar IDs/posiciones.
        """
        track_boxes = []
        pose_results = None

        # 1. Detección COMPLETA (Solo cada N frames o al inicio)
        if not tracking_only:
            # YOLO Detección/Tracking (persist=True mantiene el estado de seguimiento)
            results = self.model.track(
                frame, persist=True, tracker='bytetrack.yaml', classes=[0], conf=0.3, iou=0.5, verbose=False 
            )
            track_boxes = results[0].boxes.data.cpu().numpy() if results and results[0].boxes.id is not None else []
            
            # MediaPipe Pose (MUY COSTOSO - Solo en detección completa)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose_model.process(frame_rgb)
        
        # 2. Tracking LIGERO (para frames intermedios - si se implementa en main)
        # Nota: El 'track' de Ultralytics ya mantiene el estado si se llama con persist=True.

        # 3. DECIDIR (Análisis)
        analysis_results, is_global_anomaly = self._decidir(track_boxes, pose_results)
        
        # Devolvemos solo las cajas y resultados de anomalía para dibujar
        return track_boxes, analysis_results, is_global_anomaly

    def _decidir(self, track_boxes, pose_results):
        """Aplica las reglas de anomalía (movimiento) Y comportamiento (mirada)."""
        analysis_results = []
        is_global_anomaly = False
        
        for box in track_boxes:
            x1, y1, x2, y2, track_id, conf, cls = box
            track_id = int(track_id)
            
            # Centro de la caja normalizado
            center_x_norm = (x1 + x2) / 2 / self.width
            center_y_norm = (y1 + y2) / 2 / self.height
            current_pos = np.array([center_x_norm, center_y_norm])

            # 1. Decisión de MOVIMIENTO
            anomaly_text, color, is_anomalous = self.analyzer.analyze(track_id, current_pos)
            
            # 2. Decisión de COMPORTAMIENTO (Mirada a Cámara - solo si hay pose)
            is_looking = False
            if pose_results and pose_results.pose_landmarks:
                is_looking, mirada_color = self._analizar_mirada(pose_results.pose_landmarks)
            
            if is_looking:
                if is_anomalous:
                    anomaly_text += " + MIRA CÁMARA"
                    color = (255, 0, 255) # Púrpura 
                else:
                    anomaly_text = "MIRA CÁMARA"
                    color = mirada_color 
                    is_anomalous = True

            if is_anomalous:
                is_global_anomaly = True
                
            analysis_results.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'track_id': track_id,
                'text': anomaly_text,
                'color': color,
            })
            
        return analysis_results, is_global_anomaly

    def run_process(self):
        """Bucle principal del proceso worker."""
        # Se requiere un bucle infinito que espere datos en la cola de entrada
        while True:
            # Intentar obtener el frame sin bloquear el hilo
            try:
                frame_id, frame_bgr = self.input_queue.get(timeout=0.001) 
            except mp_.queues.Empty:
                time.sleep(0.001)
                continue

            # Ejecutar Detección Completa solo si corresponde (Omisión de Frames)
            tracking_only = (frame_id % cfg.FRAME_SKIP_RATE != 0)
            
            # Percibir y Decidir
            track_boxes, analysis_results, is_global_anomaly = self.percibir_y_decidir(frame_bgr, tracking_only)

            # Enviar los resultados (leves) de vuelta a la GUI
            if not self.output_queue.full():
                self.output_queue.put((frame_id, track_boxes, analysis_results, is_global_anomaly))
                
# --- Función de inicialización del proceso ---
def start_sids_worker(input_queue, output_queue):
    worker = SIDSWorker(input_queue, output_queue)
    worker.run_process()