# agente_seguridad/percepcion.py

import cv2
from ultralytics import YOLO
import numpy as np
from agente_seguridad.constantes import (
    MODELO_YOLO_ALTA_PATH, MODELO_YOLO_ROSTRO_PATH, 
    IDS_ROPA, IDS_RIESGO, CLASES_ENTRENAMIENTO, 
    IDS_PERSONA_PRINCIPAL
)

class Percepcion:
    """Clase base para capturar video y detección/tracking con YOLO."""
    def __init__(self, camera_index, es_camara_rostro=False):
        # 🚨 USAR LA RUTA CORRECTA SEGÚN EL TIPO DE CÁMARA 🚨
        modelo_path = MODELO_YOLO_ROSTRO_PATH if es_camara_rostro else MODELO_YOLO_ALTA_PATH
        self.modelo = YOLO(modelo_path)
        
        # 2. Inicialización de la cámara
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"DEBUG - Cámara {camera_index} inicializada a: {actual_width}x{actual_height} con modelo: {modelo_path}")
        
        if self.modelo.names:
            print(f"✅ DEBUG PERCEPCIÓN: Modelo YOLO cargado con {len(self.modelo.names)} clases.")
        else:
            print("❌ ERROR PERCEPCIÓN: Fallo al cargar el modelo YOLO. Revise la ruta en constantes.py.")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def procesar_frame(self, frame):
        """Aplica YOLO y agrupa todas las detecciones por ID."""
        
        resultados = self.modelo.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            conf=0.4, 
            verbose=False
        )
        
        datos_por_persona = {}
        # Inicializar con el frame original en caso de que no haya detecciones
        frame_yolo_plot = resultados[0].plot() if resultados and hasattr(resultados[0], 'plot') else frame.copy() 

        if resultados and resultados[0].boxes.id is not None:
            
            for box in resultados[0].boxes:
                # Obtención de datos
                track_id = int(box.id.tolist()[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                clase_id = int(box.cls.tolist()[0])
                nombre_clase = self.modelo.names.get(clase_id, "unknown")
                
                # Inicializar el diccionario para el nuevo ID
                if track_id not in datos_por_persona:
                     datos_por_persona[track_id] = {
                        "id_local": track_id, "bbox_persona": None, 
                        "ropa_bboxes": [], "objetos_sospechosos": [],
                        "nombre_clase_principal": nombre_clase
                    }
                    
                # Lógica de Agrupamiento y BBox Envolvente (Principal)
                
                # 🚨 CORRECCIÓN CRÍTICA: BBox Envolvente para la Persona 🚨
                if clase_id in IDS_PERSONA_PRINCIPAL:
                    bbox_actual = (x1, y1, x2, y2)
                    
                    if datos_por_persona[track_id]["bbox_persona"] is None:
                        datos_por_persona[track_id]["bbox_persona"] = bbox_actual
                    else:
                        # Si ya tenemos un BBox principal, lo extendemos para incluir las nuevas partes corporales
                        px1, py1, px2, py2 = datos_por_persona[track_id]["bbox_persona"]
                        datos_por_persona[track_id]["bbox_persona"] = (
                            min(px1, x1), min(py1, y1), max(px2, x2), max(py2, y2)
                        )
                        
                    # Las partes del cuerpo también se usan como "ropa" para ReID
                    datos_por_persona[track_id]["ropa_bboxes"].append(bbox_actual)

                # Si es un objeto de riesgo (cubrimiento)
                elif clase_id in IDS_RIESGO:
                    datos_por_persona[track_id]["objetos_sospechosos"].append({
                        'bbox': (x1, y1, x2, y2), 
                        'clase': nombre_clase
                    })


        # Convertir a una lista de detecciones limpias (solo si se detectó una BBox corporal)
        detecciones_limpias = []
        for _, data in datos_por_persona.items():
            if data["bbox_persona"] is not None:
                detecciones_limpias.append(data)

        return detecciones_limpias, frame_yolo_plot