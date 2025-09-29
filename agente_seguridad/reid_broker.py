# agente_seguridad/reid_broker.py
import numpy as np
import time
from scipy.spatial.distance import euclidean
from agente_seguridad.constantes import UMBRAL_REID, VENTANA_MERODEO, UMBRAL_MERODEO

class ReIDBroker:
    """Gestiona el estado interno, la unificación de ID (Multi-Cámara) y el Merodeo."""
    def __init__(self):
        self.estado_global = {}  # {ID_Maestro: {data, historial_merodeo, timestamp}}
        self.detecciones_camaras = {"CAM_ALTA": {}, "CAM_ROSTRO": {}}
        self.reid_perdidos = {}

    # --- LÓGICA DE CARACTERÍSTICAS Y DISTANCIA ---
    def _extraer_caracteristicas_ropa(self, frame, bounding_boxes_ropa):
        """Calcula el vector de características de la ropa (hist. de color)."""
        if not bounding_boxes_ropa:
            return None 

        h_total, w_total = frame.shape[:2]
        mascara = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for x1, y1, x2, y2 in bounding_boxes_ropa:
            mascara[y1:y2, x1:x2] = 255
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mascara, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist.flatten()

    def _distancia_caracteristicas(self, feat1, feat2):
        if feat1 is None or feat2 is None:
            return float('inf')
        return euclidean(feat1, feat2)

    # --- LÓGICA DE UNIFICACIÓN DE ID (CROSS-CAMERA) ---
    def unificar_id(self, deteccion, frame, es_camara_rostro):
        """Busca el mejor ID_Maestro que coincida con la detección actual."""
        
        caracteristicas_actuales = self._extraer_caracteristicas_ropa(frame, deteccion['ropa_bboxes'])
        deteccion['caracteristicas_ropa'] = caracteristicas_actuales
        
        mejor_id_maestro = None
        min_distancia = float('inf')
        
        # 1. Intentar hacer match con el ESTADO GLOBAL (Multi-cámara ReID)
        for id_maestro, estado in self.estado_global.items():
            dist = self._distancia_caracteristicas(caracteristicas_actuales, estado.get('caracteristicas_ropa'))
            
            if dist < min_distancia and dist < UMBRAL_REID:
                min_distancia = dist
                mejor_id_maestro = id_maestro
        
        if mejor_id_maestro is not None:
            return mejor_id_maestro
        else:
            # 2. Si no hay match, crear un nuevo ID_Maestro basado en el ID local
            # Esto asegura que el ID_Maestro es único y persistente.
            nuevo_id = f"{'R' if es_camara_rostro else 'A'}_{deteccion['id_local']}_{int(time.time()*100)}"
            return nuevo_id

    # --- PROCESAMIENTO DE HILO DE CÁMARA ---
    def procesar_detecciones_y_unificar(self, cam_id, detecciones, frame, es_camara_rostro):
        ids_actuales = set()
        
        for det in detecciones:
            id_maestro = self.unificar_id(det, frame, es_camara_rostro)
            ids_actuales.add(id_maestro)
            
            # Actualizar/Crear Estado Global
            if id_maestro not in self.estado_global:
                self.estado_global[id_maestro] = {
                    'id_maestro': id_maestro,
                    'historial_merodeo': [],
                    'caracteristicas_ropa': det['caracteristicas_ropa']
                }
            
            # Actualizar historial de merodeo
            self._actualizar_merodeo(id_maestro)

            # Actualizar los datos específicos de la cámara
            det['id_maestro'] = id_maestro
            self.detecciones_camaras[cam_id][id_maestro] = det
        
        # Limpiar IDs de cámaras que ya no están visibles
        ids_a_eliminar = [mid for mid in self.detecciones_camaras[cam_id].keys() if mid not in ids_actuales]
        for mid in ids_a_eliminar:
            del self.detecciones_camaras[cam_id][mid]

    def _actualizar_merodeo(self, id_maestro):
        """Actualiza el estado interno de merodeo."""
        timestamp = time.time()
        tiempos = self.estado_global[id_maestro]['historial_merodeo']
        
        if not tiempos or (timestamp - tiempos[-1]) > 10:
            tiempos.append(timestamp)
        
        # Limpiar el historial que está fuera de la ventana
        tiempos_validos = [t for t in tiempos if timestamp - t < VENTANA_MERODEO]
        self.estado_global[id_maestro]['historial_merodeo'] = tiempos_validos

    def chequear_merodeo(self, id_maestro):
        tiempos = self.estado_global[id_maestro]['historial_merodeo']
        if len(tiempos) >= UMBRAL_MERODEO:
            return True
        return False
        
    def obtener_estado_unificado(self):
        """Compila los datos de ambas cámaras para la decisión del Agente Central."""
        estado_unificado = {}
        todos_ids = set(self.detecciones_camaras['CAM_ALTA'].keys()) | set(self.detecciones_camaras['CAM_ROSTRO'].keys())

        for id_m in todos_ids:
            # Priorizar datos de la cámara de rostro si existe
            data = self.detecciones_camaras['CAM_ROSTRO'].get(id_m)
            if data is None:
                data = self.detecciones_camaras['CAM_ALTA'].get(id_m)

            # Agregar estado global (merodeo)
            data['merodeo_activo'] = self.chequear_merodeo(id_m)
            
            estado_unificado[id_m] = data
            
        return estado_unificado