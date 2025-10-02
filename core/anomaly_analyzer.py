# core/anomaly_analyzer.py

import numpy as np
import collections
# Usamos el path completo del módulo
import config.config_sids as cfg 

class AnomalyAnalyzer:
    """Clase que contiene la lógica heurística para decidir los patrones anómalos."""
    def __init__(self):
        # ... (rest of the __init__ method remains the same)
        self.settings = {
            'UMBRAL_MOVIMIENTO_BRUSCO': cfg.UMBRAL_MOVIMIENTO_BRUSCO,
            'UMBRAL_RADIO_MERODEO': cfg.UMBRAL_RADIO_MERODEO,
            'FRAMES_MERODEO': cfg.FRAMES_MERODEO,
            'UMBRAL_FRAMES_ESTATICO': cfg.UMBRAL_FRAMES_ESTATICO,
            'FACTOR_CORRECCION_Y': cfg.FACTOR_CORRECCION_Y,
            'HISTORY_LENGTH': cfg.HISTORY_LENGTH
        }
        self.last_position = {}     
        self.static_frame_count = {} 
        self.merodeo_frame_count = {}
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=cfg.HISTORY_LENGTH))

    def analyze(self, track_id, current_pos):
        """Analiza la posición y el historial de un objeto (persona)."""
        settings = self.settings
        anomaly_text = "Normal"
        color = (0, 255, 0) 
        is_anomalous = False
        
        self.history[track_id].append(current_pos)
        
        if track_id in self.last_position:
            prev_pos = self.last_position[track_id]
            distance = np.linalg.norm(current_pos - prev_pos) # Velocidad

            # === CALIBRACIÓN DINÁMICA DE VELOCIDAD (CORRECCIÓN DE PERSPECTIVA) ===
            # current_pos[1] es la posición Y normalizada (0.0=arriba, 1.0=abajo)
            y_norm = current_pos[1]
            BASE_UMBRAL = settings['UMBRAL_MOVIMIENTO_BRUSCO']
            CORRECTION_FACTOR = settings['FACTOR_CORRECCION_Y']
            
            # El factor de sensibilidad es más bajo (más sensible) en la parte superior del frame (lejos)
            # y más alto (menos sensible) en la parte inferior del frame (cerca).
            UMBRAL_DINAMICO = BASE_UMBRAL * (1.0 + (y_norm * CORRECTION_FACTOR))


            # --- 1. Detección de Movimiento Brusco (Velocidad) ---
            if distance > UMBRAL_DINAMICO: # <-- ¡Usamos el umbral corregido!
                anomaly_text = "MOVIMIENTO BRUSCO"
                color = (0, 0, 255) 
                is_anomalous = True
                self.static_frame_count[track_id] = 0
                self.merodeo_frame_count[track_id] = 0
            
            # --- 2. Detección de Posición Estática ---
            elif distance < 0.001: 
                self.static_frame_count[track_id] = self.static_frame_count.get(track_id, 0) + 1
                
                if self.static_frame_count[track_id] >= settings['UMBRAL_FRAMES_ESTATICO']:
                    anomaly_text = "ESTATICO ANÓMALO"
                    color = (0, 255, 255) 
                    is_anomalous = True
                    self.merodeo_frame_count[track_id] = 0
            
            # --- 3. Detección de Merodeo (Patrón de Permanencia) ---
            else:
                self.static_frame_count[track_id] = 0
                
                if len(self.history[track_id]) == settings['HISTORY_LENGTH']:
                    oldest_pos = self.history[track_id][0]
                    radius_distance = np.linalg.norm(current_pos - oldest_pos)
                    
                    if radius_distance < settings['UMBRAL_RADIO_MERODEO']:
                        self.merodeo_frame_count[track_id] = self.merodeo_frame_count.get(track_id, 0) + 1
                        
                        if self.merodeo_frame_count[track_id] >= settings['FRAMES_MERODEO']:
                            anomaly_text = "MERODEO/PATRÓN SOSPECHOSO"
                            color = (255, 165, 0) 
                            is_anomalous = True
                    else:
                        self.merodeo_frame_count[track_id] = 0

        self.last_position[track_id] = current_pos
        return anomaly_text, color, is_anomalous