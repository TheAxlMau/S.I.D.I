# agente_seguridad/agente_central.py

import cv2
import numpy as np 
# Importa todas las constantes, incluidas las nuevas (PESOS_SOSPECHA, UMBRALES, etc.)
from agente_seguridad.constantes import PESOS_SOSPECHA, UMBRAL_ROJO, UMBRAL_AMARILLO, UMBRAL_SNAPSHOT_AREA
from agente_seguridad.actuador import dibujar_cajas, tomar_snapshot, activar_alarma_gpio


class AgenteCentral:
    """Motor de Decisión basado en el Puntaje de Utilidad."""
    def __init__(self, broker, emotion_detector):
        self.broker = broker
        self.emotion_detector = emotion_detector

    def _chequear_proximidad(self, bbox_persona):
        x1, y1, x2, y2 = bbox_persona
        # Cálculo del área de la BBox para estimar proximidad (el objeto es grande)
        area = (x2 - x1) * (y2 - y1)
        return area > UMBRAL_SNAPSHOT_AREA

    def _calcular_puntaje(self, frame_rostro, persona_data):
        """Calcula el Puntaje de Sospecha (Utilidad) unificando los datos."""
        puntaje = 0
        # Inicialización segura: copia el frame si existe, sino None.
        frame_rostro_procesado = frame_rostro.copy() if frame_rostro is not None and frame_rostro.size > 0 else None

        detecciones = persona_data.get('objetos_sospechosos', []) 
        
        # 1. Comportamiento (Merodeo)
        if persona_data.get('merodeo_activo', False):
            puntaje += PESOS_SOSPECHA['merodeo']
            
        # 2. Objetos y Prendas de Riesgo (ACCESORIOS DE CUBRIMIENTO)
        es_rostro_cubierto_det = False
        for obj in detecciones:
            clase = obj['clase'].lower()
            
            # 🚨 LÓGICA DE RIESGO DE CUBRIMIENTO FACIAL (USANDO TUS CLASES) 🚨
            if clase == 'rostro_cubierto':
                puntaje += PESOS_SOSPECHA['rostro_cubierto']
                es_rostro_cubierto_det = True # Marcamos el flag
            
            elif clase == 'gorro':
                puntaje += PESOS_SOSPECHA['gorro']
            
            elif clase == 'gafas':
                puntaje += PESOS_SOSPECHA['gafas']

        # 3. Análisis Facial (Emoción)
        persona_data['emocion'] = 'No_Rostro'
        
        # Solo procesamos si el frame es válido Y no se detectó cubrimiento facial
        if frame_rostro_procesado is not None and not es_rostro_cubierto_det:
            
            # 🚨 Detección y análisis de emoción 🚨
            emocion, frame_rostro_proc = self.emotion_detector.detectar_y_analizar(
                frame_rostro_procesado, 
                persona_data.get('bbox_persona')
            )
            
            persona_data['emocion'] = emocion
            
            if frame_rostro_proc is not None:
                # Actualizar el frame con las cajas y etiquetas de emoción
                frame_rostro_procesado = frame_rostro_proc

            # PUNTAJE POR EMOCIÓN NEGATIVA
            if emocion in ['Enojado', 'Miedo', 'Triste', 'Disgusto']:
                puntaje += PESOS_SOSPECHA['emocion_enojo']
                
        elif es_rostro_cubierto_det:
            # Si el rostro está cubierto, la emoción es irrelevante
            persona_data['emocion'] = 'Rostro_Cubierto'
            
        # 4. Proximidad y Snapshot Condicional
        bbox_persona_alta = persona_data.get('bbox_persona', [0, 0, 0, 0])
        persona_data['es_cercano'] = self._chequear_proximidad(bbox_persona_alta)
            
        return puntaje, persona_data, frame_rostro_procesado

    def procesar_y_actuar(self, datos_unificados, frame_alta, frame_rostro):
        """Ciclo principal de Decisión y Actuación."""
        alerta_general = "NORMAL"
        frame_alta_final = frame_alta
        frame_rostro_final = frame_rostro 
        
        try:
            for id_m, data in datos_unificados.items():
                # 1. CALCULAR PUNTAJE Y EMOCIÓN
                # Usar el frame_rostro_final actual para el cálculo de emoción.
                puntaje, data, frame_rostro_proc = self._calcular_puntaje(frame_rostro, data)
                data['puntaje'] = puntaje
                
                if frame_rostro_proc is not None:
                    frame_rostro_final = frame_rostro_proc

                # 2. DECIDIR la Alerta INDIVIDUAL
                if puntaje >= UMBRAL_ROJO:
                    alerta_actual = "PELIGRO_INMINENTE"
                elif puntaje >= UMBRAL_AMARILLO: 
                    alerta_actual = "VIGILANCIA_ACTIVA"
                else:
                    alerta_actual = "NORMAL"
                
                # 3. ACTUALIZAR la Alerta General (La más alta gana)
                if alerta_actual == "PELIGRO_INMINENTE":
                    alerta_general = "PELIGRO_INMINENTE"
                elif alerta_actual == "VIGILANCIA_ACTIVA" and alerta_general == "NORMAL":
                    alerta_general = "VIGILANCIA_ACTIVA"
                
                # 4. ACTUAR Visualmente (Dibujar)
                if frame_alta_final is not None:
                    frame_alta_final = dibujar_cajas(frame_alta_final, data) 
                if frame_rostro_final is not None:
                    # El frame_rostro_final ya contiene el dibujo de emoción si se detectó
                    frame_rostro_final = dibujar_cajas(frame_rostro_final, data)
                    
                # 5. ACTUAR Físicamente (Snapshot)
                tomar_snapshot(frame_rostro_final, data)

            # 6. Actuar sobre el entorno (Alarma, Foco)
            activar_alarma_gpio(alerta_general)
            
            # RETORNO GARANTIZADO EN CASO DE ÉXITO
            return alerta_general, frame_alta_final, frame_rostro_final
        
        except Exception as e:
            # RETORNO GARANTIZADO EN CASO DE FALLO
            print(f"ERROR CRÍTICO DENTRO DEL AGENTE CENTRAL: {e}")
            # Devuelve los frames no procesados para que la interfaz no se rompa
            return "ERROR_LOGICA", frame_alta, frame_rostro