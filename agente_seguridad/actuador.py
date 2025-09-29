# agente_seguridad/actuador.py
import cv2
import time
from agente_seguridad.constantes import UMBRAL_ROJO

def dibujar_cajas(frame, persona_data):
    """Dibuja los recuadros con el color y puntaje final."""
    x1, y1, x2, y2 = persona_data['bbox_persona']
    puntaje = persona_data['puntaje']
    
    # 1. Decisión de Color
    if puntaje >= UMBRAL_ROJO:
        color = (0, 0, 255)  # ROJO
    elif puntaje > 0:
        color = (0, 255, 255) # AMARILLO
    else:
        color = (0, 255, 0)  # VERDE (Comportamiento y estado normal)
        
    # Dibujar Recuadro General
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"ID:{persona_data['id_maestro']} P:{puntaje} {persona_data.get('emocion', '')}", 
                (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
    # Dibujar Recuadros Internos (Objetos Sospechosos)
    for obj in persona_data['objetos_sospechosos']:
        ox1, oy1, ox2, oy2 = obj['bbox']
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
        cv2.putText(frame, obj['clase'], (ox1, oy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

def tomar_snapshot(frame, persona_data):
    """Captura foto si el riesgo es alto Y la persona está muy cerca."""
    if persona_data['puntaje'] >= UMBRAL_ROJO and persona_data.get('es_cercano', False):
        x1, y1, x2, y2 = persona_data['bbox_persona']
        snapshot_crop = frame[y1:y2, x1:x2]
        nombre_archivo = f"snapshots/ALERTA_{persona_data['id_maestro']}_P{persona_data['puntaje']}_{int(time.time())}.jpg"
        cv2.imwrite(nombre_archivo, snapshot_crop)
        print(f"--- ACTUACIÓN: Snapshot de ALERTA capturado para ID {persona_data['id_maestro']} ---")

def activar_alarma_gpio(alerta):
    """Simula la activación de un hardware de alerta."""
    if alerta == "PELIGRO_INMINENTE":
        print("🚨🚨 ALARMA FÍSICA ACTIVADA: PELIGRO INMINENTE 🚨🚨")