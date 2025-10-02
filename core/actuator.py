# core/actuator.py

import cv2

def draw_results(frame, analysis_results):
    """
    ACTUAR: Dibuja los resultados en el frame con texto legible.
    Este módulo se ejecuta en el proceso principal (GUI) y es muy rápido.
    """
    total_persons = len(analysis_results)
    height, width, _ = frame.shape
    
    for res in analysis_results:
        x1, y1, x2, y2 = res['box']
        color = res['color']
        anomaly_text = f"ID {res['track_id']}: {res['text']}"
        
        # 1. Definir la posición del texto
        text_size = cv2.getTextSize(anomaly_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x1
        text_y = y1 - 15 
        
        # Corrección si el texto se sale de la parte superior del frame
        if text_y < 10: 
            text_y = y1 + text_size[1] + 5 

        # 2. Dibujar un fondo para el texto (Negro para legibilidad)
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + 5), 
                      (0, 0, 0), cv2.FILLED) 
        
        # 3. Dibujar la caja y el texto (Blanco sobre Negro)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, anomaly_text, (text_x + 3, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA) 
    
    # Dibuja un texto de overlay pequeño en el frame
    cv2.putText(frame, f"Personas detectadas: {total_persons}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return frame