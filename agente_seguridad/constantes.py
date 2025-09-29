# agente_seguridad/constantes.py
import cv2
import os

# 🚨 Rutas de Archivos: Usaremos 'best.pt' para ambas cámaras 🚨
MODELO_YOLO_ALTA_PATH = 'Modelos/best.pt' 
MODELO_YOLO_ROSTRO_PATH = 'Modelos/best.pt' 
MODELO_EMOCION_PATH = 'Modelos/cnn_emociones.h5' 
HAARCASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' 

# Rutas de Archivos
# 🚨 ¡ACTUALIZADO! Apunta a tu modelo entrenado best.pt
MODELO_YOLO_PATH = 'Modelos/yolov8_fashion/best.pt'  
MODELO_EMOCION_PATH = 'Modelos/cnn_emociones.h5' 
HAARCASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Parámetros del Agente
UMBRAL_REID = 100000        
UMBRAL_MERODEO = 3          
VENTANA_MERODEO = 300       
UMBRAL_SNAPSHOT_AREA = 300 * 300 

# Lógica del Puntaje de Sospecha (Utilidad): AJUSTADA A TUS CLASES
PESOS_SOSPECHA = {
    # Riesgo de encubrimiento/detección
    'rostro_cubierto': 50, # Alto riesgo
    'gorro': 20,           # Moderado
    'gafas': 5,            # Bajo (solo accesorio)
    
    # Comportamiento/Emoción
    'merodeo': 30,
    'emocion_enojo': 20,
    'rostro_autorizado': -50 
}
UMBRAL_ROJO = 50   
UMBRAL_AMARILLO = 1 

# 🚨 Mapeo de IDs: DEBE COINCIDIR EXACTAMENTE CON TU data.yaml 🚨
CLASES_ENTRENAMIENTO = {
    'rostro_cubierto': 0, 
    'rostro_no_cubierto': 1, 
    'gafas': 2, 
    'gorro': 3, 
    'cuerpo_superior': 4, 
    'cuerpo_inferior': 5
}
# 🚨 NUEVA DEFINICIÓN DE IDs 🚨
# IDs usados para el Bounding Box PRINCIPAL de la persona (no tienes 'persona')
IDS_PERSONA_PRINCIPAL = {CLASES_ENTRENAMIENTO['cuerpo_superior'], CLASES_ENTRENAMIENTO['cuerpo_inferior']} 

# IDs usados para Re-Identificación (ropa)
IDS_ROPA = {CLASES_ENTRENAMIENTO['cuerpo_superior'], CLASES_ENTRENAMIENTO['cuerpo_inferior']} 

# IDs considerados de RIESGO
IDS_RIESGO = {CLASES_ENTRENAMIENTO['rostro_cubierto'], CLASES_ENTRENAMIENTO['gorro'], CLASES_ENTRENAMIENTO['gafas']} 

# MODELO_YOLO_PATH ya no se usa directamente en percepcion.py
# Se reemplaza por los dos paths específicos de cámara