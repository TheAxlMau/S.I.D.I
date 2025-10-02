# config/config_sids.py

import collections

# --- CONFIGURACIÓN DE CÁMARA E INTERFAZ ---
CAMERA_ID = 0 
TARGET_WIDTH = 640  
TARGET_HEIGHT = 480
# ¡Nuevo Nombre!
WINDOW_NAME = 'S.I.D.S. - Sistema Inteligente de Detección de Sospecha'

# --- CONFIGURACIÓN DE MODELOS ---
MODELO_PERSONALIZADO_ACTIVO = False 
MODELO_YOLO_PATH = './best.pt' if MODELO_PERSONALIZADO_ACTIVO else 'yolov8n.pt'

# --- OPTIMIZACIÓN DE RENDIMIENTO PARA RASPBERRY PI ---
# Procesar la detección de IA cada 5 frames (reducir carga de CPU)
FRAME_SKIP_RATE = 5 
# El tracking de YOLO se ejecuta en todos los frames.

# --- UMBRALES DE CALIBRACIÓN DE ANOMALÍAS ---
UMBRAL_MOVIMIENTO_BRUSCO = 0.10   
FACTOR_CORRECCION_Y = 0.5 
UMBRAL_RADIO_MERODEO = 0.02       
FRAMES_MERODEO = 150              
UMBRAL_FRAMES_ESTATICO = 75       
HISTORY_LENGTH = 30 

# --- UMBRAL DE DETECCIÓN DE MIRADA ---
UMBRAL_MIRADA_FRONTAL = 0.05 