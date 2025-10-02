# main_sids.py

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import sys
import os
import multiprocessing as mp

# =========================================================
# 💡 CORRECCIÓN CRÍTICA: AÑADIR LA CARPETA RAIZ A PYTHON PATH
# Esto permite que las importaciones como 'from core.agente_sids import...' funcionen.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# =========================================================

from core.agente_sids import start_sids_worker 
from core.actuator import draw_results 
import config.config_sids as cfg 

# --- Corrección para silenciar Warnings (la raya naranja) ---
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# -----------------------------------------------------------

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SIDS_App(ctk.CTk): 
    def __init__(self):
        # 1. CRÍTICO: Inicializar la GUI primero para evitar el AttributeError
        super().__init__()
        
        # 2. Configuración de Multiprocesamiento (Worker de IA en proceso separado)
        self.input_queue = mp.Queue(maxsize=1) 
        self.output_queue = mp.Queue(maxsize=1)
        self.worker_process = mp.Process(target=start_sids_worker, 
                                         args=(self.input_queue, self.output_queue))
        self.worker_process.start()
        
        # 3. Variables de Estado de la Detección
        self.last_analysis_results = []
        self.is_global_anomaly = False
        self.frame_id_counter = 0

        # 4. INICIALIZACIÓN DE CÁMARA (Ahora es seguro llamarla)
        self.cap = self._init_camera() 
        
        if not self.cap.isOpened():
            print("❌ ERROR: La cámara no se pudo iniciar. Saliendo...")
            self.on_closing()
            return
            
        # Sincronizar dimensiones
        self.agente_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.agente_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.setup_ui()
        self.delay = 1 
        self.is_running = True
        self.update_video()

    def _init_camera(self):
        """Inicializa el objeto de captura de video de OpenCV."""
        cap = cv2.VideoCapture(cfg.CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.TARGET_HEIGHT)
        return cap
        
    def setup_ui(self):
        """Configura todos los elementos de la interfaz de CustomTkinter."""
        self.title(cfg.WINDOW_NAME)
        self.geometry("1000x700") 
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Panel Superior (Estado)
        self.status_label = ctk.CTkLabel(self, text="S.I.D.S. Iniciado. Estado OK.", 
                                        font=ctk.CTkFont(size=20, weight="bold"))
        self.status_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        # Contenedor de Video (Centrado y Expansivo)
        self.video_label = ctk.CTkLabel(self, text="Cargando Video...", fg_color="#333")
        self.video_label.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

    def update_video(self):
        """Bucle principal de la GUI: Captura, envía, recibe y dibuja."""
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.frame_id_counter += 1

            # 1. Enviar el frame al Worker (NO bloquea la GUI)
            if not self.input_queue.full():
                # Enviamos una COPIA para Multiprocesamiento
                self.input_queue.put((self.frame_id_counter, frame.copy()))

            # 2. Recibir los resultados del Worker (NO bloqueante)
            try:
                # Recibimos el resultado más reciente
                _, _, self.last_analysis_results, self.is_global_anomaly = self.output_queue.get(timeout=0.0001)
            except mp.queues.Empty:
                pass # Usa los últimos resultados si no hay uno nuevo disponible
                
            # 3. ACTUAR (Dibujar en el proceso principal - ES RÁPIDO)
            output_frame = draw_results(frame, self.last_analysis_results)
            
            # 4. Actualización de la GUI (Redimensionamiento)
            widget_w = self.video_label.winfo_width()
            widget_h = self.video_label.winfo_height()
            
            if widget_w > 0 and widget_h > 0:
                output_frame_resized = cv2.resize(output_frame, (widget_w, widget_h))
            else:
                output_frame_resized = output_frame

            img = cv2.cvtColor(output_frame_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_label.img_tk = img_tk
            self.video_label.configure(image=img_tk, text="")
            
            if self.is_global_anomaly:
                self.status_label.configure(text="¡ALERTA DE SOSPECHA DETECTADA!", text_color="red")
            else:
                self.status_label.configure(text="S.I.D.S.: Operando. Estado OK.", text_color="green")
        
        self.after(self.delay, self.update_video)

    def on_closing(self):
        """Cierra la cámara, termina el proceso Worker y la aplicación de forma segura."""
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
            
        # ¡CRÍTICO! Terminar el proceso secundario
        if self.worker_process.is_alive():
            self.worker_process.terminate()
            self.worker_process.join()
            
        print("\nSesión de análisis S.I.D.S. finalizada y recursos liberados.")
        self.destroy()

if __name__ == "__main__":
    # Necesario para que multiprocessing funcione correctamente
    mp.freeze_support() 
    app = SIDS_App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing) 
    app.mainloop()