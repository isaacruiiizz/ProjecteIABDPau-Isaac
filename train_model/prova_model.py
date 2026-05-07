import os
from ultralytics import RTDETR

# 1. Ruta al teu model 'best.pt'
ruta_pesos = r"C:\Users\iruiz\OneDrive - Sa Palomera\IA&BD\C088 - Projecte\ProjecteIABDPau-Isaac\runs\detect\SmartChrono\rtdetr_local\weights\best.pt"

# 2. Ruta al vídeo o imatge que vols provar
ruta_test = r"C:\Users\iruiz\OneDrive - Sa Palomera\IA&BD\C088 - Projecte\Video\CortePartido.mp4" 

# Carreguem el model entrenat
model = RTDETR(ruta_pesos)

# Executem la detecció
results = model.predict(
    source=ruta_test,
    imgsz=480,       # Mateixa mida que l'entrenament
    conf=0.35,       # Filtre de confiança per evitar deteccions dubtoses
    save=True,       # Guarda el resultat visual
    device=0         # Fes servir la GPU
)

print(f"🎬 Vídeo processat! El resultat s'ha guardat a la carpeta: {results[0].save_dir}")