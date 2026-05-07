import os
import torch
from ultralytics import RTDETR

def resume_training():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🚀 Reprenent entrenament a la GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ No s'ha detectat GPU.")
        return

    # 1. 🚨 Carreguem el fitxer last.pt que es va guardar a l'època 58
    ruta_last_pt = r"C:\Users\iruiz\OneDrive - Sa Palomera\IA&BD\C088 - Projecte\ProjecteIABDPau-Isaac\runs\detect\SmartChrono\rtdetr_local\weights\last.pt"
    
    model = RTDETR(ruta_last_pt)

    # 2. 🚨 Llançem l'entrenament indicant resume=True
    # Ultralytics ja recordarà el data.yaml, imgsz, batch i les èpoques restants!
    model.train(resume=True)

if __name__ == "__main__":
    resume_training()