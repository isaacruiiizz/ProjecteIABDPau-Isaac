#!/usr/bin/env python3
"""
Script de verificació i test del sistema Label Studio + YOLOv8
Comprova que tot estigui instal·lat correctament
"""

import sys
import subprocess

def verificar_modul(modul, nom_paquet=None):
    """Verifica si un mòdul Python està instal·lat"""
    if nom_paquet is None:
        nom_paquet = modul
    
    try:
        __import__(modul)
        print(f"✓ {nom_paquet} instal·lat")
        return True
    except ImportError:
        print(f"✗ {nom_paquet} NO instal·lat")
        print(f"  Instal·la amb: pip install {nom_paquet}")
        return False

def verificar_versio(modul, nom_paquet=None):
    """Mostra la versió d'un paquet"""
    if nom_paquet is None:
        nom_paquet = modul
    
    try:
        mod = __import__(modul)
        if hasattr(mod, '__version__'):
            print(f"  Versió: {mod.__version__}")
        return True
    except:
        return False

def verificar_label_studio():
    """Verifica Label Studio"""
    try:
        result = subprocess.run(['label-studio', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Label Studio instal·lat")
            print(f"  {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        print("✗ Label Studio NO instal·lat")
        print("  Instal·la amb: pip install label-studio")
        return False

def verificar_pytorch():
    """Verifica PyTorch i disponibilitat de GPU"""
    try:
        import torch
        print(f"✓ PyTorch instal·lat")
        print(f"  Versió: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✓ GPU disponible: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA versió: {torch.version.cuda}")
        else:
            print(f"  ℹ️  GPU NO disponible (s'usarà CPU)")
            print(f"     Per GPU, instal·la PyTorch amb CUDA: https://pytorch.org")
        return True
    except ImportError:
        print("✗ PyTorch NO instal·lat")
        print("  Instal·la amb: pip install torch torchvision")
        return False

def crear_estructura_exemple():
    """Crea estructura de directoris d'exemple"""
    import os
    from pathlib import Path
    
    directoris = [
        'frames_video',
        'yolo_dataset/images/train',
        'yolo_dataset/images/val',
        'yolo_dataset/images/test',
        'yolo_dataset/labels/train',
        'yolo_dataset/labels/val',
        'yolo_dataset/labels/test',
    ]
    
    print("\n=== Creant estructura de directoris ===")
    for dir in directoris:
        path = Path(dir)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir}")
    
    print("\n✓ Estructura creada")

def crear_exemple_imatge():
    """Crea una imatge d'exemple per testing"""
    try:
        import cv2
        import numpy as np
        
        # Crear imatge de test amb formes de colors
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Dibuixar alguns objectes
        cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # Blau
        cv2.circle(img, (400, 200), 50, (0, 255, 0), -1)  # Verd
        cv2.rectangle(img, (300, 300), (450, 400), (0, 0, 255), -1)  # Vermell
        
        # Afegir text
        cv2.putText(img, "Imatge de test", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Guardar
        output_path = "frames_video/test_image.jpg"
        cv2.imwrite(output_path, img)
        print(f"\n✓ Imatge de test creada: {output_path}")
        return True
    except Exception as e:
        print(f"\n✗ No s'ha pogut crear imatge de test: {e}")
        return False

def mostrar_resum():
    """Mostra un resum dels scripts disponibles"""
    print("\n" + "="*60)
    print("SCRIPTS DISPONIBLES")
    print("="*60)
    
    scripts = [
        ("setup_label_studio.sh", "Instal·lar Label Studio"),
        ("preparar_dades.py", "Extreure frames i preparar dades"),
        ("convertir_a_yolo.py", "Convertir anotacions a format YOLO"),
        ("entrenar_yolo.py", "Entrenar model YOLOv8"),
        ("README.md", "Documentació completa"),
    ]
    
    for script, descripcio in scripts:
        print(f"\n📄 {script}")
        print(f"   {descripcio}")
    
    print("\n" + "="*60)
    print("WORKFLOW RECOMANAT")
    print("="*60)
    print("""
1. Executar: bash setup_label_studio.sh
2. Executar: python3 preparar_dades.py
3. Iniciar Label Studio: label-studio start
4. Anotar imatges a la interfície web
5. Exportar anotacions des de Label Studio
6. Executar: python3 convertir_a_yolo.py
7. Executar: python3 entrenar_yolo.py
    """)

def main():
    print("="*60)
    print("VERIFICACIÓ DEL SISTEMA - Label Studio + YOLOv8")
    print("="*60)
    
    print("\n=== Verificant dependències Python ===\n")
    
    tot_ok = True
    
    # Verificar mòduls essencials
    moduls = [
        ('cv2', 'opencv-python'),
        ('PIL', 'pillow'),
        ('ultralytics', 'ultralytics'),
    ]
    
    for modul, paquet in moduls:
        if verificar_modul(modul, paquet):
            verificar_versio(modul, paquet)
        else:
            tot_ok = False
    
    print("\n=== Verificant Label Studio ===\n")
    if not verificar_label_studio():
        tot_ok = False
    
    print("\n=== Verificant PyTorch ===\n")
    if not verificar_pytorch():
        tot_ok = False
    
    print("\n" + "="*60)
    if tot_ok:
        print("✓ TOT CORRECTE - Sistema preparat!")
        print("="*60)
        
        # Crear estructura
        crear_estructura_exemple()
        crear_exemple_imatge()
        
        mostrar_resum()
        
        print("\n✓ Sistema verificat i preparat per començar!")
    else:
        print("✗ CAL INSTAL·LAR DEPENDÈNCIES")
        print("="*60)
        print("\nExecuta:")
        print("  pip install ultralytics opencv-python pillow label-studio")
    
    return 0 if tot_ok else 1

if __name__ == "__main__":
    sys.exit(main())
