"""
Script per entrenar YOLOv8 amb les dades exportades de Label Studio
Requisits: ultralytics>=8.0.0
Documentació: https://docs.ultralytics.com/
"""

from ultralytics import YOLO
import os
from pathlib import Path


def verificar_dataset(data_yaml):
    """
    Verifica que el dataset estigui ben organitzat
    
    Args:
        data_yaml: Camí al fitxer data.yaml
    
    Returns:
        bool: True si el dataset és vàlid
    """
    print("=== Verificant Dataset ===\n")
    
    if not os.path.exists(data_yaml):
        print(f"❌ No es troba {data_yaml}")
        print("\n📋 Has executat el script de conversió?")
        print("   python convertir_label_studio_export.py")
        return False
    
    # Llegir data.yaml
    import yaml
    try:
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
    except ImportError:
        print("⚠️  PyYAML no instal·lat, verificació manual...")
        # Continuar igualment
        return True
    except Exception as e:
        print(f"⚠️  Error llegint data.yaml: {e}")
        return True
    
    # Verificar paths
    base_path = Path(data['path'])
    
    checks = {
        'train': base_path / data['train'],
        'val': base_path / data['val'],
        'test': base_path / data.get('test', 'images/test')
    }
    
    print("Verificant directoris:")
    all_ok = True
    for name, path in checks.items():
        if path.exists():
            num_images = len(list(path.glob('*.jpg'))) + len(list(path.glob('*.png')))
            print(f"  ✓ {name}: {num_images} imatges")
            if num_images == 0:
                print(f"    ⚠️  Sense imatges!")
                all_ok = False
        else:
            print(f"  ✗ {name}: no existeix")
            all_ok = False
    
    # Verificar labels
    print("\nVerificant etiquetes:")
    labels_base = base_path / 'labels'
    for name in ['train', 'val', 'test']:
        labels_path = labels_base / name
        if labels_path.exists():
            num_labels = len(list(labels_path.glob('*.txt')))
            print(f"  ✓ {name}: {num_labels} fitxers .txt")
        else:
            print(f"  ✗ {name}: no existeix")
    
    # Mostrar classes
    if 'names' in data:
        print(f"\nClasses detectades ({len(data['names'])}):")
        for idx, name in data['names'].items():
            print(f"  {idx}: {name}")
    
    print()
    return all_ok


def entrenar_yolo(data_yaml, model_size='n', epochs=100, imgsz=640, batch=16, device='cpu'):
    """
    Entrena un model YOLOv8 amb les dades preparades
    
    Args:
        data_yaml: Camí al fitxer data.yaml
        model_size: Mida del model ('n', 's', 'm', 'l', 'x')
        epochs: Nombre d'èpoques d'entrenament
        imgsz: Mida de les imatges (640 és l'estàndard)
        batch: Mida del batch (ajusta segons la teva GPU/RAM)
        device: 'cpu', 'cuda', '0', '1', etc.
    """
    
    print("=== Entrenament YOLOv8 ===\n")
    
    # Verificar dataset
    if not verificar_dataset(data_yaml):
        print("\n❌ El dataset té problemes. Revisa'l abans de continuar.")
        resposta = input("Vols continuar igualment? (s/n): ")
        if resposta.lower() != 's':
            return None
    
    # Carregar model preentrenat
    model_name = f"yolov8{model_size}.pt"
    print(f"Carregant model: {model_name}")
    print(f"\nMides disponibles:")
    print(f"  - yolov8n.pt: Nano (més ràpid, ~3M paràmetres)")
    print(f"  - yolov8s.pt: Small (~11M paràmetres)")
    print(f"  - yolov8m.pt: Medium (~26M paràmetres)")
    print(f"  - yolov8l.pt: Large (~43M paràmetres)")
    print(f"  - yolov8x.pt: XLarge (més precís, ~68M paràmetres)\n")
    
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"❌ Error carregant model: {e}")
        print(f"El model es descarregarà automàticament en el primer ús")
        model = YOLO(model_name)
    
    # Configurar entrenament
    print(f"Configuració d'entrenament:")
    print(f"  - Èpoques: {epochs}")
    print(f"  - Mida imatge: {imgsz}")
    print(f"  - Batch size: {batch}")
    print(f"  - Device: {device}")
    print(f"  - Dataset: {data_yaml}\n")
    
    # Ajustar batch size automàticament si és necessari
    if device == 'cpu' and batch > 8:
        print(f"⚠️  Usant CPU amb batch={batch} pot ser lent")
        print(f"   Recomanació: batch=4-8 per CPU\n")
    
    # Entrenar
    print("Iniciant entrenament...")
    print("(Això pot trigar força estona, especialment en CPU)\n")
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project='runs/detect',
            name='label_studio_yolo',
            patience=50,  # Early stopping
            save=True,
            plots=True,
            verbose=True,
            device=device,
            # Paràmetres addicionals útils
            save_period=10,  # Guardar checkpoint cada 10 èpoques
            workers=2 if device == 'cpu' else 8,
            amp=False if device == 'cpu' else True  # Mixed precision (només GPU)
        )
        
        print("\n✓ Entrenament completat!")
        print(f"✓ Model guardat a: runs/detect/label_studio_yolo/weights/best.pt")
        print(f"✓ Resultats a: runs/detect/label_studio_yolo/")
        
        return model
        
    except Exception as e:
        print(f"\n❌ Error durant l'entrenament: {e}")
        print("\nConsells:")
        print("  - Redueix batch size si tens errors de memòria")
        print("  - Verifica que data.yaml apunta als directoris correctes")
        print("  - Assegura't que tens imatges i etiquetes corresponents")
        return None


def validar_model(model_path, data_yaml):
    """
    Valida el model entrenat
    """
    print("\n=== Validació del Model ===\n")
    
    if not os.path.exists(model_path):
        print(f"❌ No es troba el model: {model_path}")
        return None
    
    model = YOLO(model_path)
    
    # Validar
    metrics = model.val(data=data_yaml)
    
    print("\nMètriques de validació:")
    print(f"  - mAP50: {metrics.box.map50:.4f}")
    print(f"  - mAP50-95: {metrics.box.map:.4f}")
    print(f"  - Precision: {metrics.box.mp:.4f}")
    print(f"  - Recall: {metrics.box.mr:.4f}")
    
    return metrics


def fer_prediccions(model_path, source, conf=0.25):
    """
    Fa prediccions amb el model entrenat
    """
    print(f"\n=== Prediccions ===\n")
    
    if not os.path.exists(model_path):
        print(f"❌ No es troba el model: {model_path}")
        return None
    
    if not os.path.exists(source):
        print(f"❌ No es troba l'origen: {source}")
        return None
    
    model = YOLO(model_path)
    
    # Fer prediccions
    results = model.predict(
        source=source,
        save=True,
        project='runs/detect',
        name='predictions',
        conf=conf,
        iou=0.45,
        show_labels=True,
        show_conf=True,
        line_width=2
    )
    
    print(f"✓ Prediccions guardades a: runs/detect/predictions")
    
    return results


def exemple_complet():
    """
    Exemple complet del workflow d'entrenament
    """
    print("="*60)
    print("ENTRENAMENT YOLOV8 AMB DADES DE LABEL STUDIO")
    print("="*60)
    print()
    
    # Configuració per defecte
    data_yaml = "yolo_dataset/data.yaml"
    
    # Verificar que existeix data.yaml
    if not os.path.exists(data_yaml):
        print(f"❌ No es troba {data_yaml}")
        print()
        print("📋 WORKFLOW COMPLET:")
        print()
        print("1️⃣  Exporta des de Label Studio:")
        print("   - Ves al teu projecte")
        print("   - Clica 'Export'")
        print("   - Selecciona format JSON")
        print("   - Descarrega i descomprimeix")
        print()
        print("2️⃣  Converteix a format YOLO:")
        print("   python convertir_label_studio_export.py")
        print()
        print("3️⃣  Entrena el model:")
        print("   python entrenar_yolo.py")
        print()
        return
    
    # Preguntar configuració
    print("Configuració de l'entrenament:\n")
    
    # Model
    print("Mida del model (n=nano, s=small, m=medium, l=large, x=xlarge):")
    model_size = input("  [n]: ").strip() or 'n'
    
    # Èpoques
    print("\nNombre d'èpoques (recomanat: 50-100 per proves, 100-300 per resultats finals):")
    epochs_input = input("  [50]: ").strip()
    epochs = int(epochs_input) if epochs_input else 50
    
    # Batch
    print("\nBatch size (redueix si tens errors de memòria):")
    batch_input = input("  [16]: ").strip()
    batch = int(batch_input) if batch_input else 16
    
    # Device
    print("\nDevice (cpu, cuda, 0, 1, etc.):")
    device = input("  [cpu]: ").strip() or 'cpu'
    
    print("\n" + "="*60)
    print("Iniciant entrenament...")
    print("="*60)
    
    # Entrenar
    model = entrenar_yolo(
        data_yaml=data_yaml,
        model_size=model_size,
        epochs=epochs,
        batch=batch,
        device=device
    )
    
    if model is None:
        return
    
    # Validar
    best_model = "runs/detect/label_studio_yolo/weights/best.pt"
    if os.path.exists(best_model):
        validar_model(best_model, data_yaml)
        
        # Oferir fer prediccions
        print("\n" + "="*60)
        print("Vols fer prediccions amb el model entrenat? (s/n): ", end='')
        if input().lower() == 's':
            print("\nIntrodueix el camí a la imatge, vídeo o carpeta:")
            source = input("  ").strip()
            if source and os.path.exists(source):
                fer_prediccions(best_model, source)
    
    print("\n" + "="*60)
    print("✓ PROCÉS COMPLETAT")
    print("="*60)
    print(f"\nFitxers generats:")
    print(f"  - Model entrenat: {best_model}")
    print(f"  - Mètriques: runs/detect/label_studio_yolo/results.png")
    print(f"  - Gràfiques: runs/detect/label_studio_yolo/")
    print()


if __name__ == "__main__":
    # Verificar que ultralytics està instal·lat
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Cal instal·lar ultralytics:")
        print("   pip install ultralytics")
        exit(1)
    
    # Executar exemple interactiu
    exemple_complet()
