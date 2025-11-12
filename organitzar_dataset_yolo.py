"""
Script per organitzar un dataset YOLO ja exportat des de Label Studio
Quan exportes en format YOLO des de Label Studio, ja tens:
  - images/ a la carpeta original de les imatges
  - labels/ amb els fitxers .txt
  - classes.txt amb les classes
  - notes.json amb metadata

Aquest script organitza aquestes dades en l'estructura necessària per YOLOv8
"""

import os
import shutil
import random
from pathlib import Path


def verificar_exportacio_yolo(export_dir, frames_dir=None):
    """
    Verifica que l'exportació de Label Studio en format YOLO sigui completa

    Args:
        export_dir: Directori d'exportació de Label Studio
        frames_dir: Directori amb les imatges originals (opcional)

    Returns:
        tuple: (bool: exportació vàlida, bool: necessita imatges externes)
    """
    print("=== Verificant exportació de Label Studio ===\n")

    export_path = Path(export_dir)

    # Verificar fitxers necessaris
    required_items = {
        'classes.txt': export_path / 'classes.txt',
        'labels': export_path / 'labels'
    }

    all_ok = True
    for name, path in required_items.items():
        if path.exists():
            if path.is_file():
                print(f"[OK] {name}: trobat")
            else:
                num_files = len(list(path.glob('*')))
                print(f"[OK] {name}/: {num_files} fitxers")
        else:
            print(f"[ERROR] {name}: no trobat")
            all_ok = False

    if not all_ok:
        print("\n[ERROR] L'exportacio no es completa")
        return False, False

    # Comprovar si hi ha imatges a la carpeta d'exportació
    images_path = export_path / 'images'
    labels_path = export_path / 'labels'

    image_files = []
    if images_path.exists():
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(list(images_path.glob(f'*{ext}')))

    label_files = list(labels_path.glob('*.txt'))

    print(f"\nTotal imatges a export: {len(image_files)}")
    print(f"Total etiquetes: {len(label_files)}")

    # Si no hi ha imatges a l'export, buscar a frames_dir
    necessita_imatges_externes = len(image_files) == 0

    if necessita_imatges_externes:
        print("\n[INFO] No s'han trobat imatges a l'exportació")
        if frames_dir and Path(frames_dir).exists():
            print(f"[INFO] Buscant imatges a: {frames_dir}")
            frames_path = Path(frames_dir)
            frame_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                frame_files.extend(list(frames_path.glob(f'*{ext}')))
            print(f"[INFO] Imatges trobades a frames: {len(frame_files)}")
        else:
            print(f"[WARNING] Cal especificar un directori frames/ amb les imatges originals")
    else:
        # Verificar correspondència imatge-etiqueta
        images_with_labels = 0
        for img_file in image_files:
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                images_with_labels += 1

        print(f"Imatges amb etiquetes: {images_with_labels}")

        if images_with_labels == 0:
            print("\n[WARNING] No s'han trobat imatges amb etiquetes corresponents")
            return False, False

    print("\n[OK] Exportacio valida")
    return True, necessita_imatges_externes


def extreure_nom_frame(label_filename):
    """
    Extreu el nom real del frame d'un fitxer d'etiqueta de Label Studio

    Els fitxers d'etiqueta tenen el format: UUID-frame_XXXXXX.txt
    Aquesta funció extreu: frame_XXXXXX

    Args:
        label_filename: Nom del fitxer d'etiqueta (ex: "35ebf3a4-frame_000001.txt")

    Returns:
        str: Nom del frame (ex: "frame_000001")
    """
    # Eliminar extensió .txt
    name_without_ext = label_filename.replace('.txt', '')

    # Buscar "frame_" al nom
    if 'frame_' in name_without_ext:
        # Extreure tot després de l'últim guió abans de "frame_"
        parts = name_without_ext.split('-')
        for i, part in enumerate(parts):
            if part.startswith('frame_'):
                # Agafar aquesta part i les següents (per si hi ha més guions)
                return '-'.join(parts[i:])

    # Si no trobem el patró, retornar el nom complet sense extensió
    return name_without_ext


def llegir_classes(classes_file):
    """
    Llegeix el fitxer classes.txt

    Args:
        classes_file: Camí al fitxer classes.txt

    Returns:
        list: Llista de noms de classes
    """
    classes = []
    with open(classes_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                classes.append(line)

    print(f"\nClasses detectades ({len(classes)}):")
    for idx, cls in enumerate(classes):
        print(f"  {idx}: {cls}")

    return classes


def organitzar_dataset(export_dir, output_dir, frames_dir=None, train_ratio=0.8, val_ratio=0.1):
    """
    Organitza el dataset en l'estructura necessària per YOLOv8

    Args:
        export_dir: Directori d'exportació de Label Studio
        output_dir: Directori de sortida
        frames_dir: Directori amb les imatges originals (opcional)
        train_ratio: Proporció d'entrenament (0.8 = 80%)
        val_ratio: Proporció de validació (0.1 = 10%)

    Returns:
        bool: True si l'organització ha tingut èxit
    """
    export_path = Path(export_dir)
    output_path = Path(output_dir)

    images_dir = export_path / 'images'
    labels_dir = export_path / 'labels'

    print(f"\n=== Organitzant dataset ===")
    print(f"Origen etiquetes: {export_dir}")
    if frames_dir:
        print(f"Origen imatges: {frames_dir}")
    print(f"Destinació: {output_dir}")

    # Crear estructura de directoris
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Determinar si cal buscar imatges a frames_dir
    usar_frames_dir = False
    if images_dir.exists():
        image_files_in_export = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files_in_export.extend(list(images_dir.glob(f'*{ext}')))
        usar_frames_dir = len(image_files_in_export) == 0
    else:
        usar_frames_dir = True

    if usar_frames_dir:
        if not frames_dir or not Path(frames_dir).exists():
            print("[ERROR] Cal especificar un directori frames/ amb les imatges originals")
            return False

        print("\n[INFO] Utilitzant imatges de:", frames_dir)
        frames_path = Path(frames_dir)

        # Crear diccionari de frames disponibles
        frames_disponibles = {}
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            for frame_file in frames_path.glob(f'*{ext}'):
                frames_disponibles[frame_file.stem] = frame_file

        print(f"[INFO] Frames disponibles: {len(frames_disponibles)}")

        # Obtenir llista d'etiquetes i buscar les imatges corresponents
        label_files = list(labels_dir.glob('*.txt'))
        image_label_pairs = []

        for label_file in label_files:
            # Extreure el nom real del frame
            frame_name = extreure_nom_frame(label_file.name)

            # Buscar la imatge corresponent
            if frame_name in frames_disponibles:
                image_label_pairs.append({
                    'image': frames_disponibles[frame_name],
                    'label': label_file,
                    'frame_name': frame_name
                })
            else:
                print(f"[WARNING] No s'ha trobat imatge per: {label_file.name} (buscat: {frame_name})")

        if not image_label_pairs:
            print("[ERROR] No s'han pogut emparellar imatges amb etiquetes")
            return False

        print(f"\n[OK] Emparellaments trobats: {len(image_label_pairs)}")

    else:
        # Cas original: imatges a la carpeta d'exportació
        print("\n[INFO] Utilitzant imatges de l'exportació")
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            for img_file in images_dir.glob(f'*{ext}'):
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    image_files.append(img_file)

        if not image_files:
            print("[ERROR] No s'han trobat imatges amb etiquetes corresponents")
            return False

        image_label_pairs = [
            {'image': img, 'label': labels_dir / f"{img.stem}.txt", 'frame_name': img.stem}
            for img in image_files
        ]

    print(f"\nTotal imatges amb etiquetes: {len(image_label_pairs)}")

    # Barrejar aleatòriament
    random.seed(42)  # Per reproductibilitat
    random.shuffle(image_label_pairs)

    # Calcular splits amb mínim d'1 imatge per conjunt
    total = len(image_label_pairs)

    if total < 3:
        print("\n[WARNING] Poques imatges! Es recomana tenir almenys 3 imatges.")
        print("Utilitzant totes per train...")
        train_pairs = image_label_pairs
        val_pairs = []
        test_pairs = []
    else:
        # Calcular splits amb mínim garantit
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        # Assegurar mínim d'1 imatge per val si hi ha prou imatges
        if val_size == 0 and total >= 3:
            val_size = 1
            train_size = total - val_size - 1  # Deixar 1 per test també

        # Assegurar que train no es quedi sense imatges
        if train_size == 0:
            train_size = 1

        train_pairs = image_label_pairs[:train_size]
        val_pairs = image_label_pairs[train_size:train_size + val_size]
        test_pairs = image_label_pairs[train_size + val_size:]

    # Copiar fitxers
    for split_name, split_pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        if not split_pairs:
            continue

        print(f"\nCopiant {len(split_pairs)} fitxers a {split_name}...")

        for pair in split_pairs:
            img_file = pair['image']
            label_file = pair['label']
            frame_name = pair['frame_name']

            # Copiar imatge amb el nom del frame (sense UUID)
            img_ext = img_file.suffix
            dst_img = output_path / 'images' / split_name / f"{frame_name}{img_ext}"
            shutil.copy2(img_file, dst_img)

            # Copiar etiqueta amb el nom del frame (sense UUID)
            dst_label = output_path / 'labels' / split_name / f"{frame_name}.txt"
            shutil.copy2(label_file, dst_label)

    print(f"\n[OK] Dataset organitzat:")
    total_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)
    if total_pairs > 0:
        print(f"  Train: {len(train_pairs)} ({len(train_pairs)/total_pairs*100:.1f}%)")
        print(f"  Val: {len(val_pairs)} ({len(val_pairs)/total_pairs*100:.1f}%)")
        print(f"  Test: {len(test_pairs)} ({len(test_pairs)/total_pairs*100:.1f}%)")
    else:
        print("  [ERROR] No s'han processat imatges")

    return True


def crear_data_yaml(output_dir, classes):
    """
    Crea el fitxer data.yaml necessari per YOLOv8

    Args:
        output_dir: Directori on crear data.yaml
        classes: Llista de noms de classes

    Returns:
        Path: Camí al fitxer data.yaml creat
    """
    output_path = Path(output_dir).absolute()

    yaml_content = f"""# Dataset per YOLOv8
# Generat automàticament des de Label Studio (exportació YOLO)

path: {output_path}  # directori arrel del dataset
train: images/train  # ruta relativa a 'path'
val: images/val      # ruta relativa a 'path'
test: images/test    # ruta relativa a 'path'

# Nombre de classes
nc: {len(classes)}

# Noms de les classes
names:
"""

    for idx, cls in enumerate(classes):
        yaml_content += f"  {idx}: {cls}\n"

    yaml_file = output_path / "data.yaml"
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n[OK] Fitxer data.yaml creat: {yaml_file}")
    return yaml_file


def mostrar_resum(output_dir):
    """
    Mostra un resum de l'estructura creada
    """
    output_path = Path(output_dir)

    print("\n" + "="*60)
    print("ORGANITZACIO COMPLETADA")
    print("="*60)

    print(f"\nEstructura creada a: {output_dir}/")
    print("|-- data.yaml          <- Configuracio per YOLOv8")
    print("|-- images/")
    print("|   |-- train/")

    # Comptar fitxers
    for split in ['train', 'val', 'test']:
        images_path = output_path / 'images' / split
        labels_path = output_path / 'labels' / split

        if images_path.exists():
            num_images = len(list(images_path.glob('*.*')))
            num_labels = len(list(labels_path.glob('*.txt')))
            print(f"|   |   +-- {num_images} imatges")

    print("|   |-- val/")
    print("|   +-- test/")
    print("+-- labels/")
    print("    |-- train/")
    print("    |-- val/")
    print("    +-- test/")

    print(f"\nSEGUENT PAS:")
    print(f"  python entrenar_yolo_v2.py")
    print()
    print("O directament amb YOLOv8:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('yolov8n.pt')")
    print(f"  model.train(data='{output_path}/data.yaml', epochs=100)")
    print()


def main():
    """
    Funció principal
    """
    print("="*60)
    print("ORGANITZACIO: Exportacio YOLO de Label Studio -> YOLOv8")
    print("="*60)
    print()

    # Configuració - MODIFICA AQUESTES RUTES SI CAL
    export_dir = "label_studio_export"  # Carpeta amb l'exportació YOLO de Label Studio
    output_dir = "yolo_dataset"          # Carpeta de sortida per YOLOv8
    frames_dir = "frames"                 # Carpeta amb les imatges originals (opcional)

    # Verificar exportació
    if not os.path.exists(export_dir):
        print(f"[ERROR] No s'ha trobat el directori: {export_dir}")
        print()
        print("INSTRUCCIONS:")
        print("1. Ves a Label Studio")
        print("2. Selecciona el teu projecte")
        print("3. Clica 'Export'")
        print("4. Selecciona format: 'YOLO'")
        print("5. Descarrega i descomprimeix a 'label_studio_export'")
        print()
        print("Estructura esperada:")
        print(f"  {export_dir}/")
        print("    |-- classes.txt")
        print("    |-- notes.json")
        print("    |-- labels/")
        print("    |     |-- UUID-frame_000001.txt")
        print("    |     +-- ...")
        print("    +-- images/ (opcional, pot estar buida)")
        print()
        print(f"Si images/ està buida, cal tenir:")
        print(f"  {frames_dir}/")
        print("    |-- frame_000000.jpg")
        print("    |-- frame_000001.jpg")
        print("    +-- ...")
        return

    # Verificar si frames_dir existeix
    frames_path = frames_dir if os.path.exists(frames_dir) else None

    # Verificar exportació
    valid, necessita_frames = verificar_exportacio_yolo(export_dir, frames_path)
    if not valid:
        print("\n[ERROR] L'exportacio de Label Studio no es valida")
        return

    # Si necessita frames però no existeix el directori
    if necessita_frames and not frames_path:
        print(f"\n[ERROR] No s'ha trobat el directori amb les imatges: {frames_dir}")
        print("Cal crear aquest directori amb les imatges originals")
        return

    # Llegir classes
    classes_file = os.path.join(export_dir, 'classes.txt')
    classes = llegir_classes(classes_file)

    if not classes:
        print("\n[ERROR] No s'han trobat classes al fitxer classes.txt")
        return

    # Confirmar abans de procedir
    print(f"\nAIXO creara/sobreescriura el directori: {output_dir}")
    try:
        resposta = input("Vols continuar? (s/n): ")
        if resposta.lower() != 's':
            print("Operació cancel·lada")
            return
    except EOFError:
        # Cas automàtic sense interacció
        print("Mode automàtic: continuant...")
        pass

    # Organitzar dataset
    success = organitzar_dataset(
        export_dir=export_dir,
        output_dir=output_dir,
        frames_dir=frames_path,
        train_ratio=0.8,
        val_ratio=0.1
    )

    if not success:
        print("\n[ERROR] No s'ha pogut organitzar el dataset")
        return

    # Crear data.yaml
    crear_data_yaml(output_dir, classes)

    # Mostrar resum
    mostrar_resum(output_dir)


if __name__ == "__main__":
    main()
