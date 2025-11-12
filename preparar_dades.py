"""
Script per extreure frames d'un vídeo i preparar-los per Label Studio
Requisits: opencv-python
Versió OpenCV recomanada: 4.8+
"""

import cv2
import os
import json
from pathlib import Path


def extraure_frames_video(video_path, output_dir, fps_target=1):
    """
    Extreu frames d'un vídeo a intervals regulars
    
    Args:
        video_path: Camí al fitxer de vídeo
        output_dir: Directori on guardar els frames
        fps_target: Nombre de frames per segon a extreure (1 = 1 frame cada segon)
    """
    # Crear directori de sortida si no existeix
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Obrir el vídeo
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: No es pot obrir el vídeo {video_path}")
        return
    
    # Obtenir informació del vídeo
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracio = total_frames / fps_original
    
    print(f"Vídeo: {video_path}")
    print(f"FPS original: {fps_original:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duració: {duracio:.2f} segons")
    print(f"Extraient {fps_target} frame(s) per segon...")
    
    # Calcular interval entre frames a extreure
    interval = int(fps_original / fps_target)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Guardar frame si correspon
        if frame_count % interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"\n✓ {saved_count} frames extrets i guardats a {output_dir}")
    return saved_count


def crear_config_label_studio(project_name="Detecció d'Objectes YOLO"):
    """
    Crea la configuració XML per Label Studio amb suport per YOLO
    Aquesta configuració permet fer bounding boxes amb etiquetes
    """
    config = '''<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="persona" background="red"/>
    <Label value="cotxe" background="blue"/>
    <Label value="moto" background="green"/>
    <Label value="bicicleta" background="yellow"/>
    <Label value="camió" background="orange"/>
    <!-- Afegeix més etiquetes segons les teves necessitats -->
  </RectangleLabels>
</View>'''
    
    print("\n=== Configuració per Label Studio ===")
    print("Copia aquesta configuració a Label Studio:")
    print("Settings > Labeling Interface > Code")
    print("\n" + config)
    
    # Guardar també a fitxer
    with open('/home/claude/label_studio_config.xml', 'w') as f:
        f.write(config)
    print("\n✓ Configuració guardada a: label_studio_config.xml")


def crear_import_json(images_dir, output_json="import_tasks.json"):
    """
    Crea un fitxer JSON per importar imatges a Label Studio
    
    Args:
        images_dir: Directori amb les imatges
        output_json: Nom del fitxer JSON de sortida
    """
    images_path = Path(images_dir)
    
    if not images_path.exists():
        print(f"Error: El directori {images_dir} no existeix")
        return
    
    # Crear llista de tasques
    tasks = []
    
    # Buscar imatges
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(images_path.glob(f'*{ext}'))
        image_files.extend(images_path.glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    
    for img_file in image_files:
        # Ruta relativa o absoluta segons preferència
        # Per Label Studio local, millor usar ruta absoluta
        task = {
            "data": {
                "image": str(img_file.absolute())
            }
        }
        tasks.append(task)
    
    # Guardar JSON
    output_path = Path(output_json)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Fitxer d'importació creat: {output_json}")
    print(f"  Total imatges: {len(tasks)}")
    print(f"\nPer importar a Label Studio:")
    print(f"  1. Obre el projecte a Label Studio")
    print(f"  2. Ves a 'Import' i carrega {output_json}")
    
    return output_path


if __name__ == "__main__":
    # Exemple d'ús
    print("=== Preparació de dades per Label Studio ===\n")
    
    # 1. Extreure frames d'un vídeo (si tens un vídeo)
    video_exemple = "video_exemple.mp4"
    output_frames = "./frames"
    
    if os.path.exists(video_exemple):
        extraure_frames_video(video_exemple, output_frames, fps_target=1)
    else:
        print(f"ℹ️  No s'ha trobat {video_exemple}")
        print(f"   Col·loca el teu vídeo amb aquest nom o modifica la ruta\n")
        
        # Crear directori de mostra per l'exemple
        Path(output_frames).mkdir(parents=True, exist_ok=True)
        print(f"   Directori creat: {output_frames}")
        print(f"   Col·loca les teves imatges aquí\n")
    
    # 2. Crear configuració per Label Studio
    crear_config_label_studio()
    
    # 3. Crear fitxer JSON per importar (si hi ha imatges)
    image_files = list(Path(output_frames).glob('*.jpg')) + list(Path(output_frames).glob('*.png'))
    
    if image_files:
        crear_import_json(output_frames, "import_tasks.json")
    else:
        print("\nℹ️  Col·loca imatges a", output_frames, "i torna a executar per crear l'import JSON")
    
    print("\n=== Passos següents ===")
    print("1. Inicia Label Studio: label-studio start")
    print("2. Crea un nou projecte")
    print("3. Configura la interfície amb label_studio_config.xml")
    print("4. Importa les imatges amb import_tasks.json")
    print("5. Comença a anotar!")
