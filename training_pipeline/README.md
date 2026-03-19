# Workflow Complet: Label Studio + YOLOv8

Sistema complet per anotar vídeos amb Label Studio i entrenar models YOLOv8.

El procés transforma el video en frames i les anotacions es fan sobre els frames.
Si es vol fer sobre el video s'ha de preparar el fitxer mp4. Revisar aquesta documentació 
per veure com treballar-ho https://labelstud.io/tags/video#Video-format

## Prerequisits

```bash
# Python utilitzar versió 3.11.6
python3 --version

# Dependències principals
# pip install label-studio només si s'instal·la en local
pip install ultralytics
pip install opencv-python
pip install pillow
```

**Versions recomanades:**

- Label Studio: 1.10+ (https://github.com/HumanSignal/label-studio)
- Ultralytics (YOLOv8): 8.0+ (https://github.com/ultralytics/ultralytics)
- OpenCV: 4.8+
- Python: 3.11.6

## Workflow Complet

### Pas 1: Instal·lació de Label Studio en local no recomandat. Millor utilitzar el docker-compose i crear-ho com contenidor

```bash
chmod +x setup_label_studio.sh
./setup_label_studio.sh
```

```bash
docker compose up -d
```

### Pas 2: Iniciar Label Studio

Opció local

```bash
label-studio start
```

Opcions:

- Local : Obre el navegador a: http://localhost:8080
- Contenidor:obre el navegador a: http://localhost:7070
  
### Pas 3: Preparar les Dades

```bash
# Si tens un vídeo
python3 preparar_dades.py

# Això farà:
# 1. Extreure frames del vídeo i deixar-los a la carpeta /frames
# 2. Crear configuració per Label Studio
# 3. Generar fitxer JSON per importar /tasks/import_tasks.json
```

**Modificacions al script:**

- Canvia `video_exemple.mp4` pel nom del teu vídeo
- Ajusta `fps_target` per extreure més o menys frames
  - `fps_target=1`: 1 frame per segon
  - `fps_target=5`: 5 frames per segon

### Pas 4: Configurar Label Studio

1. **Crear un projecte nou:**
   - Ves a Label Studio (http://localhost:xxxx)
   - Clica "Create Project"
   - Posa un nom: "Detecció Objectes Vídeo"

2. **Configurar interfície d'anotació:**
   - Ves a Settings > Labeling Interface
   - Selecciona "Code"
   - Copia el contingut de `label_studio_config.xml` o crea una definició nova si ho consideres oportú
   - Modifica les etiquetes segons les teves necessitats. També ho pots incorporar pel label-studio i actualitzar el fitxer per guardar-t0ho i portar-ho a un altre equip.
     
     ```xml
     <Label value="Jugador Local" background="red"/>
     <Label value="Jugador Remot" background="blue"/>
     <!-- Afegeix més... -->
     ```

3. **Importar imatges:**
   - Ves a "Import"
   - Carrega `tasks/import_tasks.json`
   - Espera que es carreguin totes les imatges

### Pas 5: Anotar les Imatges

**Consells per anotar:**

- Dibuixa bounding boxes al voltant de cada objecte
- Intenta ser consistent amb les mides
- Inclou objectes parcials si són recognoscibles
- Si un objecte és dubtós, millor ometre'l

**Dreceres de teclat útils:**

- `Ctrl + Enter`: Guardar i següent
- `Ctrl + Z`: Desfer
- Números 1-9: Seleccionar etiqueta ràpidament

### Pas 6: Exportar Anotacions  en format YOLO (RECOMANAT ✓)

Quan hagis acabat d'anotar, tens **moltes opcions** per exportar, pero amb molt de volum, les exportacions amb imatges fallen per timeout. Per tant millor utilitzar yolo sense imatges:

1. Ves a "Export"
2. Selecciona format **YOLO**
3. Descarrega el ZIP
4. Descomprimeix a `label_studio_export/`

Obtindràs aquesta estructura:

```
label_studio_export/
├── classes.txt      ← Llista de classes
├── notes.json       ← Metadata
├── images/          ← Buida
└── labels/          ← Anotacions en format YOLO
    ├── frame_000000.txt
    └── ...
```

### Pas 7: Organitzar Dataset (si has exportat en YOLO)

```bash
python3 organitzar_dataset_yolo.py
```

**Què fa aquest script:**

- Verifica l'exportació de Label Studio
- Divideix el dataset en train/val/test (80%/10%/10%)
- Crea l'estructura necessària per YOLOv8
- Genera el fitxer `data.yaml`
- Com a mínim tingues 10 imatges amb anotacions

### Pas 9: Entrenar YOLOv8

```bash
python3 entrenar_yolo_v2.py
```

**Paràmetres ajustables:**

```python
entrenar_yolo(
    data_yaml="yolo_dataset/data.yaml",
    model_size='n',  # 'n', 's', 'm', 'l', 'x'
    epochs=100,      # Més èpoques = millor (normalment 100-300)
    imgsz=640,       # Mida imatge (640, 1280)
    batch=16         # Redueix si tens errors de memòria
)
```

**Mides de model:**
| Mida | Paràmetres | Velocitat | Precisió | Ús recomanat |
|------|-----------|-----------|----------|--------------|
| n    | 3M        | Molt ràpid | Bàsica  | Proves, temps real |
| s    | 11M       | Ràpid     | Bona     | Aplicacions generals |
| m    | 26M       | Moderat   | Molt bona | Equilibri |
| l    | 43M       | Lent      | Excel·lent | Alta precisió |
| x    | 68M       | Molt lent | Millor   | Màxima precisió |

### Pas 10: Avaluar Resultats

**Mètriques importants:**

- **mAP50**: Mean Average Precision al 50% IoU
- **mAP50-95**: mAP des del 50% al 95% IoU (més estricte)
- **Precision**: % de deteccions correctes
- **Recall**: % d'objectes detectats

**On trobar els resultats:**

```
runs/detect/yolo_custom/
├── weights/
│   ├── best.pt      # Millor model
│   └── last.pt      # Últim checkpoint
├── results.png      # Gràfiques d'entrenament
├── confusion_matrix.png
└── val_batch0_pred.jpg  # Prediccions en validació
```

### Pas 11: Fer Prediccions

```python
from ultralytics import YOLO

# Carregar model
model = YOLO('runs/detect/yolo_custom/weights/best.pt')

# Predir en imatge
results = model.predict('imatge.jpg', save=True)

# Predir en vídeo
results = model.predict('video.mp4', save=True)

# Predir en carpeta
results = model.predict('carpeta_imatges/', save=True)
```

### Pas 12: Exportar Model (Opcional)

Per usar el model en producció:

```python
from ultralytics import YOLO

model = YOLO('best.pt')

# Per servidor (CPU)
model.export(format='onnx')

# Per Android/iOS
model.export(format='tflite')

# Per Edge TPU
model.export(format='edgetpu')
```

### GPU no detectada

```bash
# Verificar PyTorch amb CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Instal·lar PyTorch amb CUDA (exemple per CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Label Studio no carrega imatges

- Verifica que les rutes a `import_tasks.json` siguin absolutes
- Assegura't que Label Studio tingui permisos de lectura
- Prova amb rutes relatives o carrega les imatges directament via web

### Model no aprèn (loss no baixa)

- Verifica que les anotacions siguin correctes
- Augmenta el nombre d'imatges anotades (mínim 100-200 per classe)
- Comprova que `data.yaml` apunti als directoris correctes
- Prova amb learning rate diferent: `lr0=0.001`

## 📚 Referències

- **Label Studio**: https://labelstud.io/guide/
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **YOLOv8 GitHub**: https://github.com/ultralytics/ultralytics
- **YOLO Format**: https://docs.ultralytics.com/datasets/detect/

## 🎓 Recursos Educatius

### Per estudiants:

1. Comença amb 20-30 imatges ben anotades
2. Entrena amb model 'n' (nano) per veure resultats ràpids
3. Experimenta amb diferents hiperparàmetres
4. Analitza les mètriques i gràfiques

### Millors pràctiques:

- **Qualitat > Quantitat**: Millor 100 imatges ben anotades que 500 de dolentes
- **Varietat**: Inclou diferents condicions (llum, angles, escales)
- **Consistència**: Mantén criteris d'anotació coherents
- **Validació**: Sempre separa un conjunt de validació

## Notes Finals

- Aquest workflow és escalable: funciona des de 50 fins a 50.000 imatges
- Label Studio permet col·laboració: múltiples anotadors poden treballar simultàniament
- YOLOv8 permet transfer learning: aprofita models preentrenats
- Els models generats es poden desplegar en producció (servidors, edge devices, mòbils)

## Llicències

- **Label Studio**: Apache License 2.0
- **Ultralytics YOLOv8**: AGPL-3.0 (lliure per ús educatiu/recerca)
  - Per ús comercial, consulta: https://ultralytics.com/license

---

**Autor**: Francesc Barragan  
**Data**: Novembre 2025  
**Versions**: Label Studio 1.10+, YOLOv8 8.0+

bash# Instal·lar el converter
pip install label-studio-converter

# Convertir JSON a YOLO
label-studio-converter export \
  -i export.json \
  -o ./yolo_output \
  -f YOLO \
  --image-dir ./imatges_originals
