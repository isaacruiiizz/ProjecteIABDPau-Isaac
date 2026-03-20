# Lògica IA

## 1 Lògica de Cronometratge i Control de Sessió

Aquest mòdul, executat pel `sc-logic-aggregator`, és l'encarregat de transformar les coordenades espacials en temps de joc efectiu. Atès que en categories de base el temps és "a rellotge corregut", el sistema es basa en la presència física dins l'àrea de joc.

**A. Input de l'Usuari (Control des del Frontend)**
Per evitar processar l'escalfament o el temps de descans, l'usuari interaccionarà amb el `sc-frontend` per definir els límits de la sessió:
* **Match Start/End:** Mitjançant el reproductor de vídeo, l'usuari marcarà el *timestamp* exacte de l'inici i la fi del partit. Aquests valors s'enviaran a l'API com a `start_frame` i `end_frame`.
* **Definició de Pista (ROI):** L'usuari dibuixarà sobre un frame de referència el polígon (4 punts) que delimita les línies de banda i fons. Només els jugadors dins d'aquest polígon sumaran minuts.

**B. Algoritme de Presència Efectiva**
El sistema analitzarà cada frame de forma seqüencial dins del rang definit:
1. **Detecció de Posició:** Es pren el punt mig de la base del *bounding box* del jugador com a referència de la seva posició a terra.
2. **Filtre Espacial:** Si el punt està dins del polígon de pista, el jugador es marca com a `IN_GAME`. Si està fora (banqueta) o no es detecta, es marca com a `OFF_COURT`.
3. **Suma de Temps:** Per cada frame on el jugador és `IN_GAME`, s'incrementa el seu comptador individual: 
   $$Temps\_Jugador = \sum \frac{1}{FPS_{video}}$$

**C. Gestió d'Oclusions i Robustesa (Smoothing)**
Per evitar que el cronòmetre s'aturi si un jugador queda tapat momentàniament per un altre o surt de pla uns segons, s'aplicarà una **Histeresi de Persistència**:
* **Buffer de Desaparició:** Si un jugador identificat desapareix o és marcat com `OFF_COURT`, el sistema esperarà un marge de **3 segons** (configurable) abans d'aturar el seu cronòmetre. 
* **Re-identificació:** Si el `Player_ID` reapareix dins de la pista abans d'exhaurir el buffer, el sistema omplirà el buit temporal automàticament, considerant que el jugador mai ha abandonat el terreny de joc.
* **Confirmació de Canvi:** Un jugador només deixarà de sumar minuts de forma definitiva quan sigui detectat fora del polígon o no aparegui en el tracking durant més de 3 segons consecutius.

## 2. Estratègia d'Entrenament dels Models d'IA

### 2.0 Condicions Reals de Gravació

Les imatges del pavelló revelen condicions específiques que condicionen directament l'estratègia d'entrenament:

| Condició | Observació | Impacte |
| :--- | :--- | :--- |
| **Angle de càmera** | Elevat des de la graderia, no cenital pur. Perspectiva en diagonal | Els jugadors al fons apareixen més petits que els del davant. El YOLO ha d'aprendre aquesta variació d'escala |
| **Distorsió òptica** | Lleu efecte ull de peix visible a les cantonades | Cal aplicar correcció de distorsió (undistort) com a pas de preprocessat abans de la inferència |
| **Backlight** | Finestres grans al fons generen contrallum fort | Els jugadors a la meitat del camp apareixen en semisombra. L'augmentació de dades ha d'incloure variacions de llum agressives |
| **Mida del dorsal** | Jugadors propers: dorsal llegible. Jugadors al fons: dorsal de 10–15px d'alçada | La CNN ha de ser robusta a dorsals de molt baixa resolució. `INPUT_SIZE` s'ha d'ajustar a la mida real dels crops |
| **Resolució de la càmera habitual** | Qualitat moderada (càmera d'acció) | La majoria del dataset serà d'aquesta qualitat — el model ha d'estar entrenat principalment sobre això |
| **Càmera professional** | 1 sol partit disponible, resolució superior | Útil com a conjunt de validació o per generar crops d'alta qualitat per a casos difícils |
 
**Conseqüència crítica:** el dataset d'entrenament ha de reflectir la qualitat real de la càmera habitual, **no** la càmera professional. Un model entrenat majoritàriament sobre imatges d'alta qualitat fallarà en producció.

### 2.1 Visió General
 
El sistema utilitza **dos models independents** amb responsabilitats i estratègies d'entrenament diferenciades:
 
| Model | Responsabilitat | Arquitectura | Input |
| :--- | :--- | :--- | :--- |
| **YOLO (detecció)** | Localitzar jugadors a la pista i generar bounding boxes | YOLOv8 (fine-tuned) | Frame complet |
| **CNN (dorsals)** | Llegir el número del dorsal a partir d'un crop del jugador | CNN personalitzada | Crop de la zona dorsal |
 
**Principi fonamental:** cap dels dos models s'entrena sobre la identitat dels jugadors ni sobre persones concretes. El YOLO detecta la classe `player` de forma genèrica, i la CNN llegeix un número d'1 a 99. Això fa el sistema transferible entre categories, temporades i altures de jugadors sense necessitat de re-entrenament complet.

### 2.2 Model 1 — YOLO: Detecció de Jugadors

#### Objectiu

Detectar tots els jugadors del nostre equip dins del frame i retornar els seus bounding boxes. **No cal distingir jugadors individuals** — això ho fa ByteTrack. Sí cal filtrar per equipació (el nostre equip vs. rival i àrbitres).

#### Estratègia: Fine-tuning sobre YOLOv8 preentrenat
 
No entrenant des de zero. YOLOv8 preentrenat sobre COCO ja sap detectar persones amb alta precisió. El fine-tuning serveix per especialitzar-lo en:
 
- **Angle cenital/picat** de càmera fixa al sostre — perspectiva molt diferent a les imatges de COCO.
- **Filtratge per equipació** — aprendre a distingir la nostra samarreta dels rivals i àrbitres pel color i patró.
- **Escala de jugadors** pròpia del nostre pavelló (resolució i distància constants).

#### Dataset per al YOLO

**Font principal:** vídeos gravats al club amb la càmera fixa.

Procés d'extracció i etiquetatge:
1. L'admin puja el vídeo a etiquetar des del frontend (secció d'etiquetatge) → `sc-api-gateway` el puja al bucket `labeling-videos` de MinIO.
2. El sistema llança `sc-video-manager` per trossejar el vídeo a intervals regulars (1 frame cada 2 segons = ~1.800 frames per hora) i els puja al bucket `labeling-frames` de MinIO.
3. **Label Studio** (`sc-label-studio`, servei opcional) llegeix els frames directament des del bucket `labeling-frames` via la integració S3/MinIO nativa. No cal moure ni descarregar res manualment.
4. L'etiquetador marca els bounding boxes amb dues classes: `player_own` (el nostre equip) i `other` (rivals, àrbitres).
5. Label Studio exporta el dataset en format YOLO directament al bucket `datasets` de MinIO, llest per ser consumit per `sc-active-learner`.

**Volum mínim recomanat:**
- 3.000–5.000 frames etiquetats per obtenir un model robust.
- Amb 1 frame cada 2 segons d'1 hora de vídeo → ~1.800 frames per partit. Amb 2-3 partits etiquetats ja es pot fer un primer entrenament viable.

**Preprocessat obligatori abans d'etiquetar i entrenar:**
- Correcció de distorsió d'ull de peix (`cv2.undistort`) amb els paràmetres de la càmera habitual. Calibrar usant les línies rectes de la pista com a referència.

**Augmentació de dades** (via Roboflow o Albumentations):
- Flip horitzontal (la pista és simètrica).
- Variacions de brillantor i contrast **agressives** — simular el backlight de les finestres del fons (jugadors en semisombra).
- Soroll gaussià i reducció de resolució deliberada — simular la qualitat real de la càmera habitual.
- Variació d'escala — jugadors propers (grans) i jugadors al fons (petits a causa de la perspectiva).
- **No** rotar ni fer flip vertical — la càmera és fixa i l'angle sempre és el mateix.

#### Configuració d'entrenament
 
```yaml
# yolo_finetune_config.yaml
model: yolov8m.pt          # Base preentrenada (medium — bon balanç velocitat/precisió)
data: dataset/yolo/data.yaml
epochs: 50
imgsz: 1280                # Resolució alta per càmera fixa cenital
batch: 16
lr0: 0.001
freeze: 10                 # Congela les primeres 10 capes (extractor de features COCO)
classes: 2                 # player_own, other
device: 0                  # GPU
```

**Mètriques d'acceptació:** mAP@0.5 > 0.85 sobre el conjunt de validació.

### 2.3 Model 2 — CNN: Reconeixement de Dorsals

#### Objectiu
Donat un crop de la zona del dorsal d'un jugador, retornar el número (1–99) amb una puntuació de confiança. Si la confiança és inferior a `INFERENCE_CONFIDENCE_THRESHOLD=0.6`, el resultat es descarta i el crop s'afegeix a `feedback-data` per al re-entrenament.

#### Per què una CNN pròpia i no OCR genèric?

L'OCR genèric (Tesseract, EasyOCR) falla en aquest domini per diverses raons: el dorsal apareix en moviment i parcialment desenfocats, la font és específica de l'equipació del club, hi ha oclusió parcial freqüent, i l'angle de la càmera fixa genera distorsió de perspectiva. Una CNN entrenada específicament sobre crops del nostre club aprèn exactament aquestes condicions.

#### Estratègia: classificació de 99 classes (1–99)

El problema es tracta com una **classificació multiclasse** (99 classes) i no com a OCR seqüencial. Això simplifica molt l'arquitectura i és suficient per al rang 1–99.

**Arquitectura base recomanada:** MobileNetV3-Small o EfficientNet-B0 (lleugers, ràpids, bons per a imatges petites).

#### Dataset per a la CNN

**Problema principal:** els crops de dorsals de vídeos reals son petits (~64×64px), moguts i parcialment tapats. Cal un dataset gran i variat.

**Estratègia en 3 capes:**
 
**Capa 1 — Dades sintètiques (punt de partida ràpid):**
Generar imatges sintètiques de dorsals amb la mateixa font, colors i estil de l'equipació del club. Script Python amb PIL/Pillow:
- Fons del color de la samarreta del club.
- Números 1–99 amb la font real de l'equipació.
- Augmentació: rotació ±15°, perspectiva, soroll, desenfoc de moviment, oclusió parcial aleatòria.
- Generar 500–1.000 imatges per classe = 50.000–100.000 imatges sintètiques totals.

Això permet tenir un model base funcional **sense necessitat d'etiquetar res manualment** al principi.

**Capa 2 — Crops reals dels vídeos del club (millora de qualitat):**
Extreure crops reals de dorsals dels vídeos existents usant el YOLO ja entrenat. Etiquetar el número de dorsal de cada crop.
- Objectiu: 200–500 crops reals per dorsal actiu (els dorsals que realment fan servir els jugadors del club).
- No cal cobrir tots els números 1–99 amb dades reals — els sintètics cobreixen la cua llarga.

**Capa 3 — Active Learning continu (millora automàtica):**
Els crops amb confiança < 0.6 s'acumulen a `feedback-data` de MinIO. Quan `TRAINING_MIN_SAMPLES=50` nous crops estan disponibles, `sc-active-learner` llança un fine-tuning automàtic i genera una nova versió del model. Vegeu punt 2.9 i 2.3 Fase D.

#### Configuració d'entrenament
 
```python
# cnn_training_config.py
BASE_MODEL = "efficientnet_b0"  # Preentrenat ImageNet
NUM_CLASSES = 99                # Dorsals 1–99
INPUT_SIZE = (48, 48)           # Ajustat a la mida real dels crops (dorsal ~10-15px al fons)
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.0005
FREEZE_BACKBONE = True          # Primera fase: entrenar només el cap classificador
UNFREEZE_AFTER_EPOCH = 15       # Segona fase: fine-tuning complet
DROPOUT = 0.3
```

**Nota sobre la càmera professional:** el partit gravat amb la càmera professional (alta resolució) s'usa exclusivament com a **conjunt de validació** — mai per entrenar. Permet mesurar el límit superior de precisió del model en condicions ideals i detectar si el model generalitza bé o sobreajusta a la qualitat baixa.

**Mètriques d'acceptació:** accuracy Top-1 > 0.80 i Top-3 > 0.92 sobre el conjunt de validació de la càmera habitual. S'accepta un llindar lleugerament inferior al teòric donades les condicions reals de llum i resolució.

### 2.4 Pipeline d'Etiquetatge Recomanat
 
Per aprofitar els vídeos existents del club de forma eficient:
 
```
Vídeos del club
      │
      ▼
Extracció de frames (1 frame / 2s)
      │
      ▼
Etiquetatge YOLO (Roboflow)        ← ~2-3 dies de feina manual
      │
      ▼
Entrenament YOLO v1
      │
      ▼
YOLO detecta jugadors automàticament en nous vídeos
      │
      ▼
Extracció automàtica de crops de dorsals
      │
      ▼
Etiquetatge CNN (només el número)  ← molt més ràpid, crops petits
      │
      ▼
Entrenament CNN v1 (sintètic + real)
      │
      ▼
Sistema en producció → Active Learning automàtic
```

### 2.5 Gestió de Versions dels Models
 
Els models entrenats es guarden al bucket `models` de MinIO amb versionat incremental (vegeu punt 2.9). El model actiu és sempre el de versió més alta disponible.
 
**Política de promoció:** un nou model generat per `sc-active-learner` no substitueix l'actiu automàticament. Primer es valida sobre un conjunt de test fix (`models/eval/test_set/`) i només es promou si supera les mètriques d'acceptació definides als punts 4.2 i 4.3. Si no les supera, es guarda igualment com a versió candidata per a revisió manual.
 
```
models/
├── yolo/
│   ├── weights/
│   │   ├── v1.pt       ← actiu
│   │   ├── v2.pt       ← candidat pendent de validació
│   └── eval/
│       └── test_set/   ← conjunt de validació fix (mai s'usa per entrenar)
├── cnn/
│   ├── weights/
│   │   ├── v1.keras    ← actiu
│   │   └── v2.keras
│   └── eval/
│       └── test_set/
```

### 2.6 Consideracions de Privacitat
 
Els vídeos de partits contenen imatges de menors d'edat (categories de base). Cal tenir en compte:
 
- Els vídeos d'entrenament **mai surten del servidor local** del club — no es pugen a serveis externs com Roboflow Cloud ni Google Colab.
- L'etiquetatge es fa en local amb eines auto-hostatjades (Label Studio).
- El dataset final (crops de dorsals) no conté cares ni és identificable — conté únicament retalls de samarretes amb números.