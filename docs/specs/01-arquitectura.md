# Arquitectura tècnica

El sistema SmartChrono IP adopta una arquitectura de microserveis mitjançant contenidors Docker completament desacoblats mitjançant un patró de Cua de Tasques (Producer-Consumer). Això permet que el processament intensiu de vídeo no bloquegi l'API i facilita el re-entrenament del model en paral·lel.

Utilitzarem les ultimes versions suportades de tot, i si hi han components/llibreries també han de ser les últimes versions sense que hi hagin problemes de seguretat.

## 1 Stack Tecnològic

* **Backend API:** FastAPI (Python 3.11).
* **AI Engine:** Ultralytics (YOLOv8) + TensorFlow (CNN Dorsals).
* **Message Broker:** Redis (Pub/Sub & List Queues).
* **Database:** MongoDB 6.0+ (NoSQL).
* **Frontend:** React (Vite) + Tailwind CSS.
* **Retraining Engine:** PyTorch/TensorFlow (entrenament asíncron).
* **Video Processing:** FFmpeg + OpenCV.
* **Observabilitat:**
  * **Dozzle:** Visualització de logs de tots els contenidors en temps real.
  * **Sentry:** Rastreig d'errors crítics en producció (SDK integrat en serveis Python).
  * **Prometheus:** Recollida de mètriques (CPU, GPU VRAM, temps de processament per frame).
  * **Grafana:** Dashboard visual de mètriques i estat del sistema.

## 1.1 Definició de Contenidors (Docker Services)

| Servei | Responsabilitat Tècnica | Stack Intern |
| :--- | :--- | :--- |
| **`sc-api-gateway`** | **Punt d'entrada REST**. Gestiona el CRUD de MongoDB, l'autenticació i l'enviament de "Jobs" de processament a Redis. | FastAPI, Uvicorn, Motor (MongoDB Driver) |
| **`sc-video-manager`** | **Ingestió i Muntatge:** Talla el vídeo original en frames (.jpg) a la carpeta compartida /frames. Un cop finalitzat el procés, re-munta un nou vídeo sobreposant els bounding boxes i IDs. | FFmpeg, OpenCV, PyAV |
| **`sc-object-storage`**| **S3 Storage:** Servidor d'objectes per guardar frames, vídeos i models de forma centralitzada. | MinIO Server (S3 API) |
| **`sc-inference-worker`** | **Inferència d'IA:** Escolta la cua `pending_frames`. Executa YOLOv8 (jugadors) i la CNN personalitzada (dorsals) utilitzant la GPU (8GB VRAM). | PyTorch, CUDA 12.6, TensorFlow/Keras, Ultralytics |
| **`sc-logic-aggregator`** | **Tracking i Temps:** Rep els JSON de detecció de l'inference-worker. Implementa la lògica de ByteTrack per mantenir la identitat i calcula el temps de permanència a "zona activa". | FilterPy, NumPy, Pandas (Time-series) |
| **`sc-active-learner`** | **Feedback Loop:** Automatitza el fine-tuning dels models. Detecta noves dades a `/feedback` i entrena noves versions dels pesos (.pt / .keras). | Ultralytics Trainer, Scikit-learn |
| **`sc-frontend`** | **Interfície Gràfica:** Mostra per insertar el video, mostra el reproductor de vídeo i el llistat de jugadors. | React 18, Vite, Tailwind CSS, Lucide Icons |
| **`sc-dozzle`** | **Log Viewer:** Interfície web per monitoritzar logs dels contenidors en temps real. | Dozzle Runtime |
| **`sc-prometheus`** | **Metrics Scraper:** Recull mètriques d'ús de recursos i rendiment de l'IA (frames/segon). | Prometheus Server |
| **`sc-grafana`** | **Visualization:** Panells de control per visualitzar mètriques de Prometheus. | Grafana Labs |
| **`sc-redis`** | **Broker & State:** Gestió de cues de tasques (RQ/Celery style) i emmagatzematge temporal de l'estat del processament. | Redis Streams, Pub/Sub |
| **`sc-mongodb`** | **Data Persistence:** Base de dades documental per a la persistència de partits, plantilles de jugadors i estadístiques històriques. | MongoDB Document Store |
| **`sc-label-studio`** | **Etiquetatge de dades:** Interfície web per etiquetar frames de vídeo (bounding boxes de jugadors i dorsals). Servei opcional que només s'aixeca durant sessions d'etiquetatge. S'integra amb MinIO per llegir frames i exportar datasets directament als buckets. | Label Studio, integració S3/MinIO |

## 1.1 Workflow de Dades (Frame-by-Frame Pipeline)

El processament de SmartChrono IP segueix un model asíncron basat en esdeveniments per optimitzar l'ús de la GPU i garantir la persistència de les dades.

### Fase A: Ingestió i Fragmentació
1. **Upload:** L'usuari puja el fitxer `.mp4` a través del `sc-frontend`. El `sc-api-gateway` el rep i l'emmagatzema al bucket `raw-videos` de **MinIO**.
2. **Trigger:** L'API publica un missatge a la cua de Redis `video_to_process`.
3. **Decomposició:** El `sc-video-manager` descarrega el vídeo, l'analitza amb FFmpeg i extreu cada frame en format `.jpg`. 
4. **Storage:** Cada frame es puja immediatament al bucket `pending-frames` de **MinIO** amb una clau única: `partit_id/frame_000001.jpg`.
5. **Indexing:** Per cada frame pujat, s'afegeix una tasca a la cua de Redis `task_frames`.

### Fase B: Inferència i Detecció
1. **Consum de tasques:** El `sc-inference-worker` (amb accés a la GPU) extreu els IDs de frame de Redis.
2. **Download & Predict:** - Descarrega el frame des de **MinIO**.
   - Executa **YOLOv8** per localitzar jugadors i la pilota.
   - Per a cada jugador detectat, realitza un *crop* de la zona del dorsal i l'envia a la **CNN**.
3. **Publish:** Els resultats (coordenades, classe, dorsal, confiança) s'envien a la cua `detected_frames_results`.

### Fase C: Seguiment i Lògica Esportiva
1. **Tracking:** El `sc-logic-aggregator` processa els resultats seqüencialment utilitzant **ByteTrack**. Assigna un `Player_ID` persistent a cada trajectòria.
2. **Càlcul de Minuts:** - Si un `Player_ID` és identificat amb un dorsal (ex: "8") i es manté actiu en pista, el sistema incrementa el seu comptador de temps a **MongoDB**.
   - Es gestionen les "zones mortes" (banqueta) per aturar el cronòmetre automàticament.
3. **Events:** Els esdeveniments especials (chutes, canvis) es guarden a la col·lecció `events` de **MongoDB**.

### Fase D: Re-muntatge i Feedback (Active Learning)
1. **Video Render:** Un cop finalitzat el partit, el `sc-video-manager` recupera les coordenades de Mongo, descarrega els frames de **MinIO**, dibuixa els *overlays* (caixes i noms) i genera el vídeo final que es guarda al bucket `processed-videos`.
2. **Feedback Loop:** - Aquells frames amb una confiança d'identificació baixa (< 0.6) o marcats manualment per l'usuari es copien al bucket `feedback-data`.
   - El `sc-active-learner` utilitza aquestes imatges per re-entrenar la CNN o el YOLO de forma asíncrona, generant una nova versió del model a `models/weights/`.