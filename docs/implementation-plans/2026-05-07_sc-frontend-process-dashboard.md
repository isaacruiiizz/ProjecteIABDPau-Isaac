# PJM-30 — ProcessPage + Dashboard redisseny

**Data:** 2026-05-07  
**Estat:** En Procés  
**Sprint:** 3 — Pipeline IA MVP  
**Ticket:** PJM-30  
**Etiqueta:** `frontend`  

---

## Objectiu

1. **Dashboard redissenyat** — pantalla principal amb dues accions clares: calcular minuts o etiquetar frames.
2. **ProcessPage** — wizard de 2 passos: puja vídeo → dibuixa ROI + temps inici/fi.

---

## Estètica existent (referència)

| Element | Classes Tailwind |
|---|---|
| Fons global | `min-h-screen bg-gray-50` |
| Targeta | `bg-white rounded-2xl shadow-sm border border-gray-200` |
| Botó primari | `bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium` |
| Botó secundari | `bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-100` |
| Input | `border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500` |
| Error | `bg-red-50 border border-red-200 text-red-600 rounded-lg` |
| Icona brand | `bg-blue-600 text-white rounded-2xl p-3` + `<Timer size={28} />` |

---

## Fitxers afectats

| Fitxer | Acció |
|---|---|
| `src/api/matches.ts` | Crear — client API per a matches |
| `src/pages/ProcessPage.tsx` | Crear — wizard 2 passos |
| `src/pages/DashboardPage.tsx` | Modificar — redisseny amb dues targetes |
| `src/App.tsx` | Modificar — afegir ruta `/process` |

---

## 1. `src/api/matches.ts` (nou)

```typescript
import apiClient from './client';

export interface RoiPoint { x: number; y: number; }

interface MatchCreateResponse { match_id: string; status: string; }
interface MatchConfigResponse {
  match_id: string;
  roi_polygon: RoiPoint[];
  start_seconds: number;
  end_seconds: number;
}

export async function createMatch(video: File, title: string): Promise<MatchCreateResponse> {
  const form = new FormData();
  form.append('video', video);
  form.append('title', title);
  const { data } = await apiClient.post<MatchCreateResponse>('/api/v1/matches', form);
  return data;
}

export async function updateMatchConfig(
  matchId: string,
  roi_polygon: RoiPoint[],
  start_seconds: number,
  end_seconds: number,
): Promise<MatchConfigResponse> {
  const { data } = await apiClient.patch<MatchConfigResponse>(
    `/api/v1/matches/${matchId}/config`,
    { roi_polygon, start_seconds, end_seconds },
  );
  return data;
}
```

---

## 2. `src/pages/DashboardPage.tsx` (redisseny)

### Mockup

```
┌──────────────────────────────────────────────┐
│   ⏱  SmartChrono IP                         │
│   Benvingut                                  │
│                                              │
│  ┌────────────────┐  ┌────────────────┐     │
│  │  🕐             │  │  🏷️            │     │
│  │  Calcular minuts│  │  Etiquetar     │     │
│  │                 │  │  frames        │     │
│  │  Processa un    │  │  Puja vídeos   │     │
│  │  vídeo i obté   │  │  per millorar  │     │
│  │  els minuts per │  │  el model d'IA │     │
│  │  jugador        │  │  [admin only]  │     │
│  │  [Començar →]   │  │  [Etiquetar →] │     │
│  └────────────────┘  └────────────────┘     │
│                                              │
│             [Tanca sessió]                   │
└──────────────────────────────────────────────┘
```

### Implementació

```tsx
import { useNavigate } from 'react-router-dom';
import { Clock, LogOut, Tag, Timer } from 'lucide-react';
import useAuthStore from '../store/authStore';

export default function DashboardPage() {
  const navigate = useNavigate();
  const clearToken = useAuthStore((s) => s.clearToken);
  const role = useAuthStore((s) => s.role);

  function handleLogout() {
    clearToken();
    navigate('/login', { replace: true });
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center px-4">
      {/* Header */}
      <div className="flex items-center gap-3 mb-2">
        <div className="bg-blue-600 text-white rounded-2xl p-3">
          <Timer size={28} />
        </div>
        <h1 className="text-2xl font-bold text-gray-900">SmartChrono IP</h1>
      </div>
      <p className="text-gray-500 text-sm mb-10">Benvingut</p>

      {/* Targetes d'acció */}
      <div className="flex flex-col sm:flex-row gap-4 w-full max-w-2xl">
        {/* Calcular minuts */}
        <button
          onClick={() => navigate('/process')}
          className="flex-1 bg-white rounded-2xl border border-gray-200 shadow-sm p-6
                     text-left hover:border-blue-300 hover:shadow-md transition-all group"
        >
          <div className="bg-blue-100 text-blue-600 rounded-xl p-3 w-fit mb-4
                          group-hover:bg-blue-600 group-hover:text-white transition-colors">
            <Clock size={24} />
          </div>
          <h2 className="text-base font-semibold text-gray-900 mb-1">Calcular minuts</h2>
          <p className="text-sm text-gray-500 mb-4">
            Processa un vídeo de partit i obté automàticament els minuts jugats per cada jugador.
          </p>
          <span className="text-sm font-medium text-blue-600 group-hover:underline">
            Començar →
          </span>
        </button>

        {/* Etiquetar frames — només admins */}
        {role === 'admin' && (
          <button
            onClick={() => navigate('/admin/labeling')}
            className="flex-1 bg-white rounded-2xl border border-gray-200 shadow-sm p-6
                       text-left hover:border-indigo-300 hover:shadow-md transition-all group"
          >
            <div className="bg-indigo-100 text-indigo-600 rounded-xl p-3 w-fit mb-4
                            group-hover:bg-indigo-600 group-hover:text-white transition-colors">
              <Tag size={24} />
            </div>
            <h2 className="text-base font-semibold text-gray-900 mb-1">Etiquetar frames</h2>
            <p className="text-sm text-gray-500 mb-4">
              Puja vídeos per etiquetar frames manualment i millorar el model de detecció.
            </p>
            <span className="text-sm font-medium text-indigo-600 group-hover:underline">
              Etiquetar →
            </span>
          </button>
        )}
      </div>

      {/* Logout */}
      <button
        onClick={handleLogout}
        className="mt-8 flex items-center gap-2 text-sm text-gray-500
                   hover:text-gray-700 transition-colors"
      >
        <LogOut size={15} />
        Tanca sessió
      </button>
    </div>
  );
}
```

---

## 3. `src/pages/ProcessPage.tsx` (nou)

### 3 passos (canvi respecte al pla inicial)

| Pas | Nom | Contingut |
|---|---|---|
| 1 | Upload | Selecció fitxer + títol → POST /matches |
| 2 | ROI | Canvas amb primer frame, dibuix del polígon |
| 3 | Temps | Vídeo reproductible + timeline scrubber de 2 handles |

---

### Mockup Pas 1 — Upload

```
┌─────────────────────────────────────────────┐
│  ← Tornar al Dashboard      [1]─[2]─[3]    │
│                                             │
│  ⏱ SmartChrono IP                          │
│  Puja el vídeo del partit                  │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  ⬆                                  │   │
│  │  Arrossega el fitxer .mp4 aquí      │   │
│  │  o fes clic per seleccionar         │   │
│  │  [Seleccionar vídeo]                │   │
│  │  ✓ lliga_j12.mp4  (145 MB)         │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  Títol del partit                           │
│  [Lliga J12 vs Joventut_____________]      │
│                                             │
│  [⬆ Puja el vídeo]                         │
└─────────────────────────────────────────────┘
```

---

### Mockup Pas 2 — ROI

```
┌─────────────────────────────────────────────┐
│  ← Tornar              [1]─●[2]─[3]        │
│                                             │
│  Defineix la zona de joc                   │
│  Fes clic al canvas per afegir vèrtexs     │
│                                             │
│  ╔═════════════════════════════════════╗   │
│  ║  ·─────────────────────·            ║   │
│  ║  │  [primer frame      │            ║   │
│  ║  │   del vídeo]        │            ║   │
│  ║  ·─────────────────────·            ║   │
│  ╚═════════════════════════════════════╝   │
│                                             │
│  4 vèrtexs  [↩ Desfer]  [✕ Reiniciar]     │
│                                             │
│  [Continuar →]  (habilitat si ≥3 vèrtexs) │
└─────────────────────────────────────────────┘
```

---

### Mockup Pas 3 — Timeline scrubber

```
┌─────────────────────────────────────────────┐
│  ← Tornar              [1]─[2]─●[3]        │
│                                             │
│  Selecciona el temps del partit             │
│  Arrossega els extrems per definir l'inici  │
│  i el final del temps jugat                 │
│                                             │
│  ╔═════════════════════════════════════╗   │
│  ║                                     ║   │
│  ║  [VÍDEO reproductible amb controls] ║   │
│  ║                                     ║   │
│  ╚═════════════════════════════════════╝   │
│                                             │
│  ░░░░│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│░░░░░░░░░░  │
│      ◉                       ◉             │
│   05:30                   38:20            │
│                                             │
│  Inici: 05:30    Fi: 38:20    (32:50 min)  │
│                                             │
│  [Desa configuració →]                     │
└─────────────────────────────────────────────┘
```

La barra de la timeline:
- **Gris clar** (░) = fora del rang seleccionat
- **Blau** (▓) = rang seleccionat (temps de joc)
- **◉ handles** = arrossegables; en moure'ls el vídeo fa seek en temps real
- **Etiquetes** = temps en format MM:SS mostrats sota cada handle

---

### Estat complet

```typescript
type Step = 'upload' | 'roi' | 'time';

// Pas 1
const [file, setFile] = useState<File | null>(null);
const [title, setTitle] = useState('');
const [uploading, setUploading] = useState(false);
const [uploadError, setUploadError] = useState<string | null>(null);

// Navegació
const [step, setStep] = useState<Step>('upload');
const [matchId, setMatchId] = useState('');

// Pas 2 — ROI
const [roiPoints, setRoiPoints] = useState<RoiPoint[]>([]);

// Pas 3 — Timeline
const [duration, setDuration] = useState(0);
const [startSec, setStartSec] = useState(0);
const [endSec, setEndSec] = useState(0);
const [dragging, setDragging] = useState<'start' | 'end' | null>(null);

// Submit
const [saving, setSaving] = useState(false);
const [saveError, setSaveError] = useState<string | null>(null);
const [done, setDone] = useState(false);

const canvasRef = useRef<HTMLCanvasElement>(null);
const videoRef = useRef<HTMLVideoElement>(null);
const timelineRef = useRef<HTMLDivElement>(null);
```

---

### Lògica ROI (Pas 2)

**Càrrega del primer frame (local, sense backend):**
```typescript
function loadVideo(f: File) {
  const url = URL.createObjectURL(f);
  const video = videoRef.current!;
  video.src = url;
  // El vídeo es carrega per a TOTS dos usos: canvas ROI + timeline
}

// Per al canvas: quan video.onseeked al currentTime=0
function drawFirstFrame() {
  const video = videoRef.current!;
  const canvas = canvasRef.current!;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d')!.drawImage(video, 0, 0);
  redrawRoi();
}
```

**Dibuixar polígon ROI:**
```typescript
function redrawRoi() {
  const canvas = canvasRef.current!;
  const video = videoRef.current!;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(video, 0, 0);
  if (roiPoints.length === 0) return;
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 3]);
  ctx.beginPath();
  ctx.moveTo(roiPoints[0].x, roiPoints[0].y);
  roiPoints.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
  if (roiPoints.length >= 3) ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#2563eb';
  roiPoints.forEach((p) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
    ctx.fill();
  });
}
```

**Click canvas → nou vèrtex:**
```typescript
function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
  const rect = canvasRef.current!.getBoundingClientRect();
  const scaleX = canvasRef.current!.width / rect.width;
  const scaleY = canvasRef.current!.height / rect.height;
  setRoiPoints((prev) => [...prev, {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  }]);
}
```

---

### Lògica Timeline (Pas 3)

**Inicialitzar quan es carrega la durada:**
```typescript
// onLoadedMetadata del <video>
function handleVideoMetadata() {
  const d = videoRef.current!.duration;
  setDuration(d);
  setStartSec(0);
  setEndSec(d);
}
```

**Convertir posició X del ratolí a segons:**
```typescript
function clientXToSeconds(clientX: number): number {
  const rect = timelineRef.current!.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  return pct * duration;
}
```

**Drag dels handles (pointer events per suportar touch + mouse):**
```typescript
function handlePointerMove(e: PointerEvent) {
  if (!dragging || !duration) return;
  const sec = clientXToSeconds(e.clientX);

  if (dragging === 'start') {
    const s = Math.min(sec, endSec - 1);
    setStartSec(s);
    videoRef.current!.currentTime = s;   // seek en temps real
  } else {
    const s = Math.max(sec, startSec + 1);
    setEndSec(s);
    videoRef.current!.currentTime = s;   // seek en temps real
  }
}

useEffect(() => {
  if (!dragging) return;
  window.addEventListener('pointermove', handlePointerMove);
  window.addEventListener('pointerup', () => setDragging(null));
  return () => {
    window.removeEventListener('pointermove', handlePointerMove);
    window.removeEventListener('pointerup', () => setDragging(null));
  };
}, [dragging, startSec, endSec, duration]);
```

**JSX de la timeline:**
```tsx
<div className="space-y-2">
  {/* Barra */}
  <div
    ref={timelineRef}
    className="relative h-6 bg-gray-200 rounded-full select-none cursor-pointer"
  >
    {/* Rang seleccionat */}
    <div
      className="absolute h-full bg-blue-500 rounded-full"
      style={{
        left: `${(startSec / duration) * 100}%`,
        width: `${((endSec - startSec) / duration) * 100}%`,
      }}
    />
    {/* Handle inici */}
    <div
      className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2
                 w-5 h-5 bg-white border-2 border-blue-600 rounded-full
                 shadow cursor-grab active:cursor-grabbing"
      style={{ left: `${(startSec / duration) * 100}%` }}
      onPointerDown={(e) => { e.stopPropagation(); setDragging('start'); }}
    />
    {/* Handle fi */}
    <div
      className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2
                 w-5 h-5 bg-white border-2 border-blue-600 rounded-full
                 shadow cursor-grab active:cursor-grabbing"
      style={{ left: `${(endSec / duration) * 100}%` }}
      onPointerDown={(e) => { e.stopPropagation(); setDragging('end'); }}
    />
  </div>

  {/* Etiquetes de temps */}
  <div className="flex justify-between text-xs text-gray-500 px-1">
    <span>00:00</span>
    <span>{formatTime(duration)}</span>
  </div>

  {/* Resum selecció */}
  <div className="flex items-center justify-between text-sm">
    <span className="text-gray-700">
      Inici: <strong>{formatTime(startSec)}</strong>
    </span>
    <span className="text-gray-400 text-xs">
      {formatTime(endSec - startSec)} de joc
    </span>
    <span className="text-gray-700">
      Fi: <strong>{formatTime(endSec)}</strong>
    </span>
  </div>
</div>
```

**Helper format:**
```typescript
function formatTime(s: number): string {
  const m = Math.floor(s / 60).toString().padStart(2, '0');
  const sec = Math.floor(s % 60).toString().padStart(2, '0');
  return `${m}:${sec}`;
}
```

---

### Flux de submit

```typescript
// Pas 1 → Pas 2
async function handleUpload() {
  setUploading(true);
  try {
    const { match_id } = await createMatch(file!, title);
    setMatchId(match_id);
    loadVideo(file!);      // carrega el vídeo (usat per canvas ROI i timeline)
    setStep('roi');
  } catch { setUploadError('Error pujant el vídeo.'); }
  finally { setUploading(false); }
}

// Pas 2 → Pas 3
function handleRoiNext() {
  // el vídeo ja està carregat, onLoadedMetadata inicialitza duration
  setStep('time');
}

// Pas 3 → desa
async function handleSave() {
  setSaving(true);
  try {
    await updateMatchConfig(matchId, roiPoints, startSec, endSec);
    setDone(true);
  } catch { setSaveError('Error desant la configuració.'); }
  finally { setSaving(false); }
}
```

---

### Pantalla done (provisional fins Sprint 4)

```tsx
<div className="text-center p-8">
  <div className="bg-green-100 text-green-600 rounded-full p-4 w-fit mx-auto mb-4">
    <CheckCircle size={32} />
  </div>
  <h2 className="text-lg font-semibold text-gray-900 mb-1">Configuració desada</h2>
  <p className="text-sm text-gray-500 mb-6">
    El processament automàtic s'implementarà al Sprint 4.
  </p>
  <button onClick={() => navigate('/')}
    className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-5 rounded-lg text-sm">
    Tornar al Dashboard
  </button>
</div>
```

---

## 4. `src/App.tsx` (modificar)

```tsx
// Afegir import:
import ProcessPage from './pages/ProcessPage';

// Afegir dins de <Route element={<ProtectedRoute />}>:
<Route path="/process" element={<ProcessPage />} />
```

---

## Resum de rutes

| Ruta | Component | Guard |
|---|---|---|
| `/` | `DashboardPage` (redisseny) | ProtectedRoute |
| `/process` | `ProcessPage` (nou) | ProtectedRoute |
| `/admin/labeling` | `LabelingPage` (existent) | AdminRoute |
| `/login` | `LoginPage` (existent) | — |

---

## Decisions tècniques

- **3 passos en lloc de 2:** separa ROI (canvas estàtic) de la selecció de temps (vídeo reproductible), cada pas té un sol focus.
- **Un sol `<video>` per a tot:** el mateix element `<video>` s'usa en Pas 2 (ocult, drawImage al canvas) i en Pas 3 (visible, amb controls natius). Estalvia memòria i doble càrrega.
- **Seek en temps real:** `videoRef.current.currentTime = sec` dins del `pointermove` → el vídeo mostra exactament el frame del moment seleccionat mentre l'usuari arrossega.
- **Pointer events:** `onPointerDown` + `window.addEventListener('pointermove')` permet drag fluid fora dels límits del handle (millor que `onMouseMove`).
- **`startSec`/`endSec` en segons float:** van directament a `PATCH /config` sense conversió addicional.
- **ROI per clic:** cada clic afegeix un vèrtex; `ctx.closePath()` quan ≥3 punts → el polígon es veu sempre tancat.
- **Targeta labeling oculta a no-admins:** `role === 'admin'` des del store Zustand, consistent amb `AdminRoute`.
