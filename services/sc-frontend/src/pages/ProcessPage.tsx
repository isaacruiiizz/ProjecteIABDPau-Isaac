import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, CheckCircle, Timer, UploadCloud, X } from 'lucide-react';
import { createMatch, updateMatchConfig, type RoiPoint } from '../api/matches';

type Step = 'upload' | 'roi' | 'time';

interface RoiRect { x1: number; y1: number; x2: number; y2: number; }

const STEP_LABELS: { key: Step; label: string }[] = [
  { key: 'upload', label: 'Puja'  },
  { key: 'roi',    label: 'ROI'   },
  { key: 'time',   label: 'Temps' },
];
const STEP_ORDER: Step[] = ['upload', 'roi', 'time'];

function formatTime(s: number): string {
  const m   = Math.floor(s / 60).toString().padStart(2, '0');
  const sec = Math.floor(s % 60).toString().padStart(2, '0');
  return `${m}:${sec}`;
}

function rectToPolygon(r: RoiRect): RoiPoint[] {
  const x1 = Math.min(r.x1, r.x2);
  const y1 = Math.min(r.y1, r.y2);
  const x2 = Math.max(r.x1, r.x2);
  const y2 = Math.max(r.y1, r.y2);
  return [
    { x: x1, y: y1 },
    { x: x2, y: y1 },
    { x: x2, y: y2 },
    { x: x1, y: y2 },
  ];
}

export default function ProcessPage() {
  const navigate = useNavigate();

  // ── Pas 1 ──────────────────────────────────────────────────────────────────
  const [file,           setFile]           = useState<File | null>(null);
  const [title,          setTitle]          = useState('');
  const [uploading,      setUploading]      = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError,    setUploadError]    = useState<string | null>(null);

  // ── Navegació ──────────────────────────────────────────────────────────────
  const [step,    setStep]    = useState<Step>('upload');
  const [matchId, setMatchId] = useState('');

  // ── Pas 2: ROI (drag rectangle) ────────────────────────────────────────────
  const [roiRect,      setRoiRect]      = useState<RoiRect | null>(null);
  const [roiDragStart, setRoiDragStart] = useState<{ x: number; y: number } | null>(null);

  // ── Pas 3: timeline ────────────────────────────────────────────────────────
  const [duration,  setDuration]  = useState(0);
  const [startSec,  setStartSec]  = useState(0);
  const [endSec,    setEndSec]    = useState(0);
  const [dragging,  setDragging]  = useState<'start' | 'end' | null>(null);
  const [videoReady, setVideoReady] = useState(false);

  // ── Submit ─────────────────────────────────────────────────────────────────
  const [saving,    setSaving]    = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [done,      setDone]      = useState(false);

  // ── Refs ────────────────────────────────────────────────────────────────────
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const videoRef    = useRef<HTMLVideoElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const blobUrlRef  = useRef('');

  // Refs per evitar closures obsoletes als pointer events del timeline
  const draggingRef = useRef<'start' | 'end' | null>(null);
  const startRef    = useRef(0);
  const endRef      = useRef(0);
  const durRef      = useRef(0);

  useEffect(() => { startRef.current = startSec; }, [startSec]);
  useEffect(() => { endRef.current   = endSec;   }, [endSec]);
  useEffect(() => { durRef.current   = duration; }, [duration]);

  // Allibera blob URL en desmontar
  useEffect(() => () => {
    if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
  }, []);

  // ── FIX vídeo negre: re-aplica el src quan el <video> es remunta (step→time) ──
  useEffect(() => {
    if (step === 'time' && videoRef.current && blobUrlRef.current) {
      setVideoReady(false);
      videoRef.current.src = blobUrlRef.current;
    }
  }, [step]);

  // ── Pointer events del timeline (registrats una sola vegada) ──────────────
  useEffect(() => {
    function onMove(e: PointerEvent) {
      if (!draggingRef.current || !timelineRef.current || !durRef.current) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const pct  = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const sec  = pct * durRef.current;
      if (draggingRef.current === 'start') {
        const s = Math.min(sec, endRef.current - 1);
        setStartSec(s); startRef.current = s;
        if (videoRef.current) videoRef.current.currentTime = s;
      } else {
        const s = Math.max(sec, startRef.current + 1);
        setEndSec(s); endRef.current = s;
        if (videoRef.current) videoRef.current.currentTime = s;
      }
    }
    function onUp() { draggingRef.current = null; setDragging(null); }
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup',   onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup',   onUp);
    };
  }, []);

  // ── Canvas ROI: dibuixa frame + rectangle ──────────────────────────────────
  const redrawRoi = useCallback(() => {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video || !video.videoWidth) return;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(video, 0, 0);
    if (!roiRect) return;
    const x = Math.min(roiRect.x1, roiRect.x2);
    const y = Math.min(roiRect.y1, roiRect.y2);
    const w = Math.abs(roiRect.x2 - roiRect.x1);
    const h = Math.abs(roiRect.y2 - roiRect.y1);
    // Àrea seleccionada
    ctx.fillStyle = 'rgba(37, 99, 235, 0.15)';
    ctx.fillRect(x, y, w, h);
    // Vora
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(x, y, w, h);
    // Cantonades
    ctx.fillStyle = '#2563eb';
    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]].forEach(([cx, cy]) => {
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [roiRect]);

  useEffect(() => {
    if (step === 'roi') redrawRoi();
  }, [roiRect, step, redrawRoi]);

  // Inicialitza canvas quan entrem al pas ROI
  useEffect(() => {
    if (step !== 'roi') return;
    const video = videoRef.current!;

    function drawFrame() {
      const canvas = canvasRef.current!;
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
      redrawRoi();
    }
    function doSeek() {
      video.currentTime = 0;
      video.addEventListener('seeked', drawFrame, { once: true });
    }

    if (video.readyState >= 1) doSeek();
    else video.addEventListener('loadedmetadata', doSeek, { once: true });

    return () => {
      video.removeEventListener('loadedmetadata', doSeek);
      video.removeEventListener('seeked', drawFrame);
    };
  }, [step, redrawRoi]);

  // ── Helpers coordenades canvas ─────────────────────────────────────────────
  function getCanvasPoint(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current!;
    const rect   = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width  / rect.width),
      y: (e.clientY - rect.top)  * (canvas.height / rect.height),
    };
  }

  // ── Handlers ───────────────────────────────────────────────────────────────
  function pickFile(f: File) {
    setFile(f);
    setTitle(f.name.replace(/\.[^/.]+$/, ''));
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if (f) pickFile(f);
  }

  function handleDrop(e: React.DragEvent<HTMLLabelElement>) {
    e.preventDefault();
    const f = e.dataTransfer.files[0]; if (f) pickFile(f);
  }

  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    setUploadProgress(0);
    try {
      const { match_id } = await createMatch(file, title, setUploadProgress);
      setMatchId(match_id);
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = URL.createObjectURL(file);
      videoRef.current!.src = blobUrlRef.current;
      setStep('roi');
    } catch {
      setUploadError('Error pujant el vídeo. Torna-ho a intentar.');
    } finally {
      setUploading(false);
    }
  }

  // ROI drag handlers
  function handleRoiDown(e: React.PointerEvent<HTMLCanvasElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    setRoiRect(null);
    const pt = getCanvasPoint(e);
    setRoiDragStart(pt);
  }

  function handleRoiMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!roiDragStart) return;
    const pt = getCanvasPoint(e);
    setRoiRect({ x1: roiDragStart.x, y1: roiDragStart.y, x2: pt.x, y2: pt.y });
  }

  function handleRoiUp(e: React.PointerEvent<HTMLCanvasElement>) {
    e.currentTarget.releasePointerCapture(e.pointerId);
    setRoiDragStart(null);
  }

  function handleVideoMetadata() {
    const d = videoRef.current!.duration;
    setDuration(d); setStartSec(0); setEndSec(d);
    durRef.current = d; endRef.current = d;
    setVideoReady(true);
  }

  function startDrag(handle: 'start' | 'end') {
    draggingRef.current = handle; setDragging(handle);
  }

  async function handleSave() {
    if (!roiRect) return;
    setSaving(true); setSaveError(null);
    try {
      await updateMatchConfig(matchId, rectToPolygon(roiRect), startSec, endSec);
      setDone(true);
    } catch {
      setSaveError('Error desant la configuració. Torna-ho a intentar.');
    } finally {
      setSaving(false);
    }
  }

  function goBack() {
    const idx = STEP_ORDER.indexOf(step);
    if (idx === 0) navigate('/');
    else setStep(STEP_ORDER[idx - 1]);
  }

  // Validació ROI: rectangle amb àrea mínima de 20px cadascuna de les dimensions
  const roiValid = roiRect !== null
    && Math.abs(roiRect.x2 - roiRect.x1) > 20
    && Math.abs(roiRect.y2 - roiRect.y1) > 20;

  const currentIdx = STEP_ORDER.indexOf(step);

  // ── Done ───────────────────────────────────────────────────────────────────
  if (done) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-10 text-center max-w-sm w-full">
          <div className="bg-green-100 text-green-600 rounded-full p-4 w-fit mx-auto mb-4">
            <CheckCircle size={32} />
          </div>
          <h2 className="text-lg font-semibold text-gray-900 mb-1">Configuració desada</h2>
          <p className="text-sm text-gray-500 mb-6">
            El processament automàtic s'implementarà al Sprint 4.
          </p>
          <button
            onClick={() => navigate('/')}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium
                       py-2.5 rounded-lg text-sm transition-colors"
          >
            Tornar al Dashboard
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 px-4 py-8">
      {/*
        Vídeo ocult en passos 'upload' i 'roi' (però sempre al DOM si ja té src).
        En pas 'time' es mostra dins de la targeta.
        El <video> ocult permet drawImage al canvas del pas ROI.
      */}
      {step !== 'time' && (
        <video
          ref={videoRef}
          onLoadedMetadata={handleVideoMetadata}
          className="hidden"
        />
      )}

      <div className="max-w-2xl mx-auto">

        {/* Capçalera + indicador de passos */}
        <div className="flex items-center justify-between mb-8">
          <button onClick={goBack} className="text-sm text-gray-500 hover:text-gray-700 transition-colors">
            ← Tornar
          </button>
          <div className="flex items-center gap-1.5">
            {STEP_LABELS.map((s, i) => (
              <div key={s.key} className="flex items-center gap-1.5">
                <div className="flex items-center gap-1">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center
                                   text-xs font-semibold transition-colors
                    ${i < currentIdx
                      ? 'bg-blue-600 text-white'
                      : i === currentIdx
                        ? 'bg-blue-100 text-blue-600 border-2 border-blue-500'
                        : 'bg-gray-100 text-gray-400'}`}
                  >
                    {i + 1}
                  </div>
                  <span className={`text-xs hidden sm:block
                    ${i <= currentIdx ? 'text-blue-600' : 'text-gray-400'}`}>
                    {s.label}
                  </span>
                </div>
                {i < STEP_LABELS.length - 1 && (
                  <div className={`w-5 h-px ${i < currentIdx ? 'bg-blue-400' : 'bg-gray-300'}`} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* ═══════ PAS 1: UPLOAD ═══════ */}
        {step === 'upload' && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="bg-blue-600 text-white rounded-2xl p-2.5"><Timer size={22} /></div>
              <div>
                <h1 className="text-lg font-semibold text-gray-900">Puja el vídeo del partit</h1>
                <p className="text-xs text-gray-500">Fitxer MP4 del vídeo complet</p>
              </div>
            </div>

            {/* Drop zone */}
            <label
              className="block mb-4 cursor-pointer"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
            >
              <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors
                ${file
                  ? 'border-blue-300 bg-blue-50'
                  : 'border-gray-300 hover:border-blue-300 hover:bg-gray-50'}`}
              >
                <UploadCloud size={32} className={`mx-auto mb-2 ${file ? 'text-blue-500' : 'text-gray-400'}`} />
                {file ? (
                  <>
                    <p className="text-sm font-medium text-blue-700">{file.name}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
                  </>
                ) : (
                  <>
                    <p className="text-sm text-gray-600">Arrossega el fitxer .mp4 aquí</p>
                    <p className="text-xs text-gray-400 mt-0.5">o fes clic per seleccionar</p>
                  </>
                )}
              </div>
              <input type="file" accept="video/mp4,video/*" className="hidden"
                     onChange={handleFileChange} disabled={uploading} />
            </label>

            {/* Títol */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Títol del partit</label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Ex: Lliga J12 vs Joventut"
                disabled={uploading}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm
                           focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                           disabled:bg-gray-50 disabled:text-gray-400"
              />
            </div>

            {/* Barra de progrés (visible durant la pujada) */}
            {uploading && (
              <div className="mb-4">
                <div className="flex justify-between text-xs text-gray-500 mb-1.5">
                  <span>Pujant al servidor...</span>
                  <span className="font-medium">{uploadProgress}%</span>
                </div>
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-600 rounded-full transition-all duration-200"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {uploadError && (
              <div className="flex items-start gap-2 text-red-600 text-sm mb-4
                              bg-red-50 border border-red-200 rounded-lg px-3 py-2">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <span>{uploadError}</span>
              </div>
            )}

            <button
              onClick={handleUpload}
              disabled={!file || !title.trim() || uploading}
              className="w-full flex items-center justify-center gap-2 bg-blue-600
                         hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium
                         py-2.5 rounded-lg text-sm transition-colors"
            >
              <UploadCloud size={16} />
              {uploading ? 'Pujant...' : 'Puja el vídeo'}
            </button>
          </div>
        )}

        {/* ═══════ PAS 2: ROI ═══════ */}
        {step === 'roi' && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
            <h1 className="text-lg font-semibold text-gray-900 mb-1">Defineix la zona de joc</h1>
            <p className="text-xs text-gray-500 mb-4">
              Arrossega sobre el camp per seleccionar la zona de joc (ROI)
            </p>

            <div className="rounded-xl overflow-hidden bg-black mb-4 cursor-crosshair">
              <canvas
                ref={canvasRef}
                onPointerDown={handleRoiDown}
                onPointerMove={handleRoiMove}
                onPointerUp={handleRoiUp}
                className="w-full block select-none"
              />
            </div>

            <div className="flex items-center justify-between mb-6">
              <span className="text-sm text-gray-600">
                {roiValid
                  ? <span className="text-green-600 font-medium">✓ Zona seleccionada</span>
                  : roiDragStart
                    ? <span className="text-blue-500">Arrossegant...</span>
                    : <span className="text-gray-400">Arrossega per seleccionar</span>
                }
              </span>
              <button
                onClick={() => setRoiRect(null)}
                disabled={!roiRect}
                className="flex items-center gap-1 text-xs text-gray-600 border border-gray-300
                           rounded-lg px-3 py-1.5 hover:bg-gray-50 disabled:opacity-40 transition-colors"
              >
                <X size={13} /> Reiniciar
              </button>
            </div>

            <button
              onClick={() => setStep('time')}
              disabled={!roiValid}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300
                         text-white font-medium py-2.5 rounded-lg text-sm transition-colors"
            >
              Continuar →
            </button>
          </div>
        )}

        {/* ═══════ PAS 3: TIMELINE ═══════ */}
        {step === 'time' && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
            <h1 className="text-lg font-semibold text-gray-900 mb-1">
              Selecciona el temps del partit
            </h1>
            <p className="text-xs text-gray-500 mb-4">
              Arrossega els extrems de la barra per definir l'inici i el final del temps jugat
            </p>

            {/* Vídeo — ref s'aplica aquí en el pas time */}
            <div className="rounded-xl overflow-hidden bg-black mb-4">
              {!videoReady && (
                <div className="h-40 flex items-center justify-center">
                  <p className="text-gray-400 text-sm">Carregant vídeo...</p>
                </div>
              )}
              <video
                ref={videoRef}
                controls
                onLoadedMetadata={handleVideoMetadata}
                className={`w-full max-h-72 ${videoReady ? 'block' : 'hidden'}`}
              />
            </div>

            {/* Timeline scrubber */}
            {duration > 0 && (
              <div className="mb-6 space-y-3">
                <div
                  ref={timelineRef}
                  className={`relative h-6 bg-gray-200 rounded-full select-none
                    ${dragging ? 'cursor-grabbing' : ''}`}
                >
                  <div
                    className="absolute h-full bg-blue-500 rounded-full pointer-events-none"
                    style={{
                      left:  `${(startSec / duration) * 100}%`,
                      width: `${((endSec - startSec) / duration) * 100}%`,
                    }}
                  />
                  <div
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2
                               w-5 h-5 bg-white border-2 border-blue-600 rounded-full
                               shadow-md cursor-grab active:cursor-grabbing touch-none z-10"
                    style={{ left: `${(startSec / duration) * 100}%` }}
                    onPointerDown={(e) => { e.stopPropagation(); startDrag('start'); }}
                  />
                  <div
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2
                               w-5 h-5 bg-white border-2 border-blue-600 rounded-full
                               shadow-md cursor-grab active:cursor-grabbing touch-none z-10"
                    style={{ left: `${(endSec / duration) * 100}%` }}
                    onPointerDown={(e) => { e.stopPropagation(); startDrag('end'); }}
                  />
                </div>

                <div className="flex items-center justify-between text-sm">
                  <div className="text-center">
                    <p className="text-xs text-gray-400 mb-0.5">Inici</p>
                    <p className="font-semibold text-gray-900">{formatTime(startSec)}</p>
                  </div>
                  <span className="text-xs text-gray-400">
                    {formatTime(endSec - startSec)} de joc
                  </span>
                  <div className="text-center">
                    <p className="text-xs text-gray-400 mb-0.5">Fi</p>
                    <p className="font-semibold text-gray-900">{formatTime(endSec)}</p>
                  </div>
                </div>
              </div>
            )}

            {saveError && (
              <div className="flex items-start gap-2 text-red-600 text-sm mb-4
                              bg-red-50 border border-red-200 rounded-lg px-3 py-2">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <span>{saveError}</span>
              </div>
            )}

            <button
              onClick={handleSave}
              disabled={saving || !videoReady}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300
                         text-white font-medium py-2.5 rounded-lg text-sm transition-colors"
            >
              {saving ? 'Desant...' : 'Desa configuració →'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
