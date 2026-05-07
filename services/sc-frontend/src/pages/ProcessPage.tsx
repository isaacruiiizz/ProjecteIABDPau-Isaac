import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, CheckCircle, Timer, Undo2, UploadCloud, X } from 'lucide-react';
import { createMatch, updateMatchConfig, type RoiPoint } from '../api/matches';

type Step = 'upload' | 'roi' | 'time';

const STEP_LABELS: { key: Step; label: string }[] = [
  { key: 'upload', label: 'Puja' },
  { key: 'roi',    label: 'ROI'  },
  { key: 'time',   label: 'Temps' },
];
const STEP_ORDER: Step[] = ['upload', 'roi', 'time'];

function formatTime(s: number): string {
  const m   = Math.floor(s / 60).toString().padStart(2, '0');
  const sec = Math.floor(s % 60).toString().padStart(2, '0');
  return `${m}:${sec}`;
}

export default function ProcessPage() {
  const navigate = useNavigate();

  // ── Pas 1 ──────────────────────────────────────────────────────────────────
  const [file,        setFile]        = useState<File | null>(null);
  const [title,       setTitle]       = useState('');
  const [uploading,   setUploading]   = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // ── Navegació ──────────────────────────────────────────────────────────────
  const [step,    setStep]    = useState<Step>('upload');
  const [matchId, setMatchId] = useState('');

  // ── Pas 2: ROI ─────────────────────────────────────────────────────────────
  const [roiPoints, setRoiPoints] = useState<RoiPoint[]>([]);

  // ── Pas 3: timeline ────────────────────────────────────────────────────────
  const [duration, setDuration] = useState(0);
  const [startSec, setStartSec] = useState(0);
  const [endSec,   setEndSec]   = useState(0);
  const [dragging, setDragging] = useState<'start' | 'end' | null>(null);

  // ── Submit ─────────────────────────────────────────────────────────────────
  const [saving,    setSaving]    = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [done,      setDone]      = useState(false);

  // ── Refs ────────────────────────────────────────────────────────────────────
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const videoRef    = useRef<HTMLVideoElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const blobUrlRef  = useRef('');

  // Refs per evitar closures obsoletes als pointer events del window
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

  // ── Pointer events (registrats una sola vegada) ────────────────────────────
  useEffect(() => {
    function onMove(e: PointerEvent) {
      if (!draggingRef.current || !timelineRef.current || !durRef.current) return;
      const rect = timelineRef.current.getBoundingClientRect();
      const pct  = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const sec  = pct * durRef.current;

      if (draggingRef.current === 'start') {
        const s = Math.min(sec, endRef.current - 1);
        setStartSec(s);
        startRef.current = s;
        if (videoRef.current) videoRef.current.currentTime = s;
      } else {
        const s = Math.max(sec, startRef.current + 1);
        setEndSec(s);
        endRef.current = s;
        if (videoRef.current) videoRef.current.currentTime = s;
      }
    }

    function onUp() {
      draggingRef.current = null;
      setDragging(null);
    }

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup',   onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup',   onUp);
    };
  }, []);

  // ── Canvas ROI ─────────────────────────────────────────────────────────────
  const redrawRoi = useCallback(() => {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video || !video.videoWidth) return;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(video, 0, 0);
    if (roiPoints.length === 0) return;
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth   = 2;
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
  }, [roiPoints]);

  useEffect(() => {
    if (step === 'roi') redrawRoi();
  }, [roiPoints, step, redrawRoi]);

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

    if (video.readyState >= 1) {
      doSeek();
    } else {
      video.addEventListener('loadedmetadata', doSeek, { once: true });
    }

    return () => {
      video.removeEventListener('loadedmetadata', doSeek);
      video.removeEventListener('seeked', drawFrame);
    };
  }, [step, redrawRoi]);

  // ── Handlers ───────────────────────────────────────────────────────────────

  function pickFile(f: File) {
    setFile(f);
    setTitle(f.name.replace(/\.[^/.]+$/, ''));
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) pickFile(f);
  }

  function handleDrop(e: React.DragEvent<HTMLLabelElement>) {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) pickFile(f);
  }

  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      const { match_id } = await createMatch(file, title);
      setMatchId(match_id);
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
      const url = URL.createObjectURL(file);
      blobUrlRef.current = url;
      videoRef.current!.src = url;
      setStep('roi');
    } catch {
      setUploadError('Error pujant el vídeo. Torna-ho a intentar.');
    } finally {
      setUploading(false);
    }
  }

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current!;
    const rect   = canvas.getBoundingClientRect();
    const scaleX = canvas.width  / rect.width;
    const scaleY = canvas.height / rect.height;
    setRoiPoints((prev) => [
      ...prev,
      { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY },
    ]);
  }

  function handleVideoMetadata() {
    const d = videoRef.current!.duration;
    setDuration(d);
    setStartSec(0);
    setEndSec(d);
    durRef.current = d;
    endRef.current = d;
  }

  function startDrag(handle: 'start' | 'end') {
    draggingRef.current = handle;
    setDragging(handle);
  }

  async function handleSave() {
    setSaving(true);
    setSaveError(null);
    try {
      await updateMatchConfig(matchId, roiPoints, startSec, endSec);
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
        Un sol <video> per a tot el wizard.
        En pas 'roi' és ocult però usable per drawImage al canvas.
        En pas 'time' apareix visible (el JSX del pas Time el renderitza amb controls).
      */}
      {step !== 'time' && (
        <video ref={videoRef} onLoadedMetadata={handleVideoMetadata} className="hidden" />
      )}

      <div className="max-w-2xl mx-auto">

        {/* Capçalera + indicador de passos */}
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={goBack}
            className="text-sm text-gray-500 hover:text-gray-700 transition-colors"
          >
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
              <div className="bg-blue-600 text-white rounded-2xl p-2.5">
                <Timer size={22} />
              </div>
              <div>
                <h1 className="text-lg font-semibold text-gray-900">Puja el vídeo del partit</h1>
                <p className="text-xs text-gray-500">Fitxer MP4 del vídeo complet</p>
              </div>
            </div>

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
                <UploadCloud
                  size={32}
                  className={`mx-auto mb-2 ${file ? 'text-blue-500' : 'text-gray-400'}`}
                />
                {file ? (
                  <>
                    <p className="text-sm font-medium text-blue-700">{file.name}</p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {(file.size / 1024 / 1024).toFixed(1)} MB
                    </p>
                  </>
                ) : (
                  <>
                    <p className="text-sm text-gray-600">Arrossega el fitxer .mp4 aquí</p>
                    <p className="text-xs text-gray-400 mt-0.5">o fes clic per seleccionar</p>
                  </>
                )}
              </div>
              <input
                type="file"
                accept="video/mp4,video/*"
                className="hidden"
                onChange={handleFileChange}
              />
            </label>

            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Títol del partit
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Ex: Lliga J12 vs Joventut"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm
                           focus:outline-none focus:ring-2 focus:ring-blue-500
                           focus:border-transparent"
              />
            </div>

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
            <h1 className="text-lg font-semibold text-gray-900 mb-1">
              Defineix la zona de joc
            </h1>
            <p className="text-xs text-gray-500 mb-4">
              Fes clic sobre el camp per afegir els vèrtexs del polígon ROI
            </p>

            <div className="rounded-xl overflow-hidden bg-black mb-4">
              <canvas
                ref={canvasRef}
                onClick={handleCanvasClick}
                className="w-full cursor-crosshair block"
              />
            </div>

            <div className="flex items-center justify-between mb-6">
              <span className="text-sm text-gray-600">
                {roiPoints.length} vèrtex{roiPoints.length !== 1 ? 's' : ''}
                {roiPoints.length >= 3 && (
                  <span className="text-green-600 ml-1.5 font-medium">✓ llest</span>
                )}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setRoiPoints((p) => p.slice(0, -1))}
                  disabled={roiPoints.length === 0}
                  className="flex items-center gap-1 text-xs text-gray-600 border
                             border-gray-300 rounded-lg px-3 py-1.5 hover:bg-gray-50
                             disabled:opacity-40 transition-colors"
                >
                  <Undo2 size={13} /> Desfer
                </button>
                <button
                  onClick={() => setRoiPoints([])}
                  disabled={roiPoints.length === 0}
                  className="flex items-center gap-1 text-xs text-gray-600 border
                             border-gray-300 rounded-lg px-3 py-1.5 hover:bg-gray-50
                             disabled:opacity-40 transition-colors"
                >
                  <X size={13} /> Reiniciar
                </button>
              </div>
            </div>

            <button
              onClick={() => setStep('time')}
              disabled={roiPoints.length < 3}
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

            <div className="rounded-xl overflow-hidden bg-black mb-6">
              <video
                ref={videoRef}
                controls
                onLoadedMetadata={handleVideoMetadata}
                className="w-full max-h-72"
              />
            </div>

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
                  {/* Handle inici */}
                  <div
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2
                               w-5 h-5 bg-white border-2 border-blue-600 rounded-full
                               shadow-md cursor-grab active:cursor-grabbing touch-none z-10"
                    style={{ left: `${(startSec / duration) * 100}%` }}
                    onPointerDown={(e) => { e.stopPropagation(); startDrag('start'); }}
                  />
                  {/* Handle fi */}
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
              disabled={saving || duration === 0}
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
