import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, CheckCircle, Loader, RotateCcw, Timer, Undo2, UploadCloud, X } from 'lucide-react';
import { createMatch, getMatch, processMatch, updateMatchConfig, type RoiPoint } from '../api/matches';

type Step = 'upload' | 'roi' | 'time';

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

  // ── Pas 2: ROI — 4 vèrtexs, polígon lliure, arrossegables ────────────────
  const [roiPoints,      setRoiPoints]      = useState<RoiPoint[]>([]);
  // índex del vèrtex que s'està arrossegant (-1 = cap)
  const [draggingVertex, setDraggingVertex] = useState<number>(-1);

  // ── Pas 3: timeline ────────────────────────────────────────────────────────
  const [duration,   setDuration]   = useState(0);
  const [startSec,   setStartSec]   = useState(0);
  const [endSec,     setEndSec]     = useState(0);
  const [dragging,   setDragging]   = useState<'start' | 'end' | null>(null);
  const [videoReady, setVideoReady] = useState(false);

  // ── Submit ─────────────────────────────────────────────────────────────────
  const [saving,            setSaving]            = useState(false);
  const [saveError,         setSaveError]         = useState<string | null>(null);
  const [done,              setDone]              = useState(false);
  const [processStatus,     setProcessStatus]     = useState<string>('processing');
  const [progress,          setProgress]          = useState(0);
  const [processError,      setProcessError]      = useState<string | null>(null);
  const [reprocessing,      setReprocessing]      = useState(false);
  const [pollingStartedAt,  setPollingStartedAt]  = useState<number | null>(null);
  const [stuck,             setStuck]             = useState(false);

  // ── Refs ────────────────────────────────────────────────────────────────────
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const videoRef    = useRef<HTMLVideoElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const blobUrlRef  = useRef('');

  // Refs per evitar closures obsoletes als pointer events globals del timeline
  const draggingRef = useRef<'start' | 'end' | null>(null);
  const startRef    = useRef(0);
  const endRef      = useRef(0);
  const durRef      = useRef(0);

  useEffect(() => { startRef.current = startSec; }, [startSec]);
  useEffect(() => { endRef.current   = endSec;   }, [endSec]);
  useEffect(() => { durRef.current   = duration; }, [duration]);

  useEffect(() => () => {
    if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
  }, []);

  // ── Fix vídeo negre: re-aplica src quan el <video> es remunta (→time) ──────
  useEffect(() => {
    if (step === 'time' && videoRef.current && blobUrlRef.current) {
      setVideoReady(false);
      videoRef.current.src = blobUrlRef.current;
    }
  }, [step]);

  // ── Pointer events globals del timeline ────────────────────────────────────
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

  // ── Canvas ROI ─────────────────────────────────────────────────────────────
  const redrawRoi = useCallback(() => {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video || !video.videoWidth) return;
    const ctx = canvas.getContext('2d')!;

    // Escala canvas → pantalla per mides proporcionals
    const dispW  = canvas.getBoundingClientRect().width || canvas.width;
    const scale  = canvas.width / dispW;
    const vR     = Math.max(8, 10 * scale);   // radi vèrtex
    const lW     = Math.max(1.5, 2 * scale);  // gruix línia
    const fSize  = Math.round(Math.max(11, 13 * scale)); // mida font

    ctx.drawImage(video, 0, 0);
    if (roiPoints.length === 0) return;

    // Polígon (ombreig si 4 punts)
    ctx.beginPath();
    ctx.moveTo(roiPoints[0].x, roiPoints[0].y);
    roiPoints.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
    if (roiPoints.length === 4) {
      ctx.closePath();
      ctx.fillStyle = 'rgba(37, 99, 235, 0.18)';
      ctx.fill();
    }
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth   = lW;
    ctx.setLineDash([7 * scale, 4 * scale]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Vèrtexs numerats i arrossegables
    roiPoints.forEach((p, i) => {
      const active = draggingVertex === i;
      ctx.fillStyle   = active ? '#1d4ed8' : '#2563eb';
      ctx.strokeStyle = 'white';
      ctx.lineWidth   = lW;
      ctx.beginPath();
      ctx.arc(p.x, p.y, vR, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle      = 'white';
      ctx.font           = `bold ${fSize}px sans-serif`;
      ctx.textAlign      = 'center';
      ctx.textBaseline   = 'middle';
      ctx.fillText(String(i + 1), p.x, p.y);
    });
  }, [roiPoints, draggingVertex]);

  // Redibuixa quan canvien punts o vèrtex actiu
  useEffect(() => {
    if (step === 'roi') redrawRoi();
  }, [roiPoints, draggingVertex, step, redrawRoi]);

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

  // ── Coordenades canvas i hit detection ─────────────────────────────────────
  function getCanvasPoint(e: React.PointerEvent<HTMLCanvasElement>): RoiPoint {
    const canvas = canvasRef.current!;
    const rect   = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width  / rect.width),
      y: (e.clientY - rect.top)  * (canvas.height / rect.height),
    };
  }

  function findNearVertex(pt: RoiPoint): number {
    const canvas  = canvasRef.current!;
    const dispW   = canvas.getBoundingClientRect().width || canvas.width;
    const scale   = canvas.width / dispW;
    const hitR    = 22 * scale; // 22 px pantalla → canvas coords
    for (let i = 0; i < roiPoints.length; i++) {
      const dx = roiPoints[i].x - pt.x;
      const dy = roiPoints[i].y - pt.y;
      if (dx * dx + dy * dy < hitR * hitR) return i;
    }
    return -1;
  }

  // ── Handlers ROI canvas ────────────────────────────────────────────────────
  function handleRoiDown(e: React.PointerEvent<HTMLCanvasElement>) {
    const pt  = getCanvasPoint(e);
    const hit = findNearVertex(pt);
    if (hit !== -1) {
      // Arrossega vèrtex existent
      e.currentTarget.setPointerCapture(e.pointerId);
      setDraggingVertex(hit);
    } else if (roiPoints.length < 4) {
      // Afegeix nou vèrtex
      setRoiPoints((prev) => [...prev, pt]);
    }
  }

  function handleRoiMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (draggingVertex === -1) return;
    const pt = getCanvasPoint(e);
    setRoiPoints((prev) => prev.map((p, i) => (i === draggingVertex ? pt : p)));
  }

  function handleRoiUp(e: React.PointerEvent<HTMLCanvasElement>) {
    if (draggingVertex !== -1) {
      e.currentTarget.releasePointerCapture(e.pointerId);
      setDraggingVertex(-1);
    }
  }

  // ── Handlers generals ──────────────────────────────────────────────────────
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
    setUploading(true); setUploadError(null); setUploadProgress(0);
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

  function handleVideoMetadata() {
    const d = videoRef.current!.duration;
    setDuration(d); setStartSec(0); setEndSec(d);
    durRef.current = d; endRef.current = d;
    setVideoReady(true);
  }

  function startTimelineDrag(handle: 'start' | 'end') {
    draggingRef.current = handle; setDragging(handle);
  }

  async function handleSave() {
    setSaving(true); setSaveError(null);
    try {
      await updateMatchConfig(matchId, roiPoints, startSec, endSec);
      await processMatch(matchId);
      setDone(true);
    } catch {
      setSaveError('Error desant la configuració. Torna-ho a intentar.');
    } finally {
      setSaving(false);
    }
  }

  // ── Polling del processament ────────────────────────────────────────────────
  useEffect(() => {
    if (!done || !matchId) return;
    let cancelled = false;
    const startedAt = Date.now();
    setPollingStartedAt(startedAt);
    setStuck(false);

    const progressTimer = setInterval(() => {
      setProgress(prev => (prev < 88 ? prev + 0.4 : prev));
    }, 300);

    async function poll() {
      try {
        const m = await getMatch(matchId);
        if (cancelled) return;
        setProcessStatus(m.status);
        if (m.status === 'done') {
          clearInterval(progressTimer);
          setProgress(100);
          setTimeout(() => { if (!cancelled) navigate(`/matches/${matchId}/results`); }, 700);
        } else if (m.status === 'error') {
          clearInterval(progressTimer);
          setProcessError('El processament ha fallat. Revisa la configuració del partit i torna a intentar-ho.');
        } else if (Date.now() - startedAt > 5 * 60 * 1000) {
          setStuck(true);
        }
      } catch { /* ignora errors de xarxa puntuals */ }
    }

    poll();
    const pollTimer = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(progressTimer);
      clearInterval(pollTimer);
    };
  }, [done, matchId, navigate]);

  async function handleReprocess() {
    if (!matchId) return;
    setReprocessing(true);
    setProcessError(null);
    setStuck(false);
    try {
      await processMatch(matchId);
      setProcessStatus('processing');
      setProgress(0);
      setDone(false);
      setTimeout(() => setDone(true), 100);
    } catch {
      setProcessError("No s'ha pogut reiniciar el processament. Torna-ho a intentar.");
    } finally {
      setReprocessing(false);
    }
  }

  function goBack() {
    const idx = STEP_ORDER.indexOf(step);
    if (idx === 0) navigate('/');
    else setStep(STEP_ORDER[idx - 1]);
  }

  const currentIdx = STEP_ORDER.indexOf(step);

  // ── Processament en curs (polling) ─────────────────────────────────────────
  if (done) {
    const STATUS_LABEL: Record<string, string> = {
      processing:   'Processant el vídeo i extraient frames...',
      frames_ready: 'Analitzant jugadors per frame...',
      done:         'Completat!',
      error:        'El processament ha fallat.',
    };
    const label = STATUS_LABEL[processStatus] ?? 'Processant...';
    const isDone  = processStatus === 'done';
    const isError = processStatus === 'error' || !!processError;
    const isStuck = stuck && !isError && !isDone;

    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8
                        max-w-sm w-full text-center">

          <div className={`rounded-full p-4 w-fit mx-auto mb-5
            ${isError ? 'bg-red-100 text-red-500'
              : isDone ? 'bg-green-100 text-green-600'
              : 'bg-blue-50 text-blue-600'}`}>
            {isError
              ? <AlertCircle size={28} />
              : isDone
                ? <CheckCircle size={28} />
                : <Loader size={28} className="animate-spin" />
            }
          </div>

          <h2 className="text-base font-semibold text-gray-900 mb-1">
            {isError ? 'Error en el processament' : isDone ? 'Anàlisi completada' : 'Processant el partit'}
          </h2>
          <p className="text-sm text-gray-500 mb-5">{processError ?? label}</p>

          {!isError && (
            <div className="mb-5">
              <div className="flex justify-between text-xs text-gray-400 mb-1.5">
                <span>{Math.round(progress)}%</span>
                <span className="capitalize">{processStatus}</span>
              </div>
              <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300
                    ${isDone ? 'bg-green-500' : 'bg-blue-600'}`}
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {isError && (
            <div className="space-y-2">
              <button
                onClick={handleReprocess}
                disabled={reprocessing}
                className="w-full flex items-center justify-center gap-2
                           bg-orange-500 hover:bg-orange-600 disabled:opacity-60
                           text-white font-medium py-2.5 rounded-lg text-sm transition-colors"
              >
                {reprocessing
                  ? <Loader size={15} className="animate-spin" />
                  : <RotateCcw size={15} />
                }
                {reprocessing ? 'Reiniciant...' : 'Reprocessar'}
              </button>
              <button
                onClick={() => navigate('/matches')}
                className="w-full text-sm text-gray-500 hover:text-gray-700
                           py-2 transition-colors"
              >
                Veure partits
              </button>
            </div>
          )}

          {isStuck && (
            <div className="mt-1 space-y-2">
              <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200
                             rounded-lg px-3 py-2">
                El processament triga més del previst. Potser s'ha enganxat.
              </p>
              <button
                onClick={handleReprocess}
                disabled={reprocessing}
                className="w-full flex items-center justify-center gap-2
                           border border-orange-300 hover:bg-orange-50 disabled:opacity-60
                           text-orange-600 font-medium py-2 rounded-lg text-sm transition-colors"
              >
                {reprocessing
                  ? <Loader size={14} className="animate-spin" />
                  : <RotateCcw size={14} />
                }
                {reprocessing ? 'Reiniciant...' : 'Forçar reprocessat'}
              </button>
            </div>
          )}

          {!isError && !isDone && !isStuck && (
            <p className="text-xs text-gray-400">Això pot trigar uns minuts</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 px-4 py-8">
      {/* Vídeo ocult durant passos 'upload' i 'roi' (en memòria, usat per canvas) */}
      {step !== 'time' && (
        <video ref={videoRef} onLoadedMetadata={handleVideoMetadata} className="hidden" />
      )}

      <div className="max-w-2xl mx-auto">

        {/* Indicador de passos */}
        <div className="flex items-center justify-between mb-8">
          <button onClick={goBack}
            className="text-sm text-gray-500 hover:text-gray-700 transition-colors">
            ← Tornar
          </button>
          <div className="flex items-center gap-1.5">
            {STEP_LABELS.map((s, i) => (
              <div key={s.key} className="flex items-center gap-1.5">
                <div className="flex items-center gap-1">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center
                                   text-xs font-semibold transition-colors
                    ${i < currentIdx  ? 'bg-blue-600 text-white'
                      : i === currentIdx ? 'bg-blue-100 text-blue-600 border-2 border-blue-500'
                      : 'bg-gray-100 text-gray-400'}`}>
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

            <label className="block mb-4 cursor-pointer"
                   onDragOver={(e) => e.preventDefault()} onDrop={handleDrop}>
              <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors
                ${file ? 'border-blue-300 bg-blue-50' : 'border-gray-300 hover:border-blue-300 hover:bg-gray-50'}`}>
                <UploadCloud size={32}
                  className={`mx-auto mb-2 ${file ? 'text-blue-500' : 'text-gray-400'}`} />
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
              <input type="file" accept="video/mp4,video/*" className="hidden"
                     onChange={handleFileChange} disabled={uploading} />
            </label>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Títol del partit
              </label>
              <input type="text" value={title} onChange={(e) => setTitle(e.target.value)}
                placeholder="Ex: Lliga J12 vs Joventut" disabled={uploading}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm
                           focus:outline-none focus:ring-2 focus:ring-blue-500
                           disabled:bg-gray-50 disabled:text-gray-400" />
            </div>

            {uploading && (
              <div className="mb-4">
                <div className="flex justify-between text-xs text-gray-500 mb-1.5">
                  <span>Pujant al servidor...</span>
                  <span className="font-medium">{uploadProgress}%</span>
                </div>
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-600 rounded-full transition-all duration-200"
                       style={{ width: `${uploadProgress}%` }} />
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

            <button onClick={handleUpload}
              disabled={!file || !title.trim() || uploading}
              className="w-full flex items-center justify-center gap-2 bg-blue-600
                         hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium
                         py-2.5 rounded-lg text-sm transition-colors">
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
            <p className="text-xs text-gray-500 mb-1">
              Fes clic per afegir els 4 vèrtexs del polígon. Un cop col·locats, arrossega'ls per ajustar.
            </p>
            <p className="text-xs text-blue-500 mb-4">
              Consell: selecciona les 4 cantonades del camp de joc en ordre.
            </p>

            <div className="rounded-xl overflow-hidden bg-black mb-4">
              <canvas
                ref={canvasRef}
                onPointerDown={handleRoiDown}
                onPointerMove={handleRoiMove}
                onPointerUp={handleRoiUp}
                className={`w-full block select-none
                  ${draggingVertex !== -1 ? 'cursor-grabbing' : 'cursor-crosshair'}`}
              />
            </div>

            <div className="flex items-center justify-between mb-6">
              <span className="text-sm">
                {roiPoints.length === 4
                  ? <span className="text-green-600 font-medium">✓ 4 vèrtexs definits</span>
                  : <span className="text-gray-500">
                      {roiPoints.length}/4 vèrtexs
                      {roiPoints.length < 4 && (
                        <span className="text-gray-400 ml-1">
                          — fes clic per afegir el vèrtex {roiPoints.length + 1}
                        </span>
                      )}
                    </span>
                }
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setRoiPoints((p) => p.slice(0, -1))}
                  disabled={roiPoints.length === 0}
                  className="flex items-center gap-1 text-xs text-gray-600 border border-gray-300
                             rounded-lg px-3 py-1.5 hover:bg-gray-50 disabled:opacity-40 transition-colors">
                  <Undo2 size={13} /> Desfer
                </button>
                <button
                  onClick={() => setRoiPoints([])}
                  disabled={roiPoints.length === 0}
                  className="flex items-center gap-1 text-xs text-gray-600 border border-gray-300
                             rounded-lg px-3 py-1.5 hover:bg-gray-50 disabled:opacity-40 transition-colors">
                  <X size={13} /> Reiniciar
                </button>
              </div>
            </div>

            <button onClick={() => setStep('time')}
              disabled={roiPoints.length < 4}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300
                         text-white font-medium py-2.5 rounded-lg text-sm transition-colors">
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

            {duration > 0 && (
              <div className="mb-6 space-y-3">
                <div
                  ref={timelineRef}
                  className={`relative h-6 bg-gray-200 rounded-full select-none
                    ${dragging ? 'cursor-grabbing' : ''}`}
                >
                  <div className="absolute h-full bg-blue-500 rounded-full pointer-events-none"
                    style={{
                      left:  `${(startSec / duration) * 100}%`,
                      width: `${((endSec - startSec) / duration) * 100}%`,
                    }} />
                  <div
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2
                               w-5 h-5 bg-white border-2 border-blue-600 rounded-full
                               shadow-md cursor-grab active:cursor-grabbing touch-none z-10"
                    style={{ left: `${(startSec / duration) * 100}%` }}
                    onPointerDown={(e) => { e.stopPropagation(); startTimelineDrag('start'); }}
                  />
                  <div
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2
                               w-5 h-5 bg-white border-2 border-blue-600 rounded-full
                               shadow-md cursor-grab active:cursor-grabbing touch-none z-10"
                    style={{ left: `${(endSec / duration) * 100}%` }}
                    onPointerDown={(e) => { e.stopPropagation(); startTimelineDrag('end'); }}
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

            <button onClick={handleSave}
              disabled={saving || !videoReady}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300
                         text-white font-medium py-2.5 rounded-lg text-sm transition-colors">
              {saving ? 'Desant...' : 'Desa configuració →'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
