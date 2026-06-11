import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, CheckCircle, Loader, RotateCcw, Undo2, UploadCloud, X } from 'lucide-react';
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

  const [file,           setFile]           = useState<File | null>(null);
  const [title,          setTitle]          = useState('');
  const [uploading,      setUploading]      = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError,    setUploadError]    = useState<string | null>(null);

  const [step,    setStep]    = useState<Step>('upload');
  const [matchId, setMatchId] = useState('');

  const [roiPoints,      setRoiPoints]      = useState<RoiPoint[]>([]);
  const [draggingVertex, setDraggingVertex] = useState<number>(-1);

  const [duration,   setDuration]   = useState(0);
  const [startSec,   setStartSec]   = useState(0);
  const [endSec,     setEndSec]     = useState(0);
  const [dragging,   setDragging]   = useState<'start' | 'end' | null>(null);
  const [videoReady, setVideoReady] = useState(false);

  const [saving,           setSaving]           = useState(false);
  const [saveError,        setSaveError]        = useState<string | null>(null);
  const [done,             setDone]             = useState(false);
  const [processStatus,    setProcessStatus]    = useState<string>('processing');
  const [progress,         setProgress]         = useState(0);
  const [processError,     setProcessError]     = useState<string | null>(null);
  const [reprocessing,     setReprocessing]     = useState(false);
  const [pollingStartedAt, setPollingStartedAt] = useState<number | null>(null);
  const [stuck,            setStuck]            = useState(false);

  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const videoRef    = useRef<HTMLVideoElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const blobUrlRef  = useRef('');

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

  useEffect(() => {
    if (step === 'time' && videoRef.current && blobUrlRef.current) {
      setVideoReady(false);
      videoRef.current.src = blobUrlRef.current;
    }
  }, [step]);

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

  const redrawRoi = useCallback(() => {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video || !video.videoWidth) return;
    const ctx   = canvas.getContext('2d')!;
    const dispW = canvas.getBoundingClientRect().width || canvas.width;
    const scale = canvas.width / dispW;
    const vR    = Math.max(8, 10 * scale);
    const lW    = Math.max(1.5, 2 * scale);
    const fSize = Math.round(Math.max(11, 13 * scale));

    ctx.drawImage(video, 0, 0);
    if (roiPoints.length === 0) return;

    ctx.beginPath();
    ctx.moveTo(roiPoints[0].x, roiPoints[0].y);
    roiPoints.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
    if (roiPoints.length === 4) {
      ctx.closePath();
      ctx.fillStyle = 'rgba(0,255,65,0.15)';
      ctx.fill();
    }
    ctx.strokeStyle = '#00ff41';
    ctx.lineWidth   = lW;
    ctx.setLineDash([7 * scale, 4 * scale]);
    ctx.stroke();
    ctx.setLineDash([]);

    roiPoints.forEach((p, i) => {
      const active = draggingVertex === i;
      ctx.fillStyle   = active ? '#00cc33' : '#00ff41';
      ctx.strokeStyle = '#080e08';
      ctx.lineWidth   = lW;
      ctx.beginPath();
      ctx.arc(p.x, p.y, vR, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle    = '#080e08';
      ctx.font         = `bold ${fSize}px monospace`;
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(i + 1), p.x, p.y);
    });
  }, [roiPoints, draggingVertex]);

  useEffect(() => {
    if (step === 'roi') redrawRoi();
  }, [roiPoints, draggingVertex, step, redrawRoi]);

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

  function getCanvasPoint(e: React.PointerEvent<HTMLCanvasElement>): RoiPoint {
    const canvas = canvasRef.current!;
    const rect   = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width  / rect.width),
      y: (e.clientY - rect.top)  * (canvas.height / rect.height),
    };
  }

  function findNearVertex(pt: RoiPoint): number {
    const canvas = canvasRef.current!;
    const dispW  = canvas.getBoundingClientRect().width || canvas.width;
    const scale  = canvas.width / dispW;
    const hitR   = 22 * scale;
    for (let i = 0; i < roiPoints.length; i++) {
      const dx = roiPoints[i].x - pt.x;
      const dy = roiPoints[i].y - pt.y;
      if (dx * dx + dy * dy < hitR * hitR) return i;
    }
    return -1;
  }

  function handleRoiDown(e: React.PointerEvent<HTMLCanvasElement>) {
    const pt  = getCanvasPoint(e);
    const hit = findNearVertex(pt);
    if (hit !== -1) {
      e.currentTarget.setPointerCapture(e.pointerId);
      setDraggingVertex(hit);
    } else if (roiPoints.length < 4) {
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
          setProcessError('El processament ha fallat. Revisa la configuració i torna a intentar-ho.');
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

  // ── Pantalla de processament ─────────────────────────────────────────────
  if (done) {
    const STATUS_LABEL: Record<string, string> = {
      processing:   'Processant el vídeo i extraient frames...',
      frames_ready: 'Analitzant jugadors per frame...',
      done:         'Anàlisi completada!',
      error:        'El processament ha fallat.',
    };
    const label   = STATUS_LABEL[processStatus] ?? 'Processant...';
    const isDone  = processStatus === 'done';
    const isError = processStatus === 'error' || !!processError;
    const isStuck = stuck && !isError && !isDone;

    return (
      <div className="min-h-screen bg-matrix-bg flex items-center justify-center px-4">
        <div className="bg-matrix-card border border-matrix-border rounded-lg p-8 max-w-sm w-full text-center">

          <div className={`rounded-full p-4 w-fit mx-auto mb-5
            ${isError ? 'bg-[#1a0d0d] text-matrix-error'
              : isDone ? 'bg-matrix-raised text-neon'
              : 'bg-matrix-raised text-neon'}`}>
            {isError
              ? <AlertCircle size={28} />
              : isDone
                ? <CheckCircle size={28} />
                : <Loader size={28} className="animate-spin" />
            }
          </div>

          <h2 className="text-xs font-bold text-neon uppercase tracking-widest mb-1">
            {isError ? 'Error en el processament' : isDone ? 'Anàlisi completada' : 'Processant el partit'}
          </h2>
          <p className="text-xs text-matrix-text mb-5">{processError ?? label}</p>

          {!isError && (
            <div className="mb-5">
              <div className="flex justify-between text-xs text-matrix-muted mb-1.5">
                <span>{Math.round(progress)}%</span>
                <span className="text-matrix-info uppercase tracking-widest">{processStatus}</span>
              </div>
              <div className="h-1.5 bg-matrix-raised rounded-full overflow-hidden">
                <div
                  className="h-full bg-neon rounded-full transition-all duration-300"
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
                           bg-matrix-warning hover:bg-[#cc9200] disabled:opacity-50
                           text-matrix-bg font-bold py-2.5 rounded text-xs
                           uppercase tracking-widest transition-colors"
              >
                {reprocessing ? <Loader size={14} className="animate-spin" /> : <RotateCcw size={14} />}
                {reprocessing ? 'Reiniciant...' : 'Reprocessar'}
              </button>
              <button
                onClick={() => navigate('/matches')}
                className="w-full text-xs text-matrix-muted hover:text-neon py-2
                           uppercase tracking-widest transition-colors"
              >
                Veure partits
              </button>
            </div>
          )}

          {isStuck && (
            <div className="mt-1 space-y-2">
              <p className="text-xs text-matrix-warning bg-[#1a1a0d] border border-[#3a3a1a] rounded px-3 py-2">
                El processament triga més del previst.
              </p>
              <button
                onClick={handleReprocess}
                disabled={reprocessing}
                className="w-full flex items-center justify-center gap-2
                           border border-matrix-warning hover:bg-[#1a1a0d] disabled:opacity-50
                           text-matrix-warning font-bold py-2 rounded text-xs
                           uppercase tracking-widest transition-colors"
              >
                {reprocessing ? <Loader size={13} className="animate-spin" /> : <RotateCcw size={13} />}
                {reprocessing ? 'Reiniciant...' : 'Forçar reprocessat'}
              </button>
            </div>
          )}

          {!isError && !isDone && !isStuck && (
            <p className="text-xs text-matrix-muted uppercase tracking-widest">Pot trigar uns minuts</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-matrix-bg px-4 py-8">
      {step !== 'time' && (
        <video ref={videoRef} onLoadedMetadata={handleVideoMetadata} className="hidden" />
      )}

      <div className="max-w-2xl mx-auto">

        {/* Indicador de passos */}
        <div className="flex items-center justify-between mb-8">
          <button onClick={goBack}
            className="text-xs text-matrix-muted hover:text-neon uppercase tracking-widest transition-colors">
            ← Tornar
          </button>
          <div className="flex items-center gap-1.5">
            {STEP_LABELS.map((s, i) => (
              <div key={s.key} className="flex items-center gap-1.5">
                <div className="flex items-center gap-1">
                  <div className={`w-6 h-6 rounded flex items-center justify-center
                                   text-xs font-bold transition-colors
                    ${i < currentIdx  ? 'bg-neon text-matrix-bg'
                      : i === currentIdx ? 'bg-matrix-raised text-neon border border-neon'
                      : 'bg-matrix-raised text-matrix-muted border border-matrix-border'}`}>
                    {i + 1}
                  </div>
                  <span className={`text-xs hidden sm:block uppercase tracking-widest
                    ${i <= currentIdx ? 'text-neon' : 'text-matrix-muted'}`}>
                    {s.label}
                  </span>
                </div>
                {i < STEP_LABELS.length - 1 && (
                  <div className={`w-5 h-px ${i < currentIdx ? 'bg-neon' : 'bg-matrix-border'}`} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* ═══════ PAS 1: UPLOAD ═══════ */}
        {step === 'upload' && (
          <div className="bg-matrix-card border border-matrix-border rounded-lg p-8">
            <div className="flex items-center gap-3 mb-6">
              <img src="/logo.png" alt="" className="h-16 w-auto object-contain" />
              <div>
                <h1 className="text-sm font-bold text-neon uppercase tracking-widest">
                  01 / Puja el vídeo
                </h1>
                <p className="text-xs text-matrix-muted">Fitxer MP4 del vídeo complet</p>
              </div>
            </div>

            <label className="block mb-4 cursor-pointer"
                   onDragOver={(e) => e.preventDefault()} onDrop={handleDrop}>
              <div className={`border-2 border-dashed rounded p-8 text-center transition-colors
                ${file
                  ? 'border-neon bg-matrix-raised'
                  : 'border-matrix-border hover:border-neon hover:bg-matrix-raised'}`}>
                <UploadCloud size={30}
                  className={`mx-auto mb-2 ${file ? 'text-neon' : 'text-matrix-muted'}`} />
                {file ? (
                  <>
                    <p className="text-xs font-bold text-neon uppercase">{file.name}</p>
                    <p className="text-xs text-matrix-muted mt-0.5">
                      {(file.size / 1024 / 1024).toFixed(1)} MB
                    </p>
                  </>
                ) : (
                  <>
                    <p className="text-xs text-matrix-text">Arrossega el fitxer .mp4 aquí</p>
                    <p className="text-xs text-matrix-muted mt-0.5">o fes clic per seleccionar</p>
                  </>
                )}
              </div>
              <input type="file" accept="video/mp4,video/*" className="hidden"
                     onChange={handleFileChange} disabled={uploading} />
            </label>

            <div className="mb-4">
              <label className="block text-xs font-medium text-matrix-muted uppercase tracking-wider mb-1.5">
                Títol del partit
              </label>
              <input type="text" value={title} onChange={(e) => setTitle(e.target.value)}
                placeholder="Ex: Lliga J12 vs Joventut" disabled={uploading}
                className="w-full px-3 py-2.5 bg-matrix-input border border-matrix-border rounded
                           text-sm text-matrix-text placeholder-matrix-muted
                           focus:outline-none focus:border-neon focus:ring-1 focus:ring-neon
                           disabled:opacity-40 transition-colors" />
            </div>

            {uploading && (
              <div className="mb-4">
                <div className="flex justify-between text-xs text-matrix-muted mb-1.5">
                  <span>Pujant al servidor...</span>
                  <span className="font-bold text-neon">{uploadProgress}%</span>
                </div>
                <div className="h-1.5 bg-matrix-raised rounded-full overflow-hidden">
                  <div className="h-full bg-neon rounded-full transition-all duration-200"
                       style={{ width: `${uploadProgress}%` }} />
                </div>
              </div>
            )}

            {uploadError && (
              <div className="flex items-start gap-2 text-matrix-error text-xs mb-4
                              bg-[#1a0d0d] border border-[#3a1a1a] rounded px-3 py-2">
                <AlertCircle size={14} className="mt-0.5 shrink-0" />
                <span>{uploadError}</span>
              </div>
            )}

            <button onClick={handleUpload}
              disabled={!file || !title.trim() || uploading}
              className="w-full flex items-center justify-center gap-2 bg-neon
                         hover:bg-neon-600 disabled:opacity-40 text-matrix-bg font-bold
                         py-2.5 rounded text-xs uppercase tracking-widest transition-colors">
              <UploadCloud size={14} />
              {uploading ? 'Pujant...' : 'Puja el vídeo →'}
            </button>
          </div>
        )}

        {/* ═══════ PAS 2: ROI ═══════ */}
        {step === 'roi' && (
          <div className="bg-matrix-card border border-matrix-border rounded-lg p-6">
            <h1 className="text-sm font-bold text-neon uppercase tracking-widest mb-1">
              02 / Zona de joc (ROI)
            </h1>
            <p className="text-xs text-matrix-text mb-1">
              Fes clic per afegir els 4 vèrtexs del polígon. Un cop col·locats, arrossega'ls per ajustar.
            </p>
            <p className="text-xs text-neon mb-4">
              {'>'} Selecciona les 4 cantonades del camp en ordre
            </p>

            <div className="rounded overflow-hidden bg-black mb-4 border border-matrix-border">
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
              <span className="text-xs">
                {roiPoints.length === 4
                  ? <span className="text-neon font-bold uppercase tracking-widest">✓ 4 vèrtexs definits</span>
                  : <span className="text-matrix-muted">
                      {roiPoints.length}/4 vèrtexs
                      {roiPoints.length < 4 && (
                        <span className="ml-1">
                          — afegeix el vèrtex {roiPoints.length + 1}
                        </span>
                      )}
                    </span>
                }
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setRoiPoints((p) => p.slice(0, -1))}
                  disabled={roiPoints.length === 0}
                  className="flex items-center gap-1 text-xs text-matrix-muted border border-matrix-border
                             rounded px-3 py-1.5 hover:border-neon hover:text-neon
                             disabled:opacity-40 transition-colors uppercase tracking-widest">
                  <Undo2 size={12} /> Desfer
                </button>
                <button
                  onClick={() => setRoiPoints([])}
                  disabled={roiPoints.length === 0}
                  className="flex items-center gap-1 text-xs text-matrix-muted border border-matrix-border
                             rounded px-3 py-1.5 hover:border-matrix-error hover:text-matrix-error
                             disabled:opacity-40 transition-colors uppercase tracking-widest">
                  <X size={12} /> Reset
                </button>
              </div>
            </div>

            <button onClick={() => setStep('time')}
              disabled={roiPoints.length < 4}
              className="w-full bg-neon hover:bg-neon-600 disabled:opacity-40
                         text-matrix-bg font-bold py-2.5 rounded text-xs
                         uppercase tracking-widest transition-colors">
              Continuar →
            </button>
          </div>
        )}

        {/* ═══════ PAS 3: TIMELINE ═══════ */}
        {step === 'time' && (
          <div className="bg-matrix-card border border-matrix-border rounded-lg p-6">
            <h1 className="text-sm font-bold text-neon uppercase tracking-widest mb-1">
              03 / Selecciona el temps
            </h1>
            <p className="text-xs text-matrix-muted mb-4">
              Arrossega els extrems per definir l'inici i el final del temps jugat
            </p>

            <div className="rounded overflow-hidden bg-black mb-4 border border-matrix-border">
              {!videoReady && (
                <div className="h-40 flex items-center justify-center">
                  <p className="text-matrix-muted text-xs uppercase tracking-widest">
                    <Loader size={14} className="inline mr-2 animate-spin" />
                    Carregant vídeo...
                  </p>
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
                  className={`relative h-5 bg-matrix-raised rounded-full select-none
                    border border-matrix-border ${dragging ? 'cursor-grabbing' : ''}`}
                >
                  <div className="absolute h-full bg-neon bg-opacity-20 rounded-full pointer-events-none"
                    style={{
                      left:  `${(startSec / duration) * 100}%`,
                      width: `${((endSec - startSec) / duration) * 100}%`,
                    }} />
                  <div className="absolute top-0 bottom-0 bg-neon opacity-60 rounded-full pointer-events-none"
                    style={{
                      left:  `${(startSec / duration) * 100}%`,
                      width: `${((endSec - startSec) / duration) * 100}%`,
                    }} />
                  <div
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2
                               w-4 h-4 bg-matrix-bg border-2 border-neon rounded-full
                               shadow-md cursor-grab active:cursor-grabbing touch-none z-10"
                    style={{ left: `${(startSec / duration) * 100}%` }}
                    onPointerDown={(e) => { e.stopPropagation(); startTimelineDrag('start'); }}
                  />
                  <div
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2
                               w-4 h-4 bg-matrix-bg border-2 border-neon rounded-full
                               shadow-md cursor-grab active:cursor-grabbing touch-none z-10"
                    style={{ left: `${(endSec / duration) * 100}%` }}
                    onPointerDown={(e) => { e.stopPropagation(); startTimelineDrag('end'); }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs">
                  <div className="text-center">
                    <p className="text-matrix-muted uppercase tracking-widest text-xs mb-0.5">Inici</p>
                    <p className="font-bold text-neon">{formatTime(startSec)}</p>
                  </div>
                  <span className="text-matrix-muted">{formatTime(endSec - startSec)} de joc</span>
                  <div className="text-center">
                    <p className="text-matrix-muted uppercase tracking-widest text-xs mb-0.5">Fi</p>
                    <p className="font-bold text-neon">{formatTime(endSec)}</p>
                  </div>
                </div>
              </div>
            )}

            {saveError && (
              <div className="flex items-start gap-2 text-matrix-error text-xs mb-4
                              bg-[#1a0d0d] border border-[#3a1a1a] rounded px-3 py-2">
                <AlertCircle size={14} className="mt-0.5 shrink-0" />
                <span>{saveError}</span>
              </div>
            )}

            <button onClick={handleSave}
              disabled={saving || !videoReady}
              className="w-full bg-neon hover:bg-neon-600 disabled:opacity-40
                         text-matrix-bg font-bold py-2.5 rounded text-xs
                         uppercase tracking-widest transition-colors">
              {saving ? 'Desant...' : 'Iniciar processament →'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
