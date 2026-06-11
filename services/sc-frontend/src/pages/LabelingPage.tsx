import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
  type MouseEvent,
} from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  CheckCircle,
  ExternalLink,
  Loader,
  Play,
  RefreshCw,
  Trash2,
  Upload,
} from 'lucide-react';
import apiClient from '../api/client';
import {
  clearLsTasks,
  getLabelingFrame,
  getLsStats,
  startLabeling,
  uploadLabelingVideo,
  type LabelingUploadResponse,
  type LsStatsResponse,
} from '../api/labeling';
import { rgbToHsv, rgbToCss } from '../utils/color';

const REPRESENTATIVE_PCTS = [0.10, 0.25, 0.50, 0.75, 0.90];
const MAX_RETRIES = 50;
const LENS_SIZE = 88;
const LENS_ZOOM = 4;
const LENS_CROP = LENS_SIZE / LENS_ZOOM;

export default function LabelingPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lensCanvasRef = useRef<HTMLCanvasElement>(null);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lsPollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [frameInterval, setFrameInterval] = useState(2);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<LabelingUploadResponse | null>(() => {
    try {
      const saved = sessionStorage.getItem('labeling_upload_result');
      return saved ? (JSON.parse(saved) as LabelingUploadResponse) : null;
    } catch { return null; }
  });
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [resumeInput, setResumeInput] = useState('');
  const [resumeError, setResumeError] = useState<string | null>(null);

  const [totalFrames, setTotalFrames] = useState(0);
  const [frameIndex, setFrameIndex] = useState(0);
  const [frameLoading, setFrameLoading] = useState(false);
  const [frameError, setFrameError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [selectedRgb, setSelectedRgb] = useState<[number, number, number] | null>(null);
  const [selectedHsv, setSelectedHsv] = useState<[number, number, number] | null>(null);
  const [jerseyThreshold, setJerseyThreshold] = useState(30);

  const [lensVisible, setLensVisible] = useState(false);
  const [lensPos, setLensPos] = useState({ x: 0, y: 0 });
  const [hoverRgb, setHoverRgb] = useState<[number, number, number]>([0, 0, 0]);

  const [starting, setStarting] = useState(false);
  const [startResult, setStartResult] = useState<{ frames_queued: number } | null>(null);
  const [startError, setStartError] = useState<string | null>(null);

  const [lsStats, setLsStats] = useState<LsStatsResponse | null>(null);
  const [lsStatsLoading, setLsStatsLoading] = useState(false);
  const [clearing, setClearing] = useState(false);

  const labelStudioUrl =
    (import.meta.env.VITE_LABEL_STUDIO_URL as string | undefined) ?? 'http://localhost:8081';

  const representativeFrameNumbers = REPRESENTATIVE_PCTS.map(
    (pct) => Math.max(1, Math.round(pct * totalFrames)),
  );

  const loadFrame = useCallback(
    async (sessionId: string, frameNumber: number, retry = 0) => {
      if (!canvasRef.current) return;
      setFrameLoading(true);
      setFrameError(null);
      setRetryCount(retry);

      try {
        const { frame_url, total_frames } = await getLabelingFrame(sessionId, frameNumber);

        if (total_frames === 0) {
          if (retry < MAX_RETRIES) {
            setFrameLoading(false);
            retryTimerRef.current = setTimeout(
              () => loadFrame(sessionId, frameNumber, retry + 1),
              3000,
            );
            return;
          }
          setFrameError("No s'han trobat frames. El vídeo no s'ha pogut processar.");
          setFrameLoading(false);
          return;
        }
        setTotalFrames(total_frames);

        const imgResp = await apiClient.get<Blob>(frame_url, { responseType: 'blob' });
        const objectUrl = URL.createObjectURL(imgResp.data);
        const img = new Image();
        img.onload = () => {
          const canvas = canvasRef.current;
          if (!canvas) return;
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          canvas.getContext('2d')?.drawImage(img, 0, 0);
          URL.revokeObjectURL(objectUrl);
        };
        img.onerror = () => { URL.revokeObjectURL(objectUrl); setFrameError('Error renderitzant el frame.'); };
        img.src = objectUrl;
      } catch {
        setFrameError("No s'ha pogut carregar el frame.");
      } finally {
        setFrameLoading(false);
      }
    },
    [],
  );

  useEffect(() => () => { if (retryTimerRef.current) clearTimeout(retryTimerRef.current); }, []);
  useEffect(() => () => { if (lsPollingRef.current) clearInterval(lsPollingRef.current); }, []);

  useEffect(() => {
    if (uploadResult) {
      sessionStorage.setItem('labeling_upload_result', JSON.stringify(uploadResult));
    } else {
      sessionStorage.removeItem('labeling_upload_result');
    }
  }, [uploadResult]);

  useEffect(() => {
    if (!uploadResult) return;
    if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
    loadFrame(uploadResult.session_id, representativeFrameNumbers[0] || 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadResult]);

  useEffect(() => {
    if (!uploadResult || totalFrames === 0) return;
    loadFrame(uploadResult.session_id, representativeFrameNumbers[frameIndex]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameIndex, totalFrames]);

  useEffect(() => {
    if (!startResult) return;
    const fetchStats = async () => {
      setLsStatsLoading(true);
      const stats = await getLsStats();
      setLsStats(stats);
      setLsStatsLoading(false);
      if (stats.total_tasks > 0 && stats.annotated_tasks >= stats.total_tasks) {
        if (lsPollingRef.current) clearInterval(lsPollingRef.current);
      }
    };
    fetchStats();
    lsPollingRef.current = setInterval(fetchStats, 5000);
    return () => { if (lsPollingRef.current) clearInterval(lsPollingRef.current); };
  }, [startResult]);

  useEffect(() => {
    getLsStats().then(setLsStats).catch(() => null);
  }, []);

  function handleResumeSession() {
    const id = resumeInput.trim();
    if (!id) { setResumeError('Introdueix un session ID vàlid.'); return; }
    setResumeError(null);
    setUploadResult({ session_id: id });
    setResumeInput('');
    setStartResult(null);
    setSelectedHsv(null);
    setSelectedRgb(null);
    setTotalFrames(0);
    setFrameIndex(0);
  }

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0] ?? null;
    if (selected && selected.type !== 'video/mp4') {
      setUploadError("Només s'accepten fitxers .mp4");
      setFile(null);
      return;
    }
    setUploadError(null);
    setUploadResult(null);
    setFile(selected);
  }

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    setUploadResult(null);
    setStartResult(null);
    setSelectedHsv(null);
    setSelectedRgb(null);
    setTotalFrames(0);
    setFrameIndex(0);
    setRetryCount(0);
    try {
      const response = await uploadLabelingVideo(file, frameInterval);
      setUploadResult(response);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setUploadError(detail ?? 'Error en pujar el vídeo. Torna-ho a intentar.');
    } finally {
      setUploading(false);
    }
  }

  function samplePixelAvg(canvas: HTMLCanvasElement, cx: number, cy: number): [number, number, number] {
    const ctx = canvas.getContext('2d');
    if (!ctx) return [0, 0, 0];
    const half = Math.floor(LENS_CROP / 2);
    const sx = Math.max(0, Math.min(cx - half, canvas.width - LENS_CROP));
    const sy = Math.max(0, Math.min(cy - half, canvas.height - LENS_CROP));
    const data = ctx.getImageData(sx, sy, LENS_CROP, LENS_CROP).data;
    let r = 0, g = 0, b = 0;
    const count = data.length / 4;
    for (let i = 0; i < data.length; i += 4) { r += data[i]; g += data[i + 1]; b += data[i + 2]; }
    return [Math.round(r / count), Math.round(g / count), Math.round(b / count)];
  }

  function drawLens(canvas: HTMLCanvasElement, cx: number, cy: number) {
    const lensCanvas = lensCanvasRef.current;
    if (!lensCanvas) return;
    const lctx = lensCanvas.getContext('2d');
    if (!lctx) return;
    const half = LENS_CROP / 2;
    const sx = Math.max(0, Math.min(cx - half, canvas.width - LENS_CROP));
    const sy = Math.max(0, Math.min(cy - half, canvas.height - LENS_CROP));
    lctx.clearRect(0, 0, LENS_SIZE, LENS_SIZE);
    lctx.drawImage(canvas, sx, sy, LENS_CROP, LENS_CROP, 0, 0, LENS_SIZE, LENS_SIZE);
    lctx.strokeStyle = 'rgba(0,255,65,0.9)';
    lctx.lineWidth = 1;
    const mid = LENS_SIZE / 2;
    lctx.beginPath(); lctx.moveTo(mid - 6, mid); lctx.lineTo(mid + 6, mid); lctx.stroke();
    lctx.beginPath(); lctx.moveTo(mid, mid - 6); lctx.lineTo(mid, mid + 6); lctx.stroke();
  }

  function handleCanvasMouseMove(e: MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const cx = Math.floor((e.clientX - rect.left) * scaleX);
    const cy = Math.floor((e.clientY - rect.top) * scaleY);

    const rgb = samplePixelAvg(canvas, cx, cy);
    setHoverRgb(rgb);
    drawLens(canvas, cx, cy);

    const containerRect = (canvas.parentElement as HTMLElement).getBoundingClientRect();
    setLensPos({
      x: e.clientX - containerRect.left,
      y: e.clientY - containerRect.top,
    });
    setLensVisible(true);
  }

  function handleCanvasMouseLeave() {
    setLensVisible(false);
  }

  function handleCanvasClick(e: MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const cx = Math.floor((e.clientX - rect.left) * scaleX);
    const cy = Math.floor((e.clientY - rect.top) * scaleY);
    const rgb = samplePixelAvg(canvas, cx, cy);
    setSelectedRgb(rgb);
    setSelectedHsv(rgbToHsv(...rgb));
  }

  async function handleStartLabeling() {
    if (!uploadResult) return;
    setStarting(true);
    setStartError(null);
    setStartResult(null);
    const hsvString = selectedHsv ? selectedHsv.join(',') : null;
    try {
      const res = await startLabeling(uploadResult.session_id, hsvString, jerseyThreshold);
      setStartResult({ frames_queued: res.frames_queued });
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setStartError(detail ?? 'Error en iniciar el pipeline. Torna-ho a intentar.');
    } finally {
      setStarting(false);
    }
  }

  async function handleClearLsTasks() {
    if (!confirm('Segur que vols eliminar totes les tasques de Label Studio?')) return;
    setClearing(true);
    await clearLsTasks();
    setStartResult(null);
    if (lsPollingRef.current) clearInterval(lsPollingRef.current);
    const stats = await getLsStats();
    setLsStats(stats);
    setClearing(false);
  }

  const currentFrameNumber = totalFrames > 0 ? representativeFrameNumbers[frameIndex] : null;
  const isProcessing = !!(uploadResult && totalFrames === 0 && retryCount > 0 && retryCount < MAX_RETRIES);
  const labelingProgress = startResult && lsStats && lsStats.total_tasks > 0
    ? Math.round((lsStats.predicted_tasks / lsStats.total_tasks) * 100)
    : 0;

  return (
    <div className="min-h-screen bg-matrix-bg px-4 py-8">
      <div className="max-w-2xl mx-auto">

        {/* Capçalera */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="" className="h-16 w-auto object-contain" />
            <div>
              <h1 className="text-sm font-bold text-neon uppercase tracking-widest">
                Etiquetatge de vídeos
              </h1>
              <p className="text-xs text-matrix-muted">Mode administrador</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs bg-[#1a1a0d] text-matrix-warning border border-[#3a3a1a]
                             px-2 py-0.5 rounded uppercase tracking-widest">
              ADM
            </span>
            <button
              onClick={() => navigate('/')}
              className="text-xs text-matrix-muted hover:text-neon uppercase tracking-widest transition-colors"
            >
              ← Tornar
            </button>
          </div>
        </div>

        <div className="space-y-4">

          {/* ── Upload ──────────────────────────────────────────────────────── */}
          <div className="bg-matrix-card border border-matrix-border rounded-lg p-6">
            <h2 className="text-xs font-bold text-neon uppercase tracking-widest mb-4">
              Pujar vídeo per a etiquetatge
            </h2>

            <form onSubmit={handleSubmit} noValidate className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-matrix-muted uppercase tracking-wider mb-1.5">
                  Fitxer de vídeo (.mp4)
                </label>
                <div
                  className="relative border border-dashed border-matrix-border rounded px-4 py-6 text-center
                             hover:border-neon transition-colors cursor-pointer bg-matrix-input"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={20} className="mx-auto text-matrix-muted mb-2" />
                  <p className="text-xs text-matrix-muted">
                    {file
                      ? <span className="text-neon font-medium">{file.name}</span>
                      : 'Clica per seleccionar un fitxer .mp4'}
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".mp4,video/mp4"
                    className="sr-only"
                    onChange={handleFileChange}
                    disabled={uploading}
                  />
                </div>
              </div>

              <div>
                <label htmlFor="frameInterval" className="block text-xs font-medium text-matrix-muted uppercase tracking-wider mb-1.5">
                  Interval entre frames (segons)
                </label>
                <input
                  id="frameInterval"
                  type="number"
                  min={1} max={60}
                  value={frameInterval}
                  onChange={(e) => setFrameInterval(Number(e.target.value))}
                  disabled={uploading}
                  className="w-24 px-3 py-2 bg-matrix-input border border-matrix-border rounded
                             text-sm text-matrix-text
                             focus:outline-none focus:border-neon focus:ring-1 focus:ring-neon
                             disabled:opacity-40 transition-colors"
                />
                <p className="text-xs text-matrix-muted mt-1">
                  1 frame cada {frameInterval}s (~{Math.round(3600 / frameInterval).toLocaleString()} frames/hora)
                </p>
              </div>

              {uploadError && (
                <div className="flex items-start gap-2 text-matrix-error text-xs
                                bg-[#1a0d0d] border border-[#3a1a1a] rounded px-3 py-2">
                  <AlertCircle size={14} className="mt-0.5 shrink-0" /><span>{uploadError}</span>
                </div>
              )}

              {uploadResult && (
                <div className="bg-[#0d1a0d] border border-matrix-border rounded px-3 py-3 space-y-2">
                  <div className="flex items-start gap-2 text-neon text-xs">
                    <CheckCircle size={14} className="mt-0.5 shrink-0" />
                    <div>
                      <p className="font-medium">Vídeo encoat correctament</p>
                      <p className="text-xs mt-0.5 text-matrix-muted font-mono break-all">
                        Session: {uploadResult.session_id}
                      </p>
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-matrix-muted mb-1">
                      {totalFrames > 0
                        ? <span className="text-neon font-medium">{totalFrames} frames extrets</span>
                        : isProcessing ? 'Processant vídeo…' : 'Esperant processador…'}
                    </div>
                    <div className="h-1 bg-matrix-raised rounded-full overflow-hidden">
                      {totalFrames > 0
                        ? <div className="h-full w-full bg-neon rounded-full" />
                        : <div className="h-full w-1/3 bg-matrix-info rounded-full"
                            style={{ animation: 'indeterminate 1.4s ease-in-out infinite' }}
                          />
                      }
                    </div>
                  </div>
                </div>
              )}

              <button
                type="submit"
                disabled={!file || uploading}
                className="w-full flex items-center justify-center gap-2
                           bg-neon hover:bg-neon-600 disabled:opacity-40
                           text-matrix-bg font-bold py-2.5 px-4 rounded
                           text-xs uppercase tracking-widest transition-colors"
              >
                {uploading
                  ? <><Loader size={14} className="animate-spin" />Pujant vídeo…</>
                  : <><Upload size={14} />Pujar vídeo</>}
              </button>
            </form>
          </div>

          {/* ── Reprendre sessió existent ──────────────────────────────────── */}
          {!uploadResult && (
            <div className="bg-matrix-card border border-matrix-border rounded-lg p-6 space-y-3">
              <div className="flex items-center gap-2">
                <RefreshCw size={14} className="text-matrix-muted" />
                <h2 className="text-xs font-bold text-neon uppercase tracking-widest">
                  Reprendre sessió existent
                </h2>
              </div>
              <p className="text-xs text-matrix-muted">
                Introdueix el <span className="font-mono text-matrix-text">session_id</span> per continuar
                amb la selecció de color i la pre-anotació.
              </p>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={resumeInput}
                  onChange={(e) => setResumeInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleResumeSession()}
                  placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                  className="flex-1 px-3 py-2 bg-matrix-input border border-matrix-border rounded
                             text-xs font-mono text-matrix-text placeholder-matrix-muted
                             focus:outline-none focus:border-neon focus:ring-1 focus:ring-neon
                             transition-colors"
                />
                <button
                  onClick={handleResumeSession}
                  className="flex items-center gap-1.5 px-4 py-2
                             bg-matrix-raised border border-matrix-border hover:border-neon
                             text-matrix-text hover:text-neon text-xs font-medium rounded transition-colors"
                >
                  <RefreshCw size={12} />Reprendre
                </button>
              </div>
              {resumeError && (
                <div className="flex items-center gap-2 text-matrix-error text-xs">
                  <AlertCircle size={12} /><span>{resumeError}</span>
                </div>
              )}
            </div>
          )}

          {/* ── Eyedropper ────────────────────────────────────────────────── */}
          {uploadResult && (
            <div className="bg-matrix-card border border-matrix-border rounded-lg p-6 space-y-4">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full border-2 border-neon flex items-center justify-center">
                  <div className="w-1.5 h-1.5 bg-neon rounded-full" />
                </div>
                <h2 className="text-xs font-bold text-neon uppercase tracking-widest">
                  Color de samarreta
                </h2>
                <button
                  onClick={() => { setUploadResult(null); setStartResult(null); }}
                  className="ml-auto text-xs text-matrix-muted hover:text-matrix-error transition-colors"
                >
                  Canviar sessió
                </button>
              </div>
              <p className="text-xs text-matrix-muted font-mono break-all">
                Sessió: {uploadResult.session_id}
              </p>
              <p className="text-xs text-matrix-muted">
                Passa el cursor per sobre d'un jugador del teu equip.{' '}
                <span className="text-neon font-bold">Clica</span> per confirmar la selecció.
              </p>

              {/* Canvas + lupa */}
              <div className="space-y-2">
                <div className="relative rounded overflow-hidden border border-matrix-border
                                bg-matrix-input min-h-[180px] flex items-center justify-center">
                  {frameLoading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-matrix-input z-10">
                      <Loader size={24} className="animate-spin text-neon" />
                    </div>
                  )}
                  {frameError && !frameLoading && (
                    <p className="text-xs text-matrix-muted px-4 text-center">{frameError}</p>
                  )}
                  <canvas
                    ref={canvasRef}
                    onClick={handleCanvasClick}
                    onMouseMove={handleCanvasMouseMove}
                    onMouseLeave={handleCanvasMouseLeave}
                    className="w-full cursor-crosshair block"
                    style={{ display: frameLoading || frameError ? 'none' : 'block' }}
                  />

                  {lensVisible && !frameLoading && !frameError && (
                    <div
                      className="absolute pointer-events-none z-20 rounded-full border-2 border-neon shadow-xl overflow-hidden"
                      style={{
                        width: LENS_SIZE,
                        height: LENS_SIZE,
                        left: lensPos.x,
                        top: lensPos.y,
                        transform: 'translate(-50%, -120%)',
                      }}
                    >
                      <canvas ref={lensCanvasRef} width={LENS_SIZE} height={LENS_SIZE} />
                      <div
                        className="absolute bottom-0 left-0 right-0 h-5"
                        style={{ backgroundColor: rgbToCss(...hoverRgb) }}
                      />
                    </div>
                  )}
                </div>

                {/* Navegació prev/next */}
                <div className="flex items-center justify-between">
                  <button
                    onClick={() => setFrameIndex((i) => Math.max(0, i - 1))}
                    disabled={frameIndex === 0 || frameLoading}
                    className="flex items-center gap-1 px-3 py-1.5 text-xs text-matrix-muted
                               border border-matrix-border rounded hover:border-neon hover:text-neon
                               disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    <ChevronLeft size={14} />Anterior
                  </button>
                  <span className="text-xs text-matrix-muted">
                    {currentFrameNumber
                      ? `Frame ${currentFrameNumber} / ${totalFrames} (${Math.round(REPRESENTATIVE_PCTS[frameIndex] * 100)}%)`
                      : 'Carregant…'}
                  </span>
                  <button
                    onClick={() => setFrameIndex((i) => Math.min(REPRESENTATIVE_PCTS.length - 1, i + 1))}
                    disabled={frameIndex === REPRESENTATIVE_PCTS.length - 1 || frameLoading}
                    className="flex items-center gap-1 px-3 py-1.5 text-xs text-matrix-muted
                               border border-matrix-border rounded hover:border-neon hover:text-neon
                               disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    Següent<ChevronRight size={14} />
                  </button>
                </div>
              </div>

              {/* Color seleccionat */}
              {selectedHsv && selectedRgb ? (
                <div className="flex items-center gap-3 p-3 bg-matrix-raised border border-matrix-border rounded">
                  <div
                    className="w-10 h-10 rounded border border-matrix-border shrink-0"
                    style={{ backgroundColor: rgbToCss(...selectedRgb) }}
                  />
                  <div className="text-xs">
                    <p className="font-semibold text-neon">Color seleccionat</p>
                    <p className="text-matrix-muted font-mono">
                      HSV: {selectedHsv.join(', ')} &nbsp;|&nbsp; RGB: {selectedRgb.join(', ')}
                    </p>
                  </div>
                  <button
                    onClick={() => { setSelectedRgb(null); setSelectedHsv(null); }}
                    className="ml-auto text-xs text-matrix-muted hover:text-matrix-error transition-colors"
                  >
                    Esborrar
                  </button>
                </div>
              ) : (
                <p className="text-xs text-matrix-muted italic">
                  Passa el cursor per sobre la samarreta d'un jugador del teu equip i clica per seleccionar.
                </p>
              )}

              {/* Slider threshold */}
              <div>
                <label className="block text-xs font-medium text-matrix-muted uppercase tracking-wider mb-1.5">
                  Tolerància de color:{' '}
                  <span className="font-mono text-neon">{jerseyThreshold}</span>
                </label>
                <input
                  type="range" min={10} max={60} step={5}
                  value={jerseyThreshold}
                  onChange={(e) => setJerseyThreshold(Number(e.target.value))}
                  className="w-full accent-neon"
                />
                <div className="flex justify-between text-xs text-matrix-muted mt-0.5">
                  <span>Estricte (10)</span><span>Ampli (60)</span>
                </div>
              </div>

              {startError && (
                <div className="flex items-start gap-2 text-matrix-error text-xs
                                bg-[#1a0d0d] border border-[#3a1a1a] rounded px-3 py-2">
                  <AlertCircle size={14} className="mt-0.5 shrink-0" /><span>{startError}</span>
                </div>
              )}

              {/* Barra de progrés etiquetatge */}
              {startResult && (
                <div className="p-3 bg-[#0d1a0d] border border-matrix-border rounded space-y-2">
                  <div className="flex items-center gap-2 text-neon text-xs">
                    <CheckCircle size={14} className="shrink-0" />
                    <span className="font-medium">Pre-anotació en curs</span>
                    {lsStatsLoading && <Loader size={12} className="animate-spin ml-auto" />}
                  </div>
                  {lsStats && lsStats.total_tasks > 0 && (
                    <>
                      <div className="flex justify-between text-xs text-matrix-muted">
                        <span>{lsStats.predicted_tasks} / {lsStats.total_tasks} frames pre-anotats</span>
                        <span className="text-neon">{labelingProgress}%</span>
                      </div>
                      <div className="h-1 bg-matrix-raised rounded-full overflow-hidden">
                        <div
                          className="h-full bg-neon rounded-full transition-all duration-500"
                          style={{ width: `${labelingProgress}%` }}
                        />
                      </div>
                    </>
                  )}
                </div>
              )}

              {!startResult && (
                <button
                  onClick={handleStartLabeling}
                  disabled={starting || totalFrames === 0}
                  className="w-full flex items-center justify-center gap-2
                             bg-neon hover:bg-neon-600 disabled:opacity-40
                             text-matrix-bg font-bold py-2.5 px-4 rounded
                             text-xs uppercase tracking-widest transition-colors"
                >
                  {starting
                    ? <><Loader size={14} className="animate-spin" />Encuant frames…</>
                    : <><Play size={14} />Iniciar etiquetatge</>}
                </button>
              )}
              {!selectedHsv && !starting && !startResult && (
                <p className="text-xs text-matrix-warning text-center">
                  Pots iniciar sense color — tots els jugadors es classificaran com a <em>player_own</em>.
                </p>
              )}
            </div>
          )}

          {/* ── Label Studio ─────────────────────────────────────────────── */}
          <div className="bg-matrix-card border border-matrix-border rounded-lg p-6 space-y-4">
            <h2 className="text-xs font-bold text-neon uppercase tracking-widest">
              Etiquetatge
            </h2>

            {/* Stats */}
            {lsStats && (
              <div className="grid grid-cols-4 gap-px bg-matrix-border rounded overflow-hidden text-center text-xs">
                {[
                  { label: 'Totals', value: lsStats.total_tasks, color: 'text-matrix-text' },
                  { label: 'Pre-anotades', value: lsStats.predicted_tasks, color: 'text-matrix-info' },
                  { label: 'Revisades', value: lsStats.annotated_tasks, color: 'text-neon' },
                  { label: 'Pendents', value: lsStats.total_tasks - lsStats.annotated_tasks, color: 'text-matrix-muted' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="bg-matrix-raised py-3 px-2">
                    <p className={`text-xl font-bold ${color}`}>{value}</p>
                    <p className="text-matrix-muted mt-0.5">{label}</p>
                  </div>
                ))}
              </div>
            )}

            <div className="flex gap-3">
              <a
                href={labelStudioUrl}
                target="_blank"
                rel="noreferrer"
                className="flex-1 inline-flex items-center justify-center gap-2
                           bg-neon hover:bg-neon-600 text-matrix-bg font-bold
                           py-2.5 px-4 rounded text-xs uppercase tracking-widest transition-colors"
              >
                <ExternalLink size={13} />Obrir Label Studio
              </a>
              <button
                onClick={handleClearLsTasks}
                disabled={clearing || (lsStats?.total_tasks ?? 0) === 0}
                className="flex items-center gap-2
                           bg-matrix-raised border border-matrix-border hover:border-matrix-error
                           text-matrix-muted hover:text-matrix-error
                           font-medium py-2.5 px-4 rounded text-xs uppercase tracking-widest
                           transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {clearing
                  ? <Loader size={13} className="animate-spin" />
                  : <Trash2 size={13} />}
                Netejar
              </button>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
