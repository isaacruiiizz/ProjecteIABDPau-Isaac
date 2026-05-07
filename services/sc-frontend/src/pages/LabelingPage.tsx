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
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  CheckCircle,
  ExternalLink,
  Loader2,
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
const LENS_SIZE = 88;      // px — diàmetre de la lupa
const LENS_ZOOM = 4;       // factor de zoom
const LENS_CROP = LENS_SIZE / LENS_ZOOM; // px de la imatge capturats (22×22)

export default function LabelingPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lensCanvasRef = useRef<HTMLCanvasElement>(null);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lsPollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Upload ──────────────────────────────────────────────────────────────────
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

  // ── Reprendre sessió existent ────────────────────────────────────────────────
  const [resumeInput, setResumeInput] = useState('');
  const [resumeError, setResumeError] = useState<string | null>(null);

  // ── Eyedropper / canvas ─────────────────────────────────────────────────────
  const [totalFrames, setTotalFrames] = useState(0);
  const [frameIndex, setFrameIndex] = useState(0);
  const [frameLoading, setFrameLoading] = useState(false);
  const [frameError, setFrameError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [selectedRgb, setSelectedRgb] = useState<[number, number, number] | null>(null);
  const [selectedHsv, setSelectedHsv] = useState<[number, number, number] | null>(null);
  const [jerseyThreshold, setJerseyThreshold] = useState(30);

  // Lupa
  const [lensVisible, setLensVisible] = useState(false);
  const [lensPos, setLensPos] = useState({ x: 0, y: 0 });     // posició CSS px relativa al contenidor
  const [hoverRgb, setHoverRgb] = useState<[number, number, number]>([0, 0, 0]);

  // ── Inici pipeline ──────────────────────────────────────────────────────────
  const [starting, setStarting] = useState(false);
  const [startResult, setStartResult] = useState<{ frames_queued: number } | null>(null);
  const [startError, setStartError] = useState<string | null>(null);

  // ── Label Studio stats ──────────────────────────────────────────────────────
  const [lsStats, setLsStats] = useState<LsStatsResponse | null>(null);
  const [lsStatsLoading, setLsStatsLoading] = useState(false);
  const [clearing, setClearing] = useState(false);

  const labelStudioUrl =
    (import.meta.env.VITE_LABEL_STUDIO_URL as string | undefined) ?? 'http://localhost:8081';

  const representativeFrameNumbers = REPRESENTATIVE_PCTS.map(
    (pct) => Math.max(1, Math.round(pct * totalFrames)),
  );

  // ── Carrega un frame al canvas ──────────────────────────────────────────────
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

  // Persisteix la sessió activa per recuperar-la si l'usuari recarrega o navega
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

  // Polling Label Studio stats quan el pipeline ha arrencat
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

  // Stats inicials de LS (sense pipeline actiu)
  useEffect(() => {
    getLsStats().then(setLsStats).catch(() => null);
  }, []);

  // ── Handler reprendre sessió ─────────────────────────────────────────────────
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

  // ── Handlers upload ─────────────────────────────────────────────────────────
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

  // ── Lupa eyedropper ─────────────────────────────────────────────────────────
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
    // Línia de mira central
    lctx.strokeStyle = 'rgba(255,255,255,0.8)';
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

    // Posició de la lupa relativa al contenidor del canvas
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

  // ── Handler inici pipeline ──────────────────────────────────────────────────
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

  // ── Handler netejar tasques LS ───────────────────────────────────────────────
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
    <div className="min-h-screen bg-gray-50">
      {/* Capçalera */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-2xl mx-auto flex items-center gap-3">
          <button
            onClick={() => navigate('/')}
            className="text-gray-500 hover:text-gray-800 transition-colors"
            aria-label="Tornar al dashboard"
          >
            <ArrowLeft size={20} />
          </button>
          <h1 className="text-lg font-semibold text-gray-900">Etiquetatge de vídeos</h1>
          <span className="ml-auto text-xs bg-blue-100 text-blue-700 font-medium px-2 py-0.5 rounded-full">
            Admin
          </span>
        </div>
      </div>

      <div className="max-w-2xl mx-auto px-6 py-8 space-y-6">

        {/* ── Upload ──────────────────────────────────────────────────────── */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-base font-semibold text-gray-900 mb-4">Pujar vídeo per a etiquetatge</h2>

          <form onSubmit={handleSubmit} noValidate className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Fitxer de vídeo (.mp4)</label>
              <div
                className="relative border-2 border-dashed border-gray-300 rounded-lg px-4 py-6 text-center hover:border-blue-400 transition-colors cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload size={24} className="mx-auto text-gray-400 mb-2" />
                <p className="text-sm text-gray-500">
                  {file
                    ? <span className="text-blue-600 font-medium">{file.name}</span>
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
              <label htmlFor="frameInterval" className="block text-sm font-medium text-gray-700 mb-1">
                Interval entre frames (segons)
              </label>
              <input
                id="frameInterval"
                type="number"
                min={1} max={60}
                value={frameInterval}
                onChange={(e) => setFrameInterval(Number(e.target.value))}
                disabled={uploading}
                className="w-24 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-400"
              />
              <p className="text-xs text-gray-400 mt-1">
                1 frame cada {frameInterval}s (~{Math.round(3600 / frameInterval).toLocaleString()} frames/hora)
              </p>
            </div>

            {uploadError && (
              <div className="flex items-start gap-2 text-red-600 text-sm bg-red-50 border border-red-200 rounded-lg px-3 py-2">
                <AlertCircle size={16} className="mt-0.5 shrink-0" /><span>{uploadError}</span>
              </div>
            )}

            {uploadResult && (
              <div className="text-sm bg-green-50 border border-green-200 rounded-lg px-3 py-3 space-y-2">
                <div className="flex items-start gap-2 text-green-700">
                  <CheckCircle size={16} className="mt-0.5 shrink-0" />
                  <div>
                    <p className="font-medium">Vídeo encoat correctament</p>
                    <p className="text-xs mt-0.5 text-green-600 font-mono break-all">
                      Session: {uploadResult.session_id}
                    </p>
                  </div>
                </div>

                {/* Barra de progrés extracció de frames */}
                <div>
                  <div className="text-xs text-gray-500 mb-1">
                    {totalFrames > 0
                      ? <span className="text-green-600 font-medium">{totalFrames} frames extrets</span>
                      : isProcessing ? 'Processant vídeo…' : 'Esperant processador…'}
                  </div>
                  <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    {totalFrames > 0 ? (
                      <div className="h-full w-full bg-green-500 rounded-full" />
                    ) : (
                      <div className="h-full w-1/3 bg-blue-400 rounded-full animate-[indeterminate_1.4s_ease-in-out_infinite]"
                        style={{ animation: 'indeterminate 1.4s ease-in-out infinite' }}
                      />
                    )}
                  </div>
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={!file || uploading}
              className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium py-2.5 px-4 rounded-lg text-sm transition-colors"
            >
              {uploading
                ? <><Loader2 size={16} className="animate-spin" />Pujant vídeo…</>
                : <><Upload size={16} />Pujar vídeo</>}
            </button>
          </form>
        </div>

        {/* ── Reprendre sessió existent ───────────────────────────────────── */}
        {!uploadResult && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-3">
            <div className="flex items-center gap-2">
              <RefreshCw size={18} className="text-gray-500" />
              <h2 className="text-base font-semibold text-gray-900">Reprendre sessió existent</h2>
            </div>
            <p className="text-sm text-gray-500">
              Si ja has pujat un vídeo anteriorment, introdueix el <span className="font-mono text-gray-700">session_id</span> per continuar amb la selecció de color i la pre-anotació.
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={resumeInput}
                onChange={(e) => setResumeInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleResumeSession()}
                placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={handleResumeSession}
                className="flex items-center gap-1.5 px-4 py-2 bg-gray-800 hover:bg-gray-900 text-white text-sm font-medium rounded-lg transition-colors"
              >
                <RefreshCw size={14} />Reprendre
              </button>
            </div>
            {resumeError && (
              <div className="flex items-center gap-2 text-red-600 text-sm">
                <AlertCircle size={14} /><span>{resumeError}</span>
              </div>
            )}
          </div>
        )}

        {/* ── Eyedropper ──────────────────────────────────────────────────── */}
        {uploadResult && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-4">
            <div className="flex items-center gap-2">
              <div className="w-5 h-5 rounded-full border-2 border-indigo-600 flex items-center justify-center">
                <div className="w-2 h-2 bg-indigo-600 rounded-full" />
              </div>
              <h2 className="text-base font-semibold text-gray-900">Color de samarreta</h2>
              <button
                onClick={() => { setUploadResult(null); setStartResult(null); }}
                className="ml-auto text-xs text-gray-400 hover:text-gray-600 transition-colors"
              >
                Canviar sessió
              </button>
            </div>
            <p className="text-xs text-gray-400 font-mono break-all">
              Sessió: {uploadResult.session_id}
            </p>
            <p className="text-sm text-gray-500">
              Passa el cursor per sobre d'un jugador del teu equip. El color es mostra en temps real.
              <strong className="text-gray-700"> Clica</strong> per confirmar la selecció.
            </p>

            {/* Canvas + lupa */}
            <div className="space-y-2">
              <div
                className="relative rounded-lg overflow-hidden border border-gray-200 bg-gray-100 min-h-[180px] flex items-center justify-center"
              >
                {frameLoading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-100 z-10">
                    <Loader2 size={28} className="animate-spin text-gray-400" />
                  </div>
                )}
                {frameError && !frameLoading && (
                  <p className="text-sm text-gray-400 px-4 text-center">{frameError}</p>
                )}
                <canvas
                  ref={canvasRef}
                  onClick={handleCanvasClick}
                  onMouseMove={handleCanvasMouseMove}
                  onMouseLeave={handleCanvasMouseLeave}
                  className="w-full cursor-crosshair block"
                  style={{ display: frameLoading || frameError ? 'none' : 'block' }}
                />

                {/* Lupa circular */}
                {lensVisible && !frameLoading && !frameError && (
                  <div
                    className="absolute pointer-events-none z-20 rounded-full border-2 border-white shadow-xl overflow-hidden"
                    style={{
                      width: LENS_SIZE,
                      height: LENS_SIZE,
                      left: lensPos.x,
                      top: lensPos.y,
                      transform: 'translate(-50%, -120%)',
                    }}
                  >
                    <canvas
                      ref={lensCanvasRef}
                      width={LENS_SIZE}
                      height={LENS_SIZE}
                    />
                    {/* Franja de color inferior */}
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
                  className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft size={16} />Anterior
                </button>
                <span className="text-xs text-gray-400">
                  {currentFrameNumber
                    ? `Frame ${currentFrameNumber} / ${totalFrames} (${Math.round(REPRESENTATIVE_PCTS[frameIndex] * 100)}%)`
                    : 'Carregant…'}
                </span>
                <button
                  onClick={() => setFrameIndex((i) => Math.min(REPRESENTATIVE_PCTS.length - 1, i + 1))}
                  disabled={frameIndex === REPRESENTATIVE_PCTS.length - 1 || frameLoading}
                  className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  Següent<ChevronRight size={16} />
                </button>
              </div>
            </div>

            {/* Color seleccionat */}
            {selectedHsv && selectedRgb ? (
              <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                <div
                  className="w-12 h-12 rounded-lg border border-gray-300 shrink-0 shadow-sm"
                  style={{ backgroundColor: rgbToCss(...selectedRgb) }}
                />
                <div className="text-sm">
                  <p className="font-semibold text-gray-800">Color seleccionat</p>
                  <p className="text-gray-500 font-mono text-xs">
                    HSV: {selectedHsv.join(', ')} &nbsp;|&nbsp; RGB: {selectedRgb.join(', ')}
                  </p>
                </div>
                <button
                  onClick={() => { setSelectedRgb(null); setSelectedHsv(null); }}
                  className="ml-auto text-xs text-gray-400 hover:text-red-500 transition-colors"
                >
                  Esborrar
                </button>
              </div>
            ) : (
              <p className="text-xs text-gray-400 italic">
                Passa el cursor per sobre la samarreta d'un jugador del teu equip i clica per seleccionar.
              </p>
            )}

            {/* Slider threshold */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Tolerància de color: <span className="font-mono">{jerseyThreshold}</span>
              </label>
              <input
                type="range" min={10} max={60} step={5}
                value={jerseyThreshold}
                onChange={(e) => setJerseyThreshold(Number(e.target.value))}
                className="w-full accent-indigo-600"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                <span>Estricte (10)</span><span>Ampli (60)</span>
              </div>
            </div>

            {startError && (
              <div className="flex items-start gap-2 text-red-600 text-sm bg-red-50 border border-red-200 rounded-lg px-3 py-2">
                <AlertCircle size={16} className="mt-0.5 shrink-0" /><span>{startError}</span>
              </div>
            )}

            {/* Barra de progrés etiquetatge */}
            {startResult && (
              <div className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg space-y-2">
                <div className="flex items-center gap-2 text-indigo-700 text-sm">
                  <CheckCircle size={16} className="shrink-0" />
                  <span className="font-medium">Pre-anotació en curs</span>
                  {lsStatsLoading && <Loader2 size={14} className="animate-spin ml-auto" />}
                </div>
                {lsStats && lsStats.total_tasks > 0 && (
                  <>
                    <div className="flex justify-between text-xs text-indigo-600">
                      <span>{lsStats.predicted_tasks} / {lsStats.total_tasks} frames pre-anotats</span>
                      <span>{labelingProgress}%</span>
                    </div>
                    <div className="h-1.5 bg-indigo-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-indigo-500 rounded-full transition-all duration-500"
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
                className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white font-medium py-2.5 px-4 rounded-lg text-sm transition-colors"
              >
                {starting
                  ? <><Loader2 size={16} className="animate-spin" />Encuant frames…</>
                  : <><Play size={16} />Iniciar etiquetatge</>}
              </button>
            )}
            {!selectedHsv && !starting && !startResult && (
              <p className="text-xs text-amber-600 text-center">
                Pots iniciar sense color — tots els jugadors es classificaran com a <em>player_own</em>.
              </p>
            )}
          </div>
        )}

        {/* ── Label Studio ─────────────────────────────────────────────────── */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-4">
          <h2 className="text-base font-semibold text-gray-900">Etiquetatge</h2>

          {/* Stats */}
          {lsStats && (
            <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200 text-sm">
              <div className="text-center flex-1">
                <p className="text-2xl font-bold text-gray-800">{lsStats.total_tasks}</p>
                <p className="text-xs text-gray-500">Totals</p>
              </div>
              <div className="w-px h-10 bg-gray-200" />
              <div className="text-center flex-1">
                <p className="text-2xl font-bold text-violet-600">{lsStats.predicted_tasks}</p>
                <p className="text-xs text-gray-500">Pre-anotades</p>
              </div>
              <div className="w-px h-10 bg-gray-200" />
              <div className="text-center flex-1">
                <p className="text-2xl font-bold text-green-600">{lsStats.annotated_tasks}</p>
                <p className="text-xs text-gray-500">Revisades</p>
              </div>
              <div className="w-px h-10 bg-gray-200" />
              <div className="text-center flex-1">
                <p className="text-2xl font-bold text-gray-400">
                  {lsStats.total_tasks - lsStats.annotated_tasks}
                </p>
                <p className="text-xs text-gray-500">Pendents</p>
              </div>
            </div>
          )}

          <div className="flex gap-3">
            <a
              href={labelStudioUrl}
              target="_blank"
              rel="noreferrer"
              className="flex-1 inline-flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2.5 px-4 rounded-lg text-sm transition-colors"
            >
              <ExternalLink size={16} />Obrir Label Studio
            </a>
            <button
              onClick={handleClearLsTasks}
              disabled={clearing || (lsStats?.total_tasks ?? 0) === 0}
              className="flex items-center gap-2 bg-white hover:bg-red-50 border border-gray-300 hover:border-red-300 text-gray-600 hover:text-red-600 font-medium py-2.5 px-4 rounded-lg text-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {clearing
                ? <Loader2 size={16} className="animate-spin" />
                : <Trash2 size={16} />}
              Netejar
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
