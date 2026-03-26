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
  Pipette,
  Play,
  Upload,
} from 'lucide-react';
import {
  getLabelingFrame,
  startLabeling,
  uploadLabelingVideo,
  type LabelingUploadResponse,
} from '../api/labeling';
import { rgbToHsv, rgbToCss } from '../utils/color';

// Percentatges del total de frames que es mostren com a representatius.
// S'evita el 0% (primer frame = càmera ajustant-se o pista buida).
const REPRESENTATIVE_PCTS = [0.10, 0.25, 0.50, 0.75, 0.90];

export default function LabelingPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // ── Estat upload ────────────────────────────────────────────────────────────
  const [file, setFile] = useState<File | null>(null);
  const [frameInterval, setFrameInterval] = useState(2);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<LabelingUploadResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // ── Estat eyedropper ────────────────────────────────────────────────────────
  const [totalFrames, setTotalFrames] = useState(0);
  const [frameIndex, setFrameIndex] = useState(0);        // índex dins REPRESENTATIVE_PCTS
  const [frameLoading, setFrameLoading] = useState(false);
  const [frameError, setFrameError] = useState<string | null>(null);
  const [selectedRgb, setSelectedRgb] = useState<[number, number, number] | null>(null);
  const [selectedHsv, setSelectedHsv] = useState<[number, number, number] | null>(null);
  const [jerseyThreshold, setJerseyThreshold] = useState(30);

  // ── Estat inici pipeline ────────────────────────────────────────────────────
  const [starting, setStarting] = useState(false);
  const [startResult, setStartResult] = useState<{ frames_queued: number } | null>(null);
  const [startError, setStartError] = useState<string | null>(null);

  const labelStudioUrl =
    (import.meta.env.VITE_LABEL_STUDIO_URL as string | undefined) ?? 'http://localhost:8081';

  // ── Frames representatius calculats a partir de totalFrames ────────────────
  const representativeFrameNumbers = REPRESENTATIVE_PCTS.map(
    (pct) => Math.max(1, Math.round(pct * totalFrames)),
  );

  // ── Carrega i pinta un frame al canvas ─────────────────────────────────────
  const loadFrame = useCallback(
    async (sessionId: string, frameNumber: number) => {
      if (!canvasRef.current) return;
      setFrameLoading(true);
      setFrameError(null);

      try {
        const { frame_url, total_frames } = await getLabelingFrame(sessionId, frameNumber);

        if (total_frames === 0) {
          setFrameError('No s\'han trobat frames. Espera que el vídeo acabi de processar-se.');
          return;
        }
        setTotalFrames(total_frames);

        // Descarrega la imatge com a blob (evita problemes CORS amb getImageData)
        const resp = await fetch(frame_url);
        if (!resp.ok) throw new Error(`Error en carregar el frame: ${resp.status}`);
        const blob = await resp.blob();
        const objectUrl = URL.createObjectURL(blob);

        const img = new Image();
        img.onload = () => {
          const canvas = canvasRef.current;
          if (!canvas) return;
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          const ctx = canvas.getContext('2d');
          ctx?.drawImage(img, 0, 0);
          URL.revokeObjectURL(objectUrl);
        };
        img.onerror = () => {
          URL.revokeObjectURL(objectUrl);
          setFrameError('Error en renderitzar el frame.');
        };
        img.src = objectUrl;
      } catch {
        setFrameError('No s\'ha pogut carregar el frame. Comprova que el vídeo s\'ha processat.');
      } finally {
        setFrameLoading(false);
      }
    },
    [],
  );

  // Carrega el primer frame representatiu quan tenim un upload exitós
  useEffect(() => {
    if (!uploadResult) return;
    loadFrame(uploadResult.session_id, representativeFrameNumbers[0] || 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadResult]);

  // Carrega el frame representatiu quan naveguem
  useEffect(() => {
    if (!uploadResult || totalFrames === 0) return;
    loadFrame(uploadResult.session_id, representativeFrameNumbers[frameIndex]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameIndex, totalFrames]);

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

  // ── Handler eyedropper ──────────────────────────────────────────────────────
  function handleCanvasClick(e: MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    // Escala les coordenades del clic al tamany real del canvas
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = Math.floor((e.clientX - rect.left) * scaleX);
    const y = Math.floor((e.clientY - rect.top) * scaleY);

    const pixel = ctx.getImageData(x, y, 1, 1).data;
    const rgb: [number, number, number] = [pixel[0], pixel[1], pixel[2]];
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

  const currentFrameNumber = totalFrames > 0 ? representativeFrameNumbers[frameIndex] : null;

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

        {/* ── Secció upload ─────────────────────────────────────────────────── */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-base font-semibold text-gray-900 mb-4">Pujar vídeo per a etiquetatge</h2>

          <form onSubmit={handleSubmit} noValidate className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Fitxer de vídeo (.mp4)
              </label>
              <div
                className="relative border-2 border-dashed border-gray-300 rounded-lg px-4 py-6 text-center hover:border-blue-400 transition-colors cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload size={24} className="mx-auto text-gray-400 mb-2" />
                <p className="text-sm text-gray-500">
                  {file ? (
                    <span className="text-blue-600 font-medium">{file.name}</span>
                  ) : (
                    'Clica per seleccionar un fitxer .mp4'
                  )}
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
              <label
                htmlFor="frameInterval"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Interval entre frames (segons)
              </label>
              <input
                id="frameInterval"
                type="number"
                min={1}
                max={60}
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
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <span>{uploadError}</span>
              </div>
            )}

            {uploadResult && (
              <div className="flex items-start gap-2 text-green-700 text-sm bg-green-50 border border-green-200 rounded-lg px-3 py-3">
                <CheckCircle size={16} className="mt-0.5 shrink-0" />
                <div>
                  <p className="font-medium">Vídeo encoat correctament</p>
                  <p className="text-xs mt-1 text-green-600 font-mono break-all">
                    Session ID: {uploadResult.session_id}
                  </p>
                  <p className="text-xs text-green-600">
                    Els frames s'estan extraient en segon pla...
                  </p>
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={!file || uploading}
              className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium py-2.5 px-4 rounded-lg text-sm transition-colors"
            >
              {uploading ? (
                <><Loader2 size={16} className="animate-spin" />Pujant vídeo...</>
              ) : (
                <><Upload size={16} />Pujar vídeo</>
              )}
            </button>
          </form>
        </div>

        {/* ── Secció eyedropper (visible un cop hi ha uploadResult) ─────────── */}
        {uploadResult && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-4">
            <div className="flex items-center gap-2">
              <Pipette size={18} className="text-indigo-600" />
              <h2 className="text-base font-semibold text-gray-900">Color de samarreta</h2>
            </div>
            <p className="text-sm text-gray-500">
              Clica sobre un jugador del teu equip per seleccionar el color de la samarreta.
              Usa els botons per navegar entre frames representatius del vídeo.
            </p>

            {/* Canvas amb navegació */}
            <div className="space-y-2">
              <div className="relative rounded-lg overflow-hidden border border-gray-200 bg-gray-100 min-h-[180px] flex items-center justify-center">
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
                  className="w-full cursor-crosshair block"
                  style={{ display: frameLoading || frameError ? 'none' : 'block' }}
                />
              </div>

              {/* Navegació prev/next */}
              <div className="flex items-center justify-between">
                <button
                  onClick={() => setFrameIndex((i) => Math.max(0, i - 1))}
                  disabled={frameIndex === 0 || frameLoading}
                  className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft size={16} />
                  Anterior
                </button>
                <span className="text-xs text-gray-400">
                  {currentFrameNumber
                    ? `Frame ${currentFrameNumber} / ${totalFrames} (${Math.round(REPRESENTATIVE_PCTS[frameIndex] * 100)}%)`
                    : 'Carregant...'}
                </span>
                <button
                  onClick={() => setFrameIndex((i) => Math.min(REPRESENTATIVE_PCTS.length - 1, i + 1))}
                  disabled={frameIndex === REPRESENTATIVE_PCTS.length - 1 || frameLoading}
                  className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  Següent
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>

            {/* Previsualització del color seleccionat */}
            {selectedHsv && selectedRgb ? (
              <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                <div
                  className="w-10 h-10 rounded-md border border-gray-300 shrink-0"
                  style={{ backgroundColor: rgbToCss(...selectedRgb) }}
                />
                <div className="text-sm">
                  <p className="font-medium text-gray-800">Color seleccionat</p>
                  <p className="text-gray-500 font-mono text-xs">
                    HSV: {selectedHsv.join(', ')} &nbsp;|&nbsp; RGB: {selectedRgb.join(', ')}
                  </p>
                </div>
              </div>
            ) : (
              <p className="text-xs text-gray-400 italic">
                Encara no has seleccionat cap color. Clica sobre la samarreta d'un jugador del teu equip.
              </p>
            )}

            {/* Slider de threshold */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Tolerància de color: <span className="font-mono">{jerseyThreshold}</span>
              </label>
              <input
                type="range"
                min={10}
                max={60}
                step={5}
                value={jerseyThreshold}
                onChange={(e) => setJerseyThreshold(Number(e.target.value))}
                className="w-full accent-indigo-600"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                <span>Estricte (10)</span>
                <span>Ampli (60)</span>
              </div>
              <p className="text-xs text-gray-400 mt-1">
                Augmenta la tolerància si el detector classifica massa jugadors com a rivals.
              </p>
            </div>

            {/* Missatge d'error inici */}
            {startError && (
              <div className="flex items-start gap-2 text-red-600 text-sm bg-red-50 border border-red-200 rounded-lg px-3 py-2">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <span>{startError}</span>
              </div>
            )}

            {/* Resultat inici */}
            {startResult && (
              <div className="flex items-start gap-2 text-green-700 text-sm bg-green-50 border border-green-200 rounded-lg px-3 py-3">
                <CheckCircle size={16} className="mt-0.5 shrink-0" />
                <div>
                  <p className="font-medium">Pre-anotació iniciada</p>
                  <p className="text-xs text-green-600">
                    {startResult.frames_queued} frames encuats per a inferència.
                  </p>
                </div>
              </div>
            )}

            {/* Botó Iniciar etiquetatge */}
            {!startResult && (
              <button
                onClick={handleStartLabeling}
                disabled={starting}
                className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white font-medium py-2.5 px-4 rounded-lg text-sm transition-colors"
              >
                {starting ? (
                  <><Loader2 size={16} className="animate-spin" />Encuant frames...</>
                ) : (
                  <><Play size={16} />Iniciar etiquetatge</>
                )}
              </button>
            )}
            {!selectedHsv && !starting && !startResult && (
              <p className="text-xs text-amber-600 text-center">
                Pots iniciar sense color — tots els jugadors es classificaran com a <em>player_own</em>.
              </p>
            )}
          </div>
        )}

        {/* ── Secció Label Studio ───────────────────────────────────────────── */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-base font-semibold text-gray-900 mb-1">Etiquetatge</h2>
          <p className="text-sm text-gray-500 mb-4">
            Un cop els frames s'hagin pre-anotat, obre Label Studio per revisar i validar les deteccions.
          </p>
          <a
            href={labelStudioUrl}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2.5 px-4 rounded-lg text-sm transition-colors"
          >
            <ExternalLink size={16} />
            Obrir Label Studio
          </a>
        </div>

      </div>
    </div>
  );
}
