import { useRef, useState, type ChangeEvent, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, ExternalLink, Upload, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { uploadLabelingVideo, type LabelingUploadResponse } from '../api/labeling';

export default function LabelingPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [frameInterval, setFrameInterval] = useState(2);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<LabelingUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0] ?? null;
    if (selected && selected.type !== 'video/mp4') {
      setError('Només s\'accepten fitxers .mp4');
      setFile(null);
      return;
    }
    setError(null);
    setResult(null);
    setFile(selected);
  }

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setError(null);
    setResult(null);

    try {
      const response = await uploadLabelingVideo(file, frameInterval);
      setResult(response);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: unknown) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? 'Error en pujar el vídeo. Torna-ho a intentar.');
    } finally {
      setUploading(false);
    }
  }

  const labelStudioUrl =
    (import.meta.env.VITE_LABEL_STUDIO_URL as string | undefined) ??
    'http://localhost:8081';

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
        {/* Secció upload */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-base font-semibold text-gray-900 mb-4">Pujar vídeo per a etiquetatge</h2>

          <form onSubmit={handleSubmit} noValidate className="space-y-4">
            {/* Selector de fitxer */}
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

            {/* Interval de frames */}
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

            {/* Missatge d'error */}
            {error && (
              <div className="flex items-start gap-2 text-red-600 text-sm bg-red-50 border border-red-200 rounded-lg px-3 py-2">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Resultat d'upload */}
            {result && (
              <div className="flex items-start gap-2 text-green-700 text-sm bg-green-50 border border-green-200 rounded-lg px-3 py-3">
                <CheckCircle size={16} className="mt-0.5 shrink-0" />
                <div>
                  <p className="font-medium">Vídeo encoat correctament</p>
                  <p className="text-xs mt-1 text-green-600 font-mono break-all">
                    Session ID: {result.session_id}
                  </p>
                  <p className="text-xs text-green-600">Estat: {result.status}</p>
                </div>
              </div>
            )}

            {/* Botó d'enviament */}
            <button
              type="submit"
              disabled={!file || uploading}
              className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium py-2.5 px-4 rounded-lg text-sm transition-colors"
            >
              {uploading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Pujant vídeo...
                </>
              ) : (
                <>
                  <Upload size={16} />
                  Pujar vídeo
                </>
              )}
            </button>
          </form>
        </div>

        {/* Secció Label Studio */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-base font-semibold text-gray-900 mb-1">Etiquetatge</h2>
          <p className="text-sm text-gray-500 mb-4">
            Un cop el vídeo s'hagi trossejat, obre Label Studio per etiquetar els frames generats.
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
