import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  AlertCircle, Download, Film, Loader, Trophy,
} from 'lucide-react';
import { getMatch, type MatchDetail } from '../api/matches';

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('ca-ES', {
    day: '2-digit', month: 'short', year: 'numeric',
  });
}

function formatDuration(start: number | null, end: number | null): string {
  if (start === null || end === null) return '—';
  const total = Math.round(end - start);
  const m = Math.floor(total / 60).toString().padStart(2, '0');
  const s = (total % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function StatCard({ value, label }: { value: string; label: string }) {
  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm px-4 py-4 text-center">
      <p className="text-lg font-bold text-gray-900">{value}</p>
      <p className="text-xs text-gray-500 mt-0.5">{label}</p>
    </div>
  );
}

export default function ResultsPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [match,       setMatch]       = useState<MatchDetail | null>(null);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    if (!id) return;
    getMatch(id)
      .then(setMatch)
      .catch(() => setError("No s'han pogut carregar els resultats del partit."))
      .finally(() => setLoading(false));
  }, [id]);

  async function handleDownload() {
    if (!match?.download_url) return;
    setDownloading(true);
    try {
      const a = document.createElement('a');
      a.href = match.download_url;
      a.download = `${match.title.replace(/\s+/g, '_')}_processat.mp4`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } finally {
      setDownloading(false);
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Loader size={28} className="animate-spin text-blue-500" />
      </div>
    );
  }

  if (error || !match) {
    return (
      <div className="min-h-screen bg-gray-50 px-4 py-8">
        <div className="max-w-lg mx-auto">
          <button
            onClick={() => navigate('/matches')}
            className="text-sm text-gray-500 hover:text-gray-700 transition-colors mb-6 inline-block"
          >
            ← Tornar als partits
          </button>
          <div className="flex items-start gap-2 text-red-600 text-sm bg-red-50
                          border border-red-200 rounded-xl px-4 py-3">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span>{error ?? 'Partit no trobat.'}</span>
          </div>
        </div>
      </div>
    );
  }

  const duration = formatDuration(match.start_seconds, match.end_seconds);

  return (
    <div className="min-h-screen bg-gray-50 px-4 py-8">
      <div className="max-w-lg mx-auto space-y-4">

        {/* Nav */}
        <button
          onClick={() => navigate('/matches')}
          className="text-sm text-gray-500 hover:text-gray-700 transition-colors"
        >
          ← Tornar als partits
        </button>

        {/* Hero */}
        <div className="bg-gradient-to-br from-slate-800 to-blue-900 rounded-2xl p-8 text-center
                        shadow-lg">
          <div className="flex justify-center mb-4">
            <div className="bg-yellow-400/20 rounded-full p-3">
              <Trophy size={28} className="text-yellow-400" />
            </div>
          </div>
          <h1 className="text-xl font-bold text-white leading-snug mb-3">
            {match.title}
          </h1>
          <span className="inline-flex items-center gap-1.5 bg-green-500/20 text-green-300
                           text-xs font-medium px-3 py-1 rounded-full border border-green-500/30">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 inline-block" />
            Anàlisi completada
          </span>
          <p className="text-blue-200 text-sm mt-3">
            {formatDate(match.created_at)}
            {duration !== '—' && <> · <span className="font-medium">{duration}</span> de durada</>}
          </p>
        </div>

        {/* Download card */}
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 space-y-4">
          <h2 className="text-sm font-semibold text-gray-900">Vídeo processat</h2>

          {/* Preview placeholder */}
          <div className="bg-slate-900 rounded-xl py-10 flex flex-col items-center gap-2">
            <Film size={36} className="text-slate-500" />
            <span className="text-slate-500 text-xs">output.mp4</span>
          </div>

          {/* Download button */}
          {match.download_url ? (
            <button
              onClick={handleDownload}
              disabled={downloading}
              className="w-full flex items-center justify-center gap-2 bg-green-600
                         hover:bg-green-700 disabled:opacity-60 disabled:cursor-not-allowed
                         text-white font-medium py-3 rounded-xl transition-colors"
            >
              {downloading
                ? <Loader size={16} className="animate-spin" />
                : <Download size={16} />
              }
              {downloading ? 'Preparant descàrrega…' : 'Descarregar vídeo (MP4)'}
            </button>
          ) : (
            <div className="text-center text-sm text-gray-400 py-2">
              El vídeo no està disponible per a descàrrega.
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3">
          <StatCard value="RT-DETR" label="Model" />
          <StatCard value={duration} label="Durada" />
          <StatCard value={formatDate(match.created_at)} label="Data" />
        </div>

      </div>
    </div>
  );
}
