import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AlertCircle, Bot, CheckCircle, Download, Film, Loader, Play, Timer, Users } from 'lucide-react';
import { getMatch, type AiStats, type MatchDetail } from '../api/matches';

function formatDate(iso: string): string {
  const d = new Date(iso);
  return (
    d.toLocaleDateString('ca-ES', { day: '2-digit', month: 'short', year: 'numeric' }) +
    ' ' +
    d.toLocaleTimeString('ca-ES', { hour: '2-digit', minute: '2-digit' })
  );
}

function formatDuration(start: number | null, end: number | null): string {
  if (start === null || end === null) return '—';
  const total = Math.round(end - start);
  const m = Math.floor(total / 60).toString().padStart(2, '0');
  const s = (total % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

const NOTES_KEY = (id: string) => `sc-notes-${id}`;

function StatItem({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex flex-col items-center text-center px-3 py-3">
      <span className="text-lg font-bold text-gray-900">{value}</span>
      {sub && <span className="text-xs text-gray-400">{sub}</span>}
      <span className="text-xs text-gray-500 mt-0.5">{label}</span>
    </div>
  );
}

function StatsCard({ stats }: { stats: AiStats }) {
  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm px-5 py-5">
      <div className="flex items-center gap-2 mb-4">
        <Users size={16} className="text-blue-500 shrink-0" />
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
          Estadístiques de detecció
        </p>
      </div>
      <div className="grid grid-cols-3 divide-x divide-gray-100 mb-3">
        <StatItem
          label="Jugadors / frame"
          value={stats.avg_players_per_frame.toFixed(1)}
        />
        <StatItem
          label="Equip propi"
          value={stats.avg_own_per_frame.toFixed(1)}
          sub={`${stats.pct_own.toFixed(0)}%`}
        />
        <StatItem
          label="Rival"
          value={stats.avg_other_per_frame.toFixed(1)}
        />
      </div>
      <div className="grid grid-cols-3 divide-x divide-gray-100 border-t border-gray-100 pt-3">
        <StatItem
          label="Confiança mitjana"
          value={`${(stats.avg_confidence * 100).toFixed(0)}%`}
        />
        <StatItem
          label="Frames analitzats"
          value={String(stats.total_frames)}
        />
        <StatItem
          label="Pic de densitat"
          value={`${stats.max_density_time_s.toFixed(0)}s`}
          sub={`${stats.max_density_count} jugadors`}
        />
      </div>
    </div>
  );
}

function AiReportCard({ report }: { report: string }) {
  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm px-5 py-5">
      <div className="flex items-center gap-2 mb-3">
        <Bot size={16} className="text-violet-500 shrink-0" />
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
          Anàlisi IA
        </p>
      </div>
      <p className="text-sm text-gray-700 leading-relaxed">{report}</p>
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
  const [showVideo,   setShowVideo]   = useState(false);
  const [notes,       setNotes]       = useState('');
  const [savedNote,   setSavedNote]   = useState(false);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!id) return;
    getMatch(id)
      .then(setMatch)
      .catch(() => setError("No s'han pogut carregar els resultats del partit."))
      .finally(() => setLoading(false));
    setNotes(localStorage.getItem(NOTES_KEY(id)) ?? '');
  }, [id]);

  async function handleDownload() {
    if (!match?.download_url) return;
    setDownloading(true);
    try {
      const res = await fetch(match.download_url);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${match.title.replace(/\s+/g, '_')}_processat.mp4`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } finally {
      setDownloading(false);
    }
  }

  function handleSaveNotes() {
    if (!id) return;
    localStorage.setItem(NOTES_KEY(id), notes);
    setSavedNote(true);
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => setSavedNote(false), 2000);
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Loader size={24} className="animate-spin text-blue-500" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 px-4 py-8">
      <div className="max-w-2xl mx-auto">

        {/* Capçalera */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 text-white rounded-2xl p-2.5">
              <Timer size={22} />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900">Resultats</h1>
              <p className="text-xs text-gray-500">Vídeo processat i anàlisi completada</p>
            </div>
          </div>
          <button
            onClick={() => navigate('/matches')}
            className="text-sm text-gray-500 hover:text-gray-700 transition-colors"
          >
            ← Tornar
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-start gap-2 text-red-600 text-sm bg-red-50
                          border border-red-200 rounded-xl px-4 py-3">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {match && (
          <div className="space-y-3">

            {/* Info del partit */}
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm px-5 py-4">
              <div className="flex items-start gap-4">
                <div className="bg-emerald-100 text-emerald-600 rounded-xl p-2.5 shrink-0">
                  <Film size={20} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap mb-1">
                    <p className="text-sm font-semibold text-gray-900 truncate">{match.title}</p>
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full
                                     text-xs font-medium bg-green-100 text-green-700">
                      <CheckCircle size={11} />
                      Completat
                    </span>
                  </div>
                  <p className="text-xs text-gray-400">{formatDate(match.created_at)}</p>
                </div>
                <div className="text-right shrink-0">
                  <p className="text-xs text-gray-500">
                    Durada:{' '}
                    <span className="font-medium text-gray-700">
                      {formatDuration(match.start_seconds, match.end_seconds)}
                    </span>
                  </p>
                </div>
              </div>
            </div>

            {/* Vídeo processat */}
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm px-5 py-5">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-4">
                Vídeo processat
              </p>

              {match.download_url ? (
                <>
                  <div className="flex gap-2 mb-4">
                    <button
                      onClick={() => setShowVideo(v => !v)}
                      className="flex-1 flex items-center justify-center gap-2
                                 border border-gray-200 hover:border-blue-300 hover:bg-blue-50
                                 text-gray-700 hover:text-blue-700 text-sm font-medium
                                 py-2.5 rounded-xl transition-colors"
                    >
                      <Play size={15} />
                      {showVideo ? 'Amagar vídeo' : 'Veure vídeo'}
                    </button>
                    <button
                      onClick={handleDownload}
                      disabled={downloading}
                      className="flex-1 flex items-center justify-center gap-2
                                 bg-blue-600 hover:bg-blue-700 disabled:opacity-60
                                 disabled:cursor-not-allowed text-white text-sm font-medium
                                 py-2.5 rounded-xl transition-colors"
                    >
                      {downloading
                        ? <Loader size={15} className="animate-spin" />
                        : <Download size={15} />
                      }
                      {downloading ? 'Descarregant…' : 'Descarregar'}
                    </button>
                  </div>

                  {showVideo && (
                    <div className="rounded-xl overflow-hidden bg-black">
                      <video
                        src={match.download_url}
                        controls
                        className="w-full max-h-96"
                      />
                    </div>
                  )}
                </>
              ) : (
                <p className="text-sm text-gray-400 text-center py-2">
                  El vídeo no està disponible.
                </p>
              )}
            </div>

            {/* Estadístiques IA */}
            {match.ai_stats && <StatsCard stats={match.ai_stats} />}

            {/* Informe narratiu */}
            {match.ai_report && <AiReportCard report={match.ai_report} />}

            {/* Notes de l'entrenador */}
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm px-5 py-5">
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                  Notes del partit
                </p>
                {savedNote && (
                  <span className="text-xs text-green-600 font-medium">Desat ✓</span>
                )}
              </div>
              <textarea
                value={notes}
                onChange={e => setNotes(e.target.value)}
                placeholder="Escriu aquí les teves observacions del partit..."
                rows={8}
                className="w-full resize-none text-sm text-gray-700 placeholder-gray-300
                           border border-gray-200 rounded-xl px-4 py-3
                           focus:outline-none focus:ring-2 focus:ring-blue-500
                           focus:border-transparent transition-shadow"
              />
              <button
                onClick={handleSaveNotes}
                className="mt-3 w-full bg-gray-900 hover:bg-gray-800 text-white
                           text-sm font-medium py-2.5 rounded-xl transition-colors"
              >
                Desar notes
              </button>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}
