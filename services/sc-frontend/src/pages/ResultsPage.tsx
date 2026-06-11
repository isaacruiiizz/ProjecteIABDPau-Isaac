import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  AlertCircle, Bot, CheckCircle, Clipboard, ClipboardCheck,
  Download, Film, Loader, Play, RefreshCw, Timer, Users,
} from 'lucide-react';
import { getMatch, refineAiReport, type AiStats, type MatchDetail } from '../api/matches';

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
      <span className="text-base font-bold text-neon">{value}</span>
      {sub && <span className="text-xs text-matrix-muted">{sub}</span>}
      <span className="text-xs text-matrix-muted mt-0.5 uppercase tracking-widest">{label}</span>
    </div>
  );
}

function StatsCard({ stats }: { stats: AiStats }) {
  const peakPct = stats.duration_s > 0
    ? Math.max(0, Math.min(100, (stats.max_density_time_s / stats.duration_s) * 100))
    : 50;

  return (
    <div className="bg-matrix-card border border-matrix-border rounded-lg px-5 py-5">
      <div className="flex items-center gap-2 mb-4">
        <Users size={14} className="text-neon shrink-0" />
        <p className="text-xs font-medium text-matrix-muted uppercase tracking-widest">
          Estadístiques de detecció
        </p>
      </div>
      <div className="grid grid-cols-3 divide-x divide-matrix-border mb-3">
        <StatItem label="Jugadors / frame" value={stats.avg_players_per_frame.toFixed(1)} />
        <StatItem
          label="Equip propi"
          value={stats.avg_own_per_frame.toFixed(1)}
          sub={`${stats.pct_own.toFixed(0)}%`}
        />
        <StatItem label="Rival" value={stats.avg_other_per_frame.toFixed(1)} />
      </div>
      <div className="grid grid-cols-3 divide-x divide-matrix-border border-t border-matrix-border pt-3 mb-4">
        <StatItem label="Confiança" value={`${(stats.avg_confidence * 100).toFixed(0)}%`} />
        <StatItem label="Frames" value={String(stats.total_frames)} />
        <StatItem
          label="Pic densitat"
          value={`${stats.max_density_time_s.toFixed(0)}s`}
          sub={`${stats.max_density_count} jug.`}
        />
      </div>

      {/* Barra de densitat: posició del pic dins la durada total */}
      <div className="border-t border-matrix-border pt-3">
        <div className="flex items-center justify-between text-xs text-matrix-muted mb-1.5 uppercase tracking-widest">
          <span>0s</span>
          <span className="text-neon">Pic d'intensitat</span>
          <span>{stats.duration_s.toFixed(0)}s</span>
        </div>
        <div className="relative h-2 bg-matrix-raised rounded-full">
          <div className="absolute h-full bg-neon opacity-10 rounded-full w-full" />
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2 h-4 bg-neon rounded-sm"
            style={{ left: `${peakPct}%` }}
            title={`Pic a ${stats.max_density_time_s.toFixed(1)}s (${stats.max_density_count} jugadors)`}
          />
        </div>
        <p className="text-xs text-matrix-muted mt-1.5 text-center">
          Màxima aglomeració al segon{' '}
          <span className="text-neon font-bold">{stats.max_density_time_s.toFixed(1)}</span>
          {' — '}
          <span className="text-neon font-bold">{stats.max_density_count}</span> jugadors detectats
        </p>
      </div>
    </div>
  );
}

function renderReport(text: string) {
  return text.split('\n').map((line, i) => {
    const bold = line.match(/^\*\*(.+?)\*\*:?$/);
    if (bold) return (
      <p key={i} className="font-bold text-neon mt-4 mb-1 uppercase tracking-wide text-xs">
        {'>'} {bold[1]}
      </p>
    );
    if (line.startsWith('- ')) return (
      <li key={i} className="ml-4 text-xs text-matrix-text leading-relaxed list-none">
        <span className="text-neon mr-2">▸</span>{line.slice(2)}
      </li>
    );
    if (line.trim() === '') return <div key={i} className="h-2" />;
    return <p key={i} className="text-xs text-matrix-text leading-relaxed">{line}</p>;
  });
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { /* clipboard unavailable */ }
  }

  return (
    <button
      onClick={handleCopy}
      className={`flex items-center gap-1 text-xs px-2 py-1 rounded border transition-colors
        ${copied
          ? 'border-neon text-neon bg-matrix-raised'
          : 'border-matrix-border text-matrix-muted hover:border-neon hover:text-neon'}`}
      title="Copiar informe"
    >
      {copied ? <ClipboardCheck size={12} /> : <Clipboard size={12} />}
      {copied ? 'Copiat!' : 'Copiar'}
    </button>
  );
}

function AiReportCard({
  report,
  matchId,
  initialRefined,
}: {
  report: string;
  matchId: string;
  initialRefined: string | null;
}) {
  const [tab,         setTab]         = useState<'auto' | 'detail'>('auto');
  const [context,     setContext]     = useState('');
  const [refined,     setRefined]     = useState<string | null>(initialRefined);
  const [loading,     setLoading]     = useState(false);
  const [refineError, setRefineError] = useState<string | null>(null);

  async function handleRefine() {
    setLoading(true);
    setRefineError(null);
    try {
      const result = await refineAiReport(matchId, context);
      setRefined(result);
      setTab('detail');
    } catch {
      setRefineError("No s'ha pogut generar l'anàlisi. Comprova que Ollama està actiu.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-matrix-card border border-matrix-border rounded-lg px-5 py-5">
      {/* Capçalera + pestanyes */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Bot size={14} className="text-neon shrink-0" />
          <p className="text-xs font-medium text-matrix-muted uppercase tracking-widest">Anàlisi IA</p>
        </div>
        <div className="flex rounded border border-matrix-border overflow-hidden text-xs font-medium">
          <button
            onClick={() => setTab('auto')}
            className={`px-3 py-1.5 uppercase tracking-widest transition-colors ${tab === 'auto'
              ? 'bg-neon text-matrix-bg font-bold'
              : 'text-matrix-muted hover:bg-matrix-raised'}`}
          >
            Auto
          </button>
          <button
            onClick={() => setTab('detail')}
            className={`px-3 py-1.5 uppercase tracking-widest transition-colors flex items-center gap-1 ${tab === 'detail'
              ? 'bg-neon text-matrix-bg font-bold'
              : 'text-matrix-muted hover:bg-matrix-raised'}`}
          >
            Detallada
            {refined && <span className="w-1.5 h-1.5 rounded-full bg-neon inline-block" />}
          </button>
        </div>
      </div>

      {/* Pestanya automàtica */}
      {tab === 'auto' && (
        <>
          <div className="space-y-0.5 mb-3">{renderReport(report)}</div>
          <div className="flex justify-end">
            <CopyButton text={report} />
          </div>
        </>
      )}

      {/* Pestanya detallada */}
      {tab === 'detail' && (
        <div className="space-y-4">
          {refined && (
            <div className="space-y-0.5 pb-4 border-b border-matrix-border">
              {renderReport(refined)}
              <div className="flex justify-end pt-2">
                <CopyButton text={refined} />
              </div>
            </div>
          )}

          {refineError && (
            <div className="flex items-start gap-2 text-matrix-error text-xs
                            bg-[#1a0d0d] border border-[#3a1a1a] rounded px-4 py-3">
              <AlertCircle size={13} className="mt-0.5 shrink-0" />
              <span>{refineError}</span>
            </div>
          )}

          <div>
            <p className="text-xs text-matrix-muted uppercase tracking-widest mb-2">
              {refined ? 'Afegeix més context i regenera:' : 'Descriu el context del partit:'}
            </p>
            <textarea
              value={context}
              onChange={e => setContext(e.target.value)}
              placeholder="Ex: Érem l'equip de samarretes blaves, jugàvem de local. Segon temps, anàvem perdent 1-2. El rival jugava en 1-2-1 i ens pressionava al nostre cierre. El nostre pivot tenia pèrdues de pilota i als últims minuts vam fer portero-jugador sense èxit..."
              rows={4}
              className="w-full resize-none text-xs text-matrix-text placeholder-matrix-muted
                         bg-matrix-input border border-matrix-border rounded px-4 py-3
                         focus:outline-none focus:border-neon focus:ring-1 focus:ring-neon
                         transition-colors"
            />
          </div>

          <button
            onClick={handleRefine}
            disabled={loading || !context.trim()}
            className="w-full flex items-center justify-center gap-2 bg-neon
                       hover:bg-neon-600 disabled:opacity-40
                       text-matrix-bg text-xs font-bold py-2.5 rounded
                       uppercase tracking-widest transition-colors"
          >
            {loading
              ? <><Loader size={13} className="animate-spin" /> Generant anàlisi… (~60s)</>
              : <><RefreshCw size={13} /> {refined ? 'Regenerar anàlisi' : 'Generar anàlisi detallada'}</>
            }
          </button>
        </div>
      )}
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
      <div className="min-h-screen bg-matrix-bg flex items-center justify-center">
        <Loader size={22} className="animate-spin text-neon" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-matrix-bg px-4 py-8">
      <div className="max-w-2xl mx-auto">

        {/* Capçalera */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="" className="h-16 w-auto object-contain" />
            <div>
              <h1 className="text-sm font-bold text-neon uppercase tracking-widest">Resultats</h1>
              <p className="text-xs text-matrix-muted">Vídeo processat i anàlisi completada</p>
            </div>
          </div>
          <button
            onClick={() => navigate('/matches')}
            className="text-xs text-matrix-muted hover:text-neon uppercase tracking-widest transition-colors"
          >
            ← Tornar
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-start gap-2 text-matrix-error text-xs
                          bg-[#1a0d0d] border border-[#3a1a1a] rounded px-4 py-3">
            <AlertCircle size={14} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {match && (
          <div className="space-y-3">

            {/* Info del partit */}
            <div className="bg-matrix-card border border-matrix-border rounded-lg px-5 py-4">
              <div className="flex items-start gap-4">
                <div className="bg-matrix-raised text-neon border border-matrix-border rounded p-2.5 shrink-0">
                  <Film size={18} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap mb-1">
                    <p className="text-xs font-bold text-matrix-text truncate uppercase tracking-wide">
                      {match.title}
                    </p>
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs
                                     bg-matrix-raised text-neon border border-matrix-border font-medium">
                      <CheckCircle size={10} />
                      Completat
                    </span>
                  </div>
                  <p className="text-xs text-matrix-muted">{formatDate(match.created_at)}</p>
                </div>
                <div className="text-right shrink-0">
                  <p className="text-xs text-matrix-muted uppercase tracking-widest">Durada</p>
                  <p className="text-xs font-bold text-neon">
                    {formatDuration(match.start_seconds, match.end_seconds)}
                  </p>
                </div>
              </div>
            </div>

            {/* Vídeo processat */}
            <div className="bg-matrix-card border border-matrix-border rounded-lg px-5 py-5">
              <p className="text-xs font-medium text-matrix-muted uppercase tracking-widest mb-4">
                Vídeo processat
              </p>

              {match.download_url ? (
                <>
                  <div className="flex gap-2 mb-4">
                    <button
                      onClick={() => setShowVideo(v => !v)}
                      className="flex-1 flex items-center justify-center gap-2
                                 border border-matrix-border hover:border-neon
                                 text-matrix-muted hover:text-neon text-xs font-medium
                                 py-2.5 rounded uppercase tracking-widest transition-colors"
                    >
                      <Play size={13} />
                      {showVideo ? 'Amagar' : 'Reproduir'}
                    </button>
                    <button
                      onClick={handleDownload}
                      disabled={downloading}
                      className="flex-1 flex items-center justify-center gap-2
                                 bg-neon hover:bg-neon-600 disabled:opacity-40
                                 text-matrix-bg text-xs font-bold
                                 py-2.5 rounded uppercase tracking-widest transition-colors"
                    >
                      {downloading
                        ? <Loader size={13} className="animate-spin" />
                        : <Download size={13} />
                      }
                      {downloading ? 'Descarregant…' : 'Descarregar'}
                    </button>
                  </div>

                  {showVideo && (
                    <div className="rounded overflow-hidden bg-black border border-matrix-border">
                      <video src={match.download_url} controls className="w-full max-h-96" />
                    </div>
                  )}
                </>
              ) : (
                <p className="text-xs text-matrix-muted text-center py-2 uppercase tracking-widest">
                  El vídeo no està disponible.
                </p>
              )}
            </div>

            {/* Estadístiques IA */}
            {match.ai_stats && <StatsCard stats={match.ai_stats} />}

            {/* Informe narratiu */}
            {match.ai_report && (
              <AiReportCard
                report={match.ai_report}
                matchId={match.match_id}
                initialRefined={match.ai_report_refined}
              />
            )}

            {/* Notes de l'entrenador */}
            <div className="bg-matrix-card border border-matrix-border rounded-lg px-5 py-5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Timer size={14} className="text-neon shrink-0" />
                  <p className="text-xs font-medium text-matrix-muted uppercase tracking-widest">
                    Notes del partit
                  </p>
                </div>
                {savedNote && (
                  <span className="text-xs text-neon font-bold uppercase tracking-widest">✓ Desat</span>
                )}
              </div>
              <textarea
                value={notes}
                onChange={e => setNotes(e.target.value)}
                placeholder="Escriu aquí les teves observacions del partit..."
                rows={6}
                className="w-full resize-none text-xs text-matrix-text placeholder-matrix-muted
                           bg-matrix-input border border-matrix-border rounded px-4 py-3
                           focus:outline-none focus:border-neon focus:ring-1 focus:ring-neon
                           transition-colors"
              />
              <button
                onClick={handleSaveNotes}
                className="mt-3 w-full bg-matrix-raised border border-matrix-border
                           hover:border-neon hover:text-neon text-matrix-muted
                           text-xs font-medium py-2.5 rounded uppercase tracking-widest transition-colors"
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
