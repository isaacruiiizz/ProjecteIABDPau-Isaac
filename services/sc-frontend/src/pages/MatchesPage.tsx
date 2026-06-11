import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertCircle, CheckCircle, Clock, ExternalLink, Loader, Play, Plus, RotateCcw, Trash2, XCircle,
} from 'lucide-react';
import { deleteMatch, getMatches, processMatch, type MatchListItem } from '../api/matches';

const STATUS_CONFIG: Record<string, { label: string; classes: string; icon: React.ReactNode }> = {
  pending:      { label: 'Pendent',        classes: 'bg-[#1a1a0d] text-matrix-warning border border-[#3a3a1a]', icon: <Clock size={11} /> },
  processing:   { label: 'Processant',     classes: 'bg-[#0d1a1a] text-matrix-info border border-[#1a3a3a]',   icon: <Loader size={11} className="animate-spin" /> },
  frames_ready: { label: 'Frames extrets', classes: 'bg-[#0d1a1a] text-matrix-info border border-[#1a3a3a]',   icon: <Loader size={11} /> },
  done:         { label: 'Completat',      classes: 'bg-[#0d1a0d] text-neon border border-matrix-border',      icon: <CheckCircle size={11} /> },
  error:        { label: 'Error',          classes: 'bg-[#1a0d0d] text-matrix-error border border-[#3a1a1a]',  icon: <XCircle size={11} /> },
};

function StatusBadge({ status }: { status: string }) {
  const cfg = STATUS_CONFIG[status] ?? {
    label: status, classes: 'bg-matrix-card text-matrix-muted border border-matrix-border', icon: null,
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${cfg.classes}`}>
      {cfg.icon}
      {cfg.label}
    </span>
  );
}

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

function canProcess(m: MatchListItem): boolean {
  return (m.status === 'pending' || m.status === 'error' || m.status === 'processing') &&
    m.has_roi && m.start_seconds !== null;
}

function isReprocess(m: MatchListItem): boolean {
  return m.status === 'error' || m.status === 'processing';
}

export default function MatchesPage() {
  const navigate = useNavigate();
  const [matches,       setMatches]       = useState<MatchListItem[]>([]);
  const [loading,       setLoading]       = useState(true);
  const [error,         setError]         = useState<string | null>(null);
  const [processingIds, setProcessingIds] = useState<Set<string>>(new Set());
  const [processError,  setProcessError]  = useState<string | null>(null);
  const [selectedIds,   setSelectedIds]   = useState<Set<string>>(new Set());
  const [confirmingId,  setConfirmingId]  = useState<string | null>(null);
  const [deletingIds,   setDeletingIds]   = useState<Set<string>>(new Set());
  const [deleteError,   setDeleteError]   = useState<string | null>(null);

  useEffect(() => {
    getMatches()
      .then(setMatches)
      .catch(() => setError("No s'han pogut carregar els partits."))
      .finally(() => setLoading(false));
  }, []);

  async function handleProcess(matchId: string) {
    setProcessingIds(prev => new Set(prev).add(matchId));
    setProcessError(null);
    try {
      await processMatch(matchId);
      setMatches(prev =>
        prev.map(m => m.match_id === matchId ? { ...m, status: 'processing' } : m),
      );
    } catch {
      setProcessError("No s'ha pogut iniciar el processament. Torna-ho a intentar.");
    } finally {
      setProcessingIds(prev => { const next = new Set(prev); next.delete(matchId); return next; });
    }
  }

  function toggleSelect(id: string) {
    setSelectedIds(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function toggleSelectAll() {
    if (selectedIds.size === matches.length) setSelectedIds(new Set());
    else setSelectedIds(new Set(matches.map(m => m.match_id)));
  }

  async function handleDeleteOne(matchId: string) {
    setDeletingIds(prev => new Set(prev).add(matchId));
    setDeleteError(null);
    try {
      await deleteMatch(matchId);
      setMatches(prev => prev.filter(m => m.match_id !== matchId));
      setSelectedIds(prev => { const n = new Set(prev); n.delete(matchId); return n; });
    } catch {
      setDeleteError("No s'ha pogut eliminar el partit. Torna-ho a intentar.");
    } finally {
      setDeletingIds(prev => { const n = new Set(prev); n.delete(matchId); return n; });
      setConfirmingId(null);
    }
  }

  async function handleDeleteSelected() {
    const ids = Array.from(selectedIds);
    setDeleteError(null);
    for (const id of ids) {
      setDeletingIds(prev => new Set(prev).add(id));
      try {
        await deleteMatch(id);
        setMatches(prev => prev.filter(m => m.match_id !== id));
        setSelectedIds(prev => { const n = new Set(prev); n.delete(id); return n; });
      } catch {
        setDeleteError("Alguns partits no s'han pogut eliminar.");
      } finally {
        setDeletingIds(prev => { const n = new Set(prev); n.delete(id); return n; });
      }
    }
  }

  return (
    <div className="min-h-screen bg-matrix-bg px-4 py-8">
      <div className="max-w-2xl mx-auto">

        {/* Capçalera */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="" className="h-16 w-auto object-contain" />
            <div>
              <h1 className="text-sm font-bold text-neon uppercase tracking-widest">
                Registre de partits
              </h1>
              <p className="text-xs text-matrix-muted">Vídeos pujats i estat de processament</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/')}
              className="text-xs text-matrix-muted hover:text-neon uppercase tracking-widest transition-colors"
            >
              ← Tornar
            </button>
            <button
              onClick={() => navigate('/process')}
              className="flex items-center gap-1.5 bg-neon hover:bg-neon-600
                         text-matrix-bg text-xs font-bold px-3 py-1.5 rounded
                         uppercase tracking-widest transition-colors"
            >
              <Plus size={13} /> Nou
            </button>
          </div>
        </div>

        {/* Carregant */}
        {loading && (
          <div className="flex items-center justify-center py-16">
            <Loader size={22} className="animate-spin text-neon" />
          </div>
        )}

        {/* Error càrrega */}
        {error && (
          <div className="flex items-start gap-2 text-matrix-error text-xs
                          bg-[#1a0d0d] border border-[#3a1a1a] rounded px-4 py-3">
            <AlertCircle size={14} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Error processament */}
        {processError && (
          <div className="flex items-start gap-2 text-matrix-error text-xs
                          bg-[#1a0d0d] border border-[#3a1a1a] rounded px-4 py-3 mb-3">
            <AlertCircle size={14} className="mt-0.5 shrink-0" />
            <span>{processError}</span>
          </div>
        )}

        {/* Error eliminació */}
        {deleteError && (
          <div className="flex items-start gap-2 text-matrix-error text-xs
                          bg-[#1a0d0d] border border-[#3a1a1a] rounded px-4 py-3 mb-3">
            <AlertCircle size={14} className="mt-0.5 shrink-0" />
            <span>{deleteError}</span>
          </div>
        )}

        {/* Barra d'accions en bloc */}
        {selectedIds.size > 0 && (
          <div className="flex items-center justify-between bg-matrix-raised border border-matrix-border
                          rounded px-4 py-2.5 mb-3 text-xs">
            <span className="text-neon font-medium">
              {selectedIds.size} seleccionat{selectedIds.size !== 1 ? 's' : ''}
            </span>
            <div className="flex items-center gap-3">
              <button
                onClick={toggleSelectAll}
                className="text-matrix-muted hover:text-neon uppercase tracking-widest transition-colors"
              >
                {selectedIds.size === matches.length ? 'Desseleccionar tots' : 'Seleccionar tots'}
              </button>
              <button
                onClick={handleDeleteSelected}
                className="flex items-center gap-1.5 bg-matrix-error hover:bg-[#cc3333]
                           text-matrix-bg font-bold px-3 py-1.5 rounded uppercase tracking-widest transition-colors"
              >
                <Trash2 size={11} />
                Eliminar ({selectedIds.size})
              </button>
            </div>
          </div>
        )}

        {/* Llista buida */}
        {!loading && !error && matches.length === 0 && (
          <div className="bg-matrix-card border border-matrix-border rounded-lg p-12 text-center">
            <Clock size={36} className="mx-auto text-matrix-muted mb-3" />
            <p className="text-matrix-muted text-xs uppercase tracking-widest mb-4">
              Cap vídeo pujat
            </p>
            <button
              onClick={() => navigate('/process')}
              className="inline-flex items-center gap-1.5 bg-neon hover:bg-neon-600
                         text-matrix-bg text-xs font-bold px-4 py-2 rounded
                         uppercase tracking-widest transition-colors"
            >
              <Plus size={13} /> Pujar primer vídeo
            </button>
          </div>
        )}

        {/* Llista de partits */}
        {!loading && !error && matches.length > 0 && (
          <div className="space-y-2">
            {matches.map((m) => (
              <div
                key={m.match_id}
                className={`rounded-lg border px-4 py-3.5 flex items-center gap-3 transition-colors
                  ${selectedIds.has(m.match_id)
                    ? 'border-neon bg-matrix-raised'
                    : m.status === 'error'
                      ? 'border-[#3a1a1a] bg-[#100808]'
                      : 'border-matrix-border bg-matrix-card'}`}
              >
                <input
                  type="checkbox"
                  checked={selectedIds.has(m.match_id)}
                  onChange={() => toggleSelect(m.match_id)}
                  className="w-3.5 h-3.5 shrink-0 accent-neon cursor-pointer"
                />

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <p className="text-xs font-semibold text-matrix-text truncate">{m.title}</p>
                    <StatusBadge status={m.status} />
                  </div>
                  <p className="text-xs text-matrix-muted">{formatDate(m.created_at)}</p>
                </div>

                <div className="flex flex-col items-end gap-1 shrink-0 text-right">
                  <span className="text-xs text-matrix-muted">
                    <span className="text-matrix-text">{formatDuration(m.start_seconds, m.end_seconds)}</span>
                  </span>
                  <span className="text-xs text-matrix-muted">
                    ROI:{' '}
                    {m.has_roi
                      ? <span className="text-neon font-bold">✓</span>
                      : <span className="text-matrix-muted">—</span>
                    }
                  </span>
                </div>

                {m.status === 'done' && (
                  <button
                    onClick={() => navigate(`/matches/${m.match_id}/results`)}
                    className="shrink-0 flex items-center gap-1 bg-neon hover:bg-neon-600
                               text-matrix-bg text-xs font-bold px-2.5 py-1.5 rounded
                               uppercase tracking-widest transition-colors"
                  >
                    <ExternalLink size={11} />
                    Veure
                  </button>
                )}

                {canProcess(m) && (
                  <button
                    onClick={() => handleProcess(m.match_id)}
                    disabled={processingIds.has(m.match_id)}
                    className={`shrink-0 flex items-center gap-1 disabled:opacity-40
                               text-matrix-bg text-xs font-bold px-2.5 py-1.5 rounded
                               uppercase tracking-widest transition-colors
                               ${isReprocess(m)
                                 ? 'bg-matrix-warning hover:bg-[#cc9200]'
                                 : 'bg-neon hover:bg-neon-600'}`}
                  >
                    {processingIds.has(m.match_id)
                      ? <Loader size={11} className="animate-spin" />
                      : isReprocess(m) ? <RotateCcw size={11} /> : <Play size={11} />
                    }
                    {isReprocess(m) ? 'Reset' : 'Process'}
                  </button>
                )}

                {confirmingId === m.match_id ? (
                  <div className="shrink-0 flex items-center gap-1.5">
                    <button
                      onClick={() => handleDeleteOne(m.match_id)}
                      disabled={deletingIds.has(m.match_id)}
                      className="flex items-center gap-1 bg-matrix-error hover:bg-[#cc3333]
                                 disabled:opacity-50 text-matrix-bg text-xs font-bold
                                 px-2.5 py-1.5 rounded uppercase tracking-widest transition-colors"
                    >
                      {deletingIds.has(m.match_id)
                        ? <Loader size={11} className="animate-spin" />
                        : <Trash2 size={11} />
                      }
                      OK
                    </button>
                    <button
                      onClick={() => setConfirmingId(null)}
                      className="text-xs text-matrix-muted hover:text-matrix-text px-1.5 py-1.5 transition-colors"
                    >
                      ✕
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setConfirmingId(m.match_id)}
                    className="shrink-0 p-1.5 text-matrix-muted hover:text-matrix-error
                               hover:bg-[#1a0d0d] rounded transition-colors"
                    title="Eliminar"
                  >
                    <Trash2 size={13} />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
