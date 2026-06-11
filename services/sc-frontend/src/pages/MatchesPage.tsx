import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertCircle, CheckCircle, Clock, ExternalLink, Loader, Play, Plus, RotateCcw, Timer, Trash2, XCircle,
} from 'lucide-react';
import { deleteMatch, getMatches, processMatch, type MatchListItem } from '../api/matches';

const STATUS_CONFIG: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  pending:      { label: 'Pendent',        color: 'bg-gray-100 text-gray-600',     icon: <Clock size={12} /> },
  processing:   { label: 'Processant',     color: 'bg-yellow-100 text-yellow-700', icon: <Loader size={12} className="animate-spin" /> },
  frames_ready: { label: 'Frames extrets', color: 'bg-blue-100 text-blue-700',     icon: <Loader size={12} /> },
  done:         { label: 'Completat',      color: 'bg-green-100 text-green-700',   icon: <CheckCircle size={12} /> },
  error:        { label: 'Error',          color: 'bg-red-100 text-red-700',       icon: <XCircle size={12} /> },
};

function StatusBadge({ status }: { status: string }) {
  const cfg = STATUS_CONFIG[status] ?? {
    label: status, color: 'bg-gray-100 text-gray-600', icon: null,
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full
                      text-xs font-medium ${cfg.color}`}>
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
    m.has_roi &&
    m.start_seconds !== null;
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
      setProcessingIds(prev => {
        const next = new Set(prev);
        next.delete(matchId);
        return next;
      });
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
    if (selectedIds.size === matches.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(matches.map(m => m.match_id)));
    }
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
    <div className="min-h-screen bg-gray-50 px-4 py-8">
      <div className="max-w-2xl mx-auto">

        {/* Capçalera */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 text-white rounded-2xl p-2.5">
              <Timer size={22} />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900">Els meus partits</h1>
              <p className="text-xs text-gray-500">Vídeos pujats i estat de processament</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/')}
              className="text-sm text-gray-500 hover:text-gray-700 transition-colors"
            >
              ← Tornar
            </button>
            <button
              onClick={() => navigate('/process')}
              className="flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700 text-white
                         text-sm font-medium px-3 py-1.5 rounded-lg transition-colors"
            >
              <Plus size={15} /> Nou
            </button>
          </div>
        </div>

        {/* Carregant */}
        {loading && (
          <div className="flex items-center justify-center py-16">
            <Loader size={24} className="animate-spin text-blue-500" />
          </div>
        )}

        {/* Error càrrega */}
        {error && (
          <div className="flex items-start gap-2 text-red-600 text-sm bg-red-50
                          border border-red-200 rounded-xl px-4 py-3">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Error processament */}
        {processError && (
          <div className="flex items-start gap-2 text-red-600 text-sm bg-red-50
                          border border-red-200 rounded-xl px-4 py-3 mb-3">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span>{processError}</span>
          </div>
        )}

        {/* Error eliminació */}
        {deleteError && (
          <div className="flex items-start gap-2 text-red-600 text-sm bg-red-50
                          border border-red-200 rounded-xl px-4 py-3 mb-3">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span>{deleteError}</span>
          </div>
        )}

        {/* Barra d'accions en bloc */}
        {selectedIds.size > 0 && (
          <div className="flex items-center justify-between bg-blue-50 border border-blue-200
                          rounded-xl px-4 py-2.5 mb-3 text-sm">
            <span className="text-blue-700 font-medium">
              {selectedIds.size} seleccionat{selectedIds.size !== 1 ? 's' : ''}
            </span>
            <div className="flex items-center gap-3">
              <button
                onClick={toggleSelectAll}
                className="text-blue-600 hover:text-blue-800 text-xs font-medium transition-colors"
              >
                {selectedIds.size === matches.length ? 'Desseleccionar tots' : 'Seleccionar tots'}
              </button>
              <button
                onClick={handleDeleteSelected}
                className="flex items-center gap-1.5 bg-red-600 hover:bg-red-700
                           text-white text-xs font-medium px-3 py-1.5 rounded-lg transition-colors"
              >
                <Trash2 size={12} />
                Eliminar ({selectedIds.size})
              </button>
            </div>
          </div>
        )}

        {/* Llista buida */}
        {!loading && !error && matches.length === 0 && (
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-12 text-center">
            <Clock size={40} className="mx-auto text-gray-300 mb-3" />
            <p className="text-gray-500 text-sm mb-4">Encara no has pujat cap vídeo.</p>
            <button
              onClick={() => navigate('/process')}
              className="inline-flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700
                         text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
            >
              <Plus size={15} /> Pujar primer vídeo
            </button>
          </div>
        )}

        {/* Llista de partits */}
        {!loading && !error && matches.length > 0 && (
          <div className="space-y-3">
            {matches.map((m) => (
              <div
                key={m.match_id}
                className={`rounded-2xl border shadow-sm px-5 py-4 flex items-center gap-4
                            transition-colors ${selectedIds.has(m.match_id)
                              ? 'border-blue-300 bg-blue-50/40'
                              : m.status === 'error'
                                ? 'border-red-200 bg-red-50/40'
                                : 'border-gray-200 bg-white'}`}
              >
                {/* Checkbox selecció */}
                <input
                  type="checkbox"
                  checked={selectedIds.has(m.match_id)}
                  onChange={() => toggleSelect(m.match_id)}
                  className="w-4 h-4 shrink-0 accent-blue-600 cursor-pointer"
                />

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <p className="text-sm font-semibold text-gray-900 truncate">{m.title}</p>
                    <StatusBadge status={m.status} />
                  </div>
                  <p className="text-xs text-gray-400">{formatDate(m.created_at)}</p>
                </div>

                <div className="flex flex-col items-end gap-1 shrink-0 text-right">
                  <span className="text-xs text-gray-500">
                    Durada:{' '}
                    <span className="font-medium text-gray-700">
                      {formatDuration(m.start_seconds, m.end_seconds)}
                    </span>
                  </span>
                  <span className="text-xs text-gray-500">
                    ROI:{' '}
                    {m.has_roi
                      ? <span className="text-green-600 font-medium">✓</span>
                      : <span className="text-gray-400">—</span>
                    }
                  </span>
                </div>

                {m.status === 'done' && (
                  <button
                    onClick={() => navigate(`/matches/${m.match_id}/results`)}
                    className="shrink-0 flex items-center gap-1.5 bg-blue-600
                               hover:bg-blue-700 text-white text-xs font-medium
                               px-3 py-1.5 rounded-lg transition-colors"
                  >
                    <ExternalLink size={12} />
                    Veure resultats
                  </button>
                )}

                {canProcess(m) && (
                  <button
                    onClick={() => handleProcess(m.match_id)}
                    disabled={processingIds.has(m.match_id)}
                    className={`shrink-0 flex items-center gap-1.5 disabled:opacity-50
                               disabled:cursor-not-allowed text-white text-xs
                               font-medium px-3 py-1.5 rounded-lg transition-colors
                               ${isReprocess(m)
                                 ? 'bg-orange-500 hover:bg-orange-600'
                                 : 'bg-blue-600 hover:bg-blue-700'}`}
                  >
                    {processingIds.has(m.match_id)
                      ? <Loader size={12} className="animate-spin" />
                      : isReprocess(m) ? <RotateCcw size={12} /> : <Play size={12} />
                    }
                    {isReprocess(m) ? 'Reprocessar' : 'Processar'}
                  </button>
                )}

                {/* Botó eliminar individual */}
                {confirmingId === m.match_id ? (
                  <div className="shrink-0 flex items-center gap-1.5">
                    <button
                      onClick={() => handleDeleteOne(m.match_id)}
                      disabled={deletingIds.has(m.match_id)}
                      className="flex items-center gap-1 bg-red-600 hover:bg-red-700
                                 disabled:opacity-50 text-white text-xs font-medium
                                 px-2.5 py-1.5 rounded-lg transition-colors"
                    >
                      {deletingIds.has(m.match_id)
                        ? <Loader size={11} className="animate-spin" />
                        : <Trash2 size={11} />
                      }
                      Confirmar
                    </button>
                    <button
                      onClick={() => setConfirmingId(null)}
                      className="text-xs text-gray-500 hover:text-gray-700 px-1.5 py-1.5
                                 transition-colors"
                    >
                      Cancel·lar
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setConfirmingId(m.match_id)}
                    className="shrink-0 p-1.5 text-gray-400 hover:text-red-500
                               hover:bg-red-50 rounded-lg transition-colors"
                    title="Eliminar partit"
                  >
                    <Trash2 size={15} />
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
