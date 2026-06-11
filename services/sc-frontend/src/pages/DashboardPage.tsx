import { useNavigate } from 'react-router-dom';
import { Film, LogOut, Tag, Zap } from 'lucide-react';
import useAuthStore from '../store/authStore';

export default function DashboardPage() {
  const navigate   = useNavigate();
  const clearToken = useAuthStore((s) => s.clearToken);
  const role       = useAuthStore((s) => s.role);

  function handleLogout() {
    clearToken();
    navigate('/login', { replace: true });
  }

  return (
    <div className="min-h-screen bg-matrix-bg flex flex-col items-center justify-center px-4">

      {/* Header */}
      <div className="flex flex-col items-center mb-2">
        <img src="/logo.png" alt="SmartChrono IP" className="h-36 w-auto mb-4 object-contain" />
        <h1 className="text-3xl font-bold text-neon neon-glow tracking-widest uppercase">
          SmartChrono IP
        </h1>
      </div>
      <p className="text-matrix-muted text-xs tracking-widest uppercase mb-10">
        {'>'} Sistema d'anàlisi tàctica de futbol sala
      </p>

      {/* Targetes d'acció */}
      <div className="flex flex-col sm:flex-row sm:items-stretch gap-4 w-full max-w-2xl">

        {/* Analitzar partit */}
        <button
          onClick={() => navigate('/process')}
          className="flex-1 flex flex-col bg-matrix-card border border-matrix-border rounded-lg p-6
                     text-left hover:border-neon hover:bg-matrix-raised transition-all group"
        >
          <div className="flex items-center gap-2 mb-4">
            <div className="bg-[#001a08] text-neon border border-[#1a3a1a] rounded p-2.5 w-fit
                            group-hover:bg-neon group-hover:text-matrix-bg transition-colors">
              <Zap size={22} />
            </div>
            <span className="text-matrix-muted text-xs uppercase tracking-widest">01</span>
          </div>
          <h2 className="text-sm font-bold text-neon uppercase tracking-wider mb-2">
            Analitzar Partit
          </h2>
          <p className="flex-1 text-xs text-matrix-text leading-relaxed mb-4">
            Processa un vídeo de partit. El sistema detecta els jugadors per color de samarreta
            i genera un informe tàctic automàtic.
          </p>
          <span className="text-xs font-medium text-neon uppercase tracking-widest">
            {'>'} Iniciar →
          </span>
        </button>

        {/* Els meus partits */}
        <button
          onClick={() => navigate('/matches')}
          className="flex-1 flex flex-col bg-matrix-card border border-matrix-border rounded-lg p-6
                     text-left hover:border-neon hover:bg-matrix-raised transition-all group"
        >
          <div className="flex items-center gap-2 mb-4">
            <div className="bg-[#001a08] text-neon border border-[#1a3a1a] rounded p-2.5 w-fit
                            group-hover:bg-neon group-hover:text-matrix-bg transition-colors">
              <Film size={22} />
            </div>
            <span className="text-matrix-muted text-xs uppercase tracking-widest">02</span>
          </div>
          <h2 className="text-sm font-bold text-neon uppercase tracking-wider mb-2">
            Els Meus Partits
          </h2>
          <p className="flex-1 text-xs text-matrix-text leading-relaxed mb-4">
            Consulta l'estat dels vídeos pujats, visualitza els resultats i
            descarrega els vídeos processats.
          </p>
          <span className="text-xs font-medium text-neon uppercase tracking-widest">
            {'>'} Veure registres →
          </span>
        </button>

        {/* Etiquetar frames — només admins */}
        {role === 'admin' && (
          <button
            onClick={() => navigate('/admin/labeling')}
            className="flex-1 flex flex-col bg-matrix-card border border-matrix-border rounded-lg p-6
                       text-left hover:border-neon hover:bg-matrix-raised transition-all group"
          >
            <div className="flex items-center gap-2 mb-4">
              <div className="bg-[#001a08] text-neon border border-[#1a3a1a] rounded p-2.5 w-fit
                              group-hover:bg-neon group-hover:text-matrix-bg transition-colors">
                <Tag size={22} />
              </div>
              <span className="text-matrix-muted text-xs uppercase tracking-widest">ADM</span>
            </div>
            <h2 className="text-sm font-bold text-neon uppercase tracking-wider mb-2">
              Etiquetar Frames
            </h2>
            <p className="flex-1 text-xs text-matrix-text leading-relaxed mb-4">
              Puja vídeos per etiquetar frames manualment i millorar el model de detecció.
            </p>
            <span className="text-xs font-medium text-neon uppercase tracking-widest">
              {'>'} Mode admin →
            </span>
          </button>
        )}
      </div>

      {/* Logout */}
      <button
        onClick={handleLogout}
        className="mt-10 flex items-center gap-2 text-xs text-matrix-muted
                   hover:text-matrix-error uppercase tracking-widest transition-colors"
      >
        <LogOut size={14} />
        Tancar sessió
      </button>
    </div>
  );
}
