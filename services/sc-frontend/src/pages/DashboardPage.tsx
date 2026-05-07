import { useNavigate } from 'react-router-dom';
import { Clock, Film, LogOut, Tag, Timer } from 'lucide-react';
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
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center px-4">
      {/* Header */}
      <div className="flex items-center gap-3 mb-2">
        <div className="bg-blue-600 text-white rounded-2xl p-3">
          <Timer size={28} />
        </div>
        <h1 className="text-2xl font-bold text-gray-900">SmartChrono IP</h1>
      </div>
      <p className="text-gray-500 text-sm mb-10">Benvingut</p>

      {/* Targetes d'acció */}
      <div className="flex flex-col sm:flex-row gap-4 w-full max-w-2xl">

        {/* Calcular minuts */}
        <button
          onClick={() => navigate('/process')}
          className="flex-1 bg-white rounded-2xl border border-gray-200 shadow-sm p-6
                     text-left hover:border-blue-300 hover:shadow-md transition-all group"
        >
          <div className="bg-blue-100 text-blue-600 rounded-xl p-3 w-fit mb-4
                          group-hover:bg-blue-600 group-hover:text-white transition-colors">
            <Clock size={24} />
          </div>
          <h2 className="text-base font-semibold text-gray-900 mb-1">Calcular minuts</h2>
          <p className="text-sm text-gray-500 mb-4">
            Processa un vídeo de partit i obté automàticament els minuts jugats per cada jugador.
          </p>
          <span className="text-sm font-medium text-blue-600 group-hover:underline">
            Començar →
          </span>
        </button>

        {/* Els meus partits */}
        <button
          onClick={() => navigate('/matches')}
          className="flex-1 bg-white rounded-2xl border border-gray-200 shadow-sm p-6
                     text-left hover:border-emerald-300 hover:shadow-md transition-all group"
        >
          <div className="bg-emerald-100 text-emerald-600 rounded-xl p-3 w-fit mb-4
                          group-hover:bg-emerald-600 group-hover:text-white transition-colors">
            <Film size={24} />
          </div>
          <h2 className="text-base font-semibold text-gray-900 mb-1">Els meus partits</h2>
          <p className="text-sm text-gray-500 mb-4">
            Consulta l'estat dels vídeos pujats i el progrés de cada processament.
          </p>
          <span className="text-sm font-medium text-emerald-600 group-hover:underline">
            Veure partits →
          </span>
        </button>

        {/* Etiquetar frames — només admins */}
        {role === 'admin' && (
          <button
            onClick={() => navigate('/admin/labeling')}
            className="flex-1 bg-white rounded-2xl border border-gray-200 shadow-sm p-6
                       text-left hover:border-indigo-300 hover:shadow-md transition-all group"
          >
            <div className="bg-indigo-100 text-indigo-600 rounded-xl p-3 w-fit mb-4
                            group-hover:bg-indigo-600 group-hover:text-white transition-colors">
              <Tag size={24} />
            </div>
            <h2 className="text-base font-semibold text-gray-900 mb-1">Etiquetar frames</h2>
            <p className="text-sm text-gray-500 mb-4">
              Puja vídeos per etiquetar frames manualment i millorar el model de detecció.
            </p>
            <span className="text-sm font-medium text-indigo-600 group-hover:underline">
              Etiquetar →
            </span>
          </button>
        )}
      </div>

      {/* Logout */}
      <button
        onClick={handleLogout}
        className="mt-8 flex items-center gap-2 text-sm text-gray-500
                   hover:text-gray-700 transition-colors"
      >
        <LogOut size={15} />
        Tanca sessió
      </button>
    </div>
  );
}
