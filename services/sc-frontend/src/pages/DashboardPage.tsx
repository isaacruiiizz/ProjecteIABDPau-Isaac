import { useNavigate } from 'react-router-dom';
import { Timer, LogOut } from 'lucide-react';
import useAuthStore from '../store/authStore';

export default function DashboardPage() {
  const navigate = useNavigate();
  const clearToken = useAuthStore((state) => state.clearToken);

  function handleLogout() {
    clearToken();
    navigate('/login', { replace: true });
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center px-4">
      <div className="flex items-center gap-3 mb-6">
        <div className="bg-blue-600 text-white rounded-2xl p-3">
          <Timer size={28} />
        </div>
        <h1 className="text-2xl font-bold text-gray-900">SmartChrono IP</h1>
      </div>

      <p className="text-gray-600 mb-8">Benvingut. Sessió iniciada correctament.</p>

      <button
        onClick={handleLogout}
        className="flex items-center gap-2 bg-white hover:bg-gray-100 border border-gray-300 text-gray-700 font-medium py-2 px-4 rounded-lg text-sm transition-colors"
      >
        <LogOut size={16} />
        Tanca sessió
      </button>
    </div>
  );
}
