import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Timer, LogIn, AlertCircle } from 'lucide-react';
import { login } from '../api/auth';
import useAuthStore from '../store/authStore';

export default function LoginPage() {
  const navigate = useNavigate();
  const setToken = useAuthStore((state) => state.setToken);

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const { access_token } = await login(email, password);
      setToken(access_token);
      navigate('/', { replace: true });
    } catch {
      setError('Credencials incorrectes. Torna-ho a intentar.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        {/* Logo i títol */}
        <div className="flex flex-col items-center mb-8">
          <div className="bg-blue-600 text-white rounded-2xl p-3 mb-4">
            <Timer size={32} />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">SmartChrono IP</h1>
          <p className="text-sm text-gray-500 mt-1">Accedeix al teu compte</p>
        </div>

        {/* Targeta */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8">
          <form onSubmit={handleSubmit} noValidate>
            <div className="mb-4">
              <label
                htmlFor="email"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Correu electrònic
              </label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-400"
                placeholder="usuari@club.cat"
                disabled={loading}
              />
            </div>

            <div className="mb-6">
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Contrasenya
              </label>
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-400"
                placeholder="••••••••"
                disabled={loading}
              />
            </div>

            {error && (
              <div className="flex items-start gap-2 text-red-600 text-sm mb-4 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium py-2.5 px-4 rounded-lg text-sm transition-colors"
            >
              <LogIn size={16} />
              {loading ? 'Accedint...' : 'Inicia sessió'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
