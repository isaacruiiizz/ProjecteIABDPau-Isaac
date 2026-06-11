import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogIn, AlertCircle } from 'lucide-react';
import { login } from '../api/auth';
import useAuthStore from '../store/authStore';
import { parseJwtRole } from '../utils/jwt';

export default function LoginPage() {
  const navigate = useNavigate();
  const setToken = useAuthStore((state) => state.setToken);
  const setRole  = useAuthStore((state) => state.setRole);

  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const [error,    setError]    = useState<string | null>(null);
  const [loading,  setLoading]  = useState(false);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const { access_token } = await login(email, password);
      setToken(access_token);
      setRole(parseJwtRole(access_token));
      navigate('/', { replace: true });
    } catch {
      setError('Credencials incorrectes. Torna-ho a intentar.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-matrix-bg flex items-center justify-center px-4">
      <div className="w-full max-w-sm">

        {/* Logo i títol */}
        <div className="flex flex-col items-center mb-8">
          <img src="/logo.png" alt="SmartChrono IP" className="h-40 w-auto mb-4 object-contain" />
          <h1 className="text-2xl font-bold text-neon neon-glow tracking-widest uppercase">
            SmartChrono<span className="text-matrix-text"> IP</span>
          </h1>
          <p className="text-xs text-matrix-muted mt-1 tracking-widest uppercase">
            Sistema d'anàlisi tàctica
          </p>
        </div>

        {/* Targeta */}
        <div className="bg-matrix-card border border-matrix-border rounded-lg p-8">
          <p className="text-matrix-muted text-xs uppercase tracking-widest mb-6">
            {'>'} Identificació requerida
          </p>
          <form onSubmit={handleSubmit} noValidate>
            <div className="mb-4">
              <label htmlFor="email" className="block text-xs font-medium text-matrix-muted uppercase tracking-wider mb-1.5">
                Correu electrònic
              </label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3 py-2.5 bg-matrix-input border border-matrix-border rounded
                           text-sm text-matrix-text placeholder-matrix-muted
                           focus:outline-none focus:border-neon focus:ring-1 focus:ring-neon
                           disabled:opacity-40 transition-colors"
                placeholder="usuari@club.cat"
                disabled={loading}
              />
            </div>

            <div className="mb-6">
              <label htmlFor="password" className="block text-xs font-medium text-matrix-muted uppercase tracking-wider mb-1.5">
                Contrasenya
              </label>
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-3 py-2.5 bg-matrix-input border border-matrix-border rounded
                           text-sm text-matrix-text placeholder-matrix-muted
                           focus:outline-none focus:border-neon focus:ring-1 focus:ring-neon
                           disabled:opacity-40 transition-colors"
                placeholder="••••••••"
                disabled={loading}
              />
            </div>

            {error && (
              <div className="flex items-start gap-2 text-matrix-error text-xs mb-4
                              bg-[#1a0d0d] border border-[#3a1a1a] rounded px-3 py-2">
                <AlertCircle size={14} className="mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2
                         bg-neon hover:bg-neon-600 disabled:opacity-40
                         text-matrix-bg font-bold py-2.5 px-4 rounded
                         text-sm uppercase tracking-widest transition-colors"
            >
              <LogIn size={15} />
              {loading ? 'Verificant...' : 'Accedir'}
            </button>
          </form>
        </div>

        <p className="text-center text-matrix-muted text-xs mt-6 tracking-widest">
          SC-IP v2.1 // ACCÉS RESTRINGIT
        </p>
      </div>
    </div>
  );
}
