import { useEffect, useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { refreshToken } from './api/auth';
import AdminRoute from './components/AdminRoute';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import LabelingPage from './pages/LabelingPage';
import ProcessPage from './pages/ProcessPage';
import MatchesPage from './pages/MatchesPage';
import useAuthStore from './store/authStore';
import type { AuthState } from './types/auth';
import { parseJwtRole } from './utils/jwt';

export default function App() {
  // Mentre es comprova la sessió, no renderitzem les rutes per evitar
  // un flash de redirecció innecessari cap a /login.
  const [checking, setChecking] = useState(true);
  const setToken = useAuthStore((state: AuthState) => state.setToken);
  const setRole  = useAuthStore((state: AuthState) => state.setRole);

  useEffect(() => {
    refreshToken()
      .then(({ access_token }) => {
        setToken(access_token);
        setRole(parseJwtRole(access_token));
      })
      .catch(() => {/* sessió caducada o no iniciada — ProtectedRoute redirigirà */})
      .finally(() => setChecking(false));
  }, [setToken, setRole]);

  if (checking) return null;

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<ProtectedRoute />}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/process" element={<ProcessPage />} />
          <Route path="/matches" element={<MatchesPage />} />
        </Route>
        <Route element={<AdminRoute />}>
          <Route path="/admin/labeling" element={<LabelingPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
