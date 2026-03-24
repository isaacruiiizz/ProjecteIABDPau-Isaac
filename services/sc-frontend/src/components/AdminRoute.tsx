import { Navigate, Outlet } from 'react-router-dom';
import useAuthStore from '../store/authStore';

export default function AdminRoute() {
  const token = useAuthStore((s) => s.token);
  const role  = useAuthStore((s) => s.role);

  if (!token) return <Navigate to="/login" replace />;
  if (role !== 'admin') return <Navigate to="/" replace />;
  return <Outlet />;
}
