import axios, {
  isAxiosError,
  type AxiosResponse,
  type InternalAxiosRequestConfig,
} from 'axios';
import useAuthStore from '../store/authStore';

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL as string,
  // withCredentials envia la cookie HttpOnly del refresh token automàticament.
  // JavaScript no pot llegir ni escriure la cookie — el navegador la gestiona.
  withCredentials: true,
});

// ── Request interceptor ──────────────────────────────────────────────────────
// Afegeix el Bearer token a totes les peticions si n'hi ha un al store.
apiClient.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Response interceptor ─────────────────────────────────────────────────────
// Si rep un 401, intenta fer refresh. Si el refresh també falla → logout.
let isRefreshing = false;
let pendingQueue: Array<{
  resolve: (token: string) => void;
  reject: (err: unknown) => void;
}> = [];

function flushQueue(token: string | null, error: unknown = null) {
  pendingQueue.forEach(({ resolve, reject }) => {
    if (token) resolve(token);
    else reject(error);
  });
  pendingQueue = [];
}

apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: unknown) => {
    if (!isAxiosError(error)) return Promise.reject(error);

    const originalRequest = error.config;

    // Els endpoints d'auth gestionen els seus propis errors → no intentar refresh
    const isAuthEndpoint = originalRequest?.url?.includes('/auth/');
    if (error.response?.status === 401 && isAuthEndpoint) {
      if (originalRequest?.url?.includes('/auth/refresh')) {
        // Refresh fallit → sessió caducada → logout
        useAuthStore.getState().clearToken();
        if (window.location.pathname !== '/login') {
          window.location.href = '/login';
        }
      }
      // Login fallit → deixar propagar l'error al catch del component
      return Promise.reject(error);
    }

    if (error.response?.status === 401 && !originalRequest?._retry) {
      originalRequest._retry = true;

      if (isRefreshing) {
        // Ja hi ha un refresh en curs — encua la petició i espera el nou token
        return new Promise((resolve, reject) => {
          pendingQueue.push({
            resolve: (token) => {
              originalRequest.headers.Authorization = `Bearer ${token}`;
              resolve(apiClient(originalRequest));
            },
            reject,
          });
        });
      }

      isRefreshing = true;

      try {
        const { data } = await apiClient.post<{ access_token: string }>(
          '/auth/refresh',
        );
        const newToken = data.access_token;
        useAuthStore.getState().setToken(newToken);
        originalRequest.headers.Authorization = `Bearer ${newToken}`;
        flushQueue(newToken);
        return apiClient(originalRequest);
      } catch (refreshError) {
        flushQueue(null, refreshError);
        useAuthStore.getState().clearToken();
        window.location.href = '/login';
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  },
);

export default apiClient;
