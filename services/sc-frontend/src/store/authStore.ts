import { create } from 'zustand';
import type { AuthState } from '../types/auth';

// Access token i role guardats ÚNICAMENT en memòria (Zustand).
// Mai a localStorage ni sessionStorage — es perden en recarregar la pàgina
// i el refresh automàtic via cookie HttpOnly els recupera de forma transparent.
const useAuthStore = create<AuthState>((set) => ({
  token: null,
  role: null,
  setToken: (token) => set({ token }),
  setRole: (role) => set({ role }),
  clearToken: () => set({ token: null, role: null }),
}));

export default useAuthStore;
