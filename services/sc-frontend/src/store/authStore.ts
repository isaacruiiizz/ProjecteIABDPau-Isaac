import { create } from 'zustand';
import type { AuthState } from '../types/auth';

// Access token guardat ÚNICAMENT en memòria (Zustand).
// Mai a localStorage ni sessionStorage — es perd en recarregar la pàgina
// i el refresh automàtic via cookie HttpOnly el recupera de forma transparent.
const useAuthStore = create<AuthState>((set) => ({
  token: null,
  setToken: (token) => set({ token }),
  clearToken: () => set({ token: null }),
}));

export default useAuthStore;
