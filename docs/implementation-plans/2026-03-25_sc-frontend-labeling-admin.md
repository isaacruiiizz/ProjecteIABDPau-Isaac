# PJM-27 — Frontend: Secció admin d'etiquetatge

**Ticket:** PJM-27
**Estat:** Completat ✓ (2026-03-25)
**Data:** 2026-03-25
**Etiqueta Jira:** `frontend`

---

## Descripció

Implementar la pàgina `/admin/labeling` accessible únicament per rol `admin`. Inclou:
- Formulari d'upload de vídeo (crida `POST /api/v1/labeling/upload` del PJM-24)
- Selector de `frame_interval` (defecte: 2s)
- Visualització de l'estat de la sessió després de l'upload
- Botó/link directe a Label Studio (`VITE_LABEL_STUDIO_URL`, port `:8081`), obre en nova pestanya

---

## Context clau

- Endpoint backend: `POST /api/v1/labeling/upload` (PJM-24 ✓) — `multipart/form-data`, paràmetre `frame_interval` (query, defecte 2), retorna `202 { session_id, minio_key, status: "queued" }`
- Label Studio: `http://localhost:8081` (configurable via `VITE_LABEL_STUDIO_URL`)
- Rol admin: el JWT conté el camp `role`. S'ha de decodificar el payload del token (base64) i emmagatzemar `role` a Zustand (memòria). **Mai localStorage.**
- El `ProtectedRoute` existent comprova token; el nou `AdminRoute` comprova a més `role === 'admin'`

---

## Fitxers afectats / nous

| Fitxer | Acció |
|---|---|
| `services/sc-frontend/.env.example` | + `VITE_LABEL_STUDIO_URL=http://localhost:8081` |
| `src/types/auth.ts` | + camp `role: string \| null` + acció `setRole` a `AuthState` |
| `src/store/authStore.ts` | + estat `role`, acció `setRole` |
| `src/App.tsx` | + decodificació `role` del JWT en `refreshToken()`, + ruta `/admin/labeling` protegida per `AdminRoute` |
| `src/api/labeling.ts` | **Nou** — `uploadLabelingVideo(file, frameInterval)` |
| `src/components/AdminRoute.tsx` | **Nou** — redirigeix a `/` si `role !== 'admin'` |
| `src/pages/LabelingPage.tsx` | **Nou** — pàgina admin d'etiquetatge |

---

## Fases d'implementació

### Fase 1 — Entorn i tipus

**1.1 `.env.example`** — afegir:
```env
VITE_LABEL_STUDIO_URL=http://localhost:8081
```

**1.2 `src/types/auth.ts`** — ampliar `AuthState`:
```typescript
export interface AuthState {
  token: string | null;
  role: string | null;
  setToken: (token: string) => void;
  setRole: (role: string | null) => void;
  clearToken: () => void;
}
```

**1.3 `src/store/authStore.ts`** — afegir `role` i `setRole`:
```typescript
const useAuthStore = create<AuthState>((set) => ({
  token: null,
  role: null,
  setToken: (token) => set({ token }),
  setRole: (role) => set({ role }),
  clearToken: () => set({ token: null, role: null }),
}));
```

---

### Fase 2 — Decodificació JWT i `App.tsx`

El JWT payload (segment central base64url) conté el camp `role`. Caldrà una funció utilitària `parseJwtRole(token)` i cridar-la cada cop que es rep un access token nou.

**Utilitat (inline a `App.tsx`):**
```typescript
function parseJwtRole(token: string): string | null {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return payload.role ?? null;
  } catch {
    return null;
  }
}
```

**`App.tsx`** — modificacions:
1. Llegir `setRole` del store.
2. Cridar `setRole(parseJwtRole(access_token))` just després de `setToken(access_token)`.
3. Afegir ruta nova:
```tsx
<Route element={<AdminRoute />}>
  <Route path="/admin/labeling" element={<LabelingPage />} />
</Route>
```
4. Afegir imports: `AdminRoute`, `LabelingPage`.

---

### Fase 3 — `AdminRoute` i `api/labeling.ts`

**`src/components/AdminRoute.tsx`:**
```tsx
import { Navigate, Outlet } from 'react-router-dom';
import useAuthStore from '../store/authStore';

export default function AdminRoute() {
  const token = useAuthStore((s) => s.token);
  const role  = useAuthStore((s) => s.role);

  if (!token) return <Navigate to="/login" replace />;
  if (role !== 'admin') return <Navigate to="/" replace />;
  return <Outlet />;
}
```

**`src/api/labeling.ts`:**
```typescript
import apiClient from './client';

export interface LabelingUploadResponse {
  session_id: string;
  minio_key: string;
  status: string;
}

export async function uploadLabelingVideo(
  file: File,
  frameInterval = 2,
): Promise<LabelingUploadResponse> {
  const form = new FormData();
  form.append('video', file);
  const { data } = await apiClient.post<LabelingUploadResponse>(
    `/api/v1/labeling/upload?frame_interval=${frameInterval}`,
    form,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  );
  return data;
}
```

---

### Fase 4 — `LabelingPage.tsx`

Estat del component:
- `file: File | null` — fitxer seleccionat
- `frameInterval: number` — defecte 2
- `uploading: boolean` — mostra spinner mentre s'envia
- `result: LabelingUploadResponse | null` — resposta de l'API
- `error: string | null` — missatge d'error

Estructura visual (Tailwind):
```
┌──────────────────────────────────────────┐
│  [← Dashboard]  Etiquetatge de vídeos    │
├──────────────────────────────────────────┤
│  UPLOAD NOU VÍDEO                        │
│  ┌────────────────────────────────────┐  │
│  │  📎 Selecciona fitxer .mp4         │  │
│  └────────────────────────────────────┘  │
│  Interval de frames: [2] s               │
│  [Pujar vídeo]                           │
│                                          │
│  (un cop pujat)                          │
│  ✓ Vídeo encoat correctament             │
│  Session ID: b7e2f1a0-...               │
│  Estat: queued                           │
├──────────────────────────────────────────┤
│  ETIQUETATGE                             │
│  [Obrir Label Studio →]  (nova pestanya) │
└──────────────────────────────────────────┘
```

- Botó "Obrir Label Studio" sempre visible: `<a href={import.meta.env.VITE_LABEL_STUDIO_URL} target="_blank" rel="noreferrer">`
- Validació client: accept=".mp4, video/mp4" + comprovació `file.type`
- Error de l'API: mostrar `error.response.data.detail` si disponible

---

### Fase 5 — Verificació

1. Iniciar sessió com a `admin` → `/admin/labeling` accessible.
2. Iniciar sessió com a un altre rol → `/admin/labeling` redirigeix a `/`.
3. Seleccionar un `.mp4` → botó "Pujar vídeo" actiu.
4. Clicar "Pujar vídeo" → spinner → `session_id` visible.
5. Clicar "Obrir Label Studio" → s'obre `http://localhost:8081` en nova pestanya.
6. Sense autenticació → redirigeix a `/login`.

---

## Regles crítiques (recordatori)

- `role` **mai** a localStorage — es guarda a Zustand (memòria).
- Nom fitxers: `AdminRoute.tsx` i `LabelingPage.tsx` (PascalCase), `labeling.ts` (kebab-case).
- El botó Label Studio: `target="_blank" rel="noreferrer"` obligatori.
- `clearToken` ha de netejar també `role` (ja inclòs al pla).
