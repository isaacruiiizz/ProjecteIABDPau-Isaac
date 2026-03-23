# Pla d'Implementació — PJM-20
# [frontend] Crear pantalla de login amb Vite + Tailwind + crida real a l'API

**Estat:** Completat ✓ (2026-03-24)
**Ticket:** PJM-20
**Data:** 2026-03-24
**Etiqueta commit:** `feat(frontend): crear pantalla de login amb Vite + Tailwind + crida real a l'API [PJM-20]`

---

## Context

El servei `sc-frontend` és ara un placeholder buit. Cal inicialitzar el projecte React 18 + Vite + Tailwind CSS i implementar la pantalla de login completa amb crida real a `POST /auth/login`, gestió del token en memòria (Zustand) i interceptor Axios per al refresh automàtic.

---

## Estructura final de `services/sc-frontend/`

```
services/sc-frontend/
├── Dockerfile                 ← ja existent (multistage node:22-alpine)
├── .env.example               ← ja existent
├── index.html                 ← MODIFICAR (afegir script src)
├── package.json               ← REESCRIURE (deps reals)
├── vite.config.ts             ← NOU
├── tailwind.config.ts         ← NOU
├── tsconfig.json              ← NOU
├── tsconfig.node.json         ← NOU
├── postcss.config.js          ← NOU
└── src/
    ├── main.tsx               ← NOU
    ├── App.tsx                ← NOU (router + ruta protegida)
    ├── index.css              ← NOU (directives Tailwind)
    ├── types/
    │   └── auth.ts            ← NOU (interfaces TypeScript)
    ├── store/
    │   └── authStore.ts       ← NOU (Zustand — access token en memòria)
    ├── api/
    │   ├── client.ts          ← NOU (axios instance + interceptor refresh)
    │   └── auth.ts            ← NOU (login(), refreshToken())
    ├── components/
    │   └── ProtectedRoute.tsx ← NOU (redirigeix a /login si no autenticat)
    └── pages/
        ├── LoginPage.tsx      ← NOU (formulari email + password)
        └── DashboardPage.tsx  ← NOU (placeholder mínim post-login)
```

---

## Dependències

### `dependencies`
| Paquet | Versió | Ús |
|---|---|---|
| `react` | `^18.3.0` | UI |
| `react-dom` | `^18.3.0` | DOM |
| `react-router-dom` | `^6.28.0` | Routing |
| `zustand` | `^5.0.0` | Access token en memòria |
| `axios` | `^1.7.0` | Client HTTP + interceptor |
| `lucide-react` | `^0.468.0` | Icones |

### `devDependencies`
| Paquet | Versió | Ús |
|---|---|---|
| `vite` | `^6.0.0` | Build tool |
| `@vitejs/plugin-react` | `^4.3.0` | Plugin React per Vite |
| `typescript` | `^5.7.0` | TypeScript |
| `@types/react` | `^18.3.0` | Tipus React |
| `@types/react-dom` | `^18.3.0` | Tipus React DOM |
| `tailwindcss` | `^3.4.0` | Estils |
| `autoprefixer` | `^10.4.0` | PostCSS vendor prefixes |
| `postcss` | `^8.4.0` | Processador CSS |

---

## Flux d'autenticació (frontend)

```
Usuari envia email + password
        │
        ▼
api/auth.ts: login(email, password)
  POST /auth/login  {withCredentials: true}
        │
        ├── 401 → mostrar error "Credencials incorrectes"
        │
        └── 200 → { access_token, token_type }
              │
              ▼
        authStore.setToken(access_token)
              │
              ▼
        navigate('/') → DashboardPage
```

### Interceptor Axios (refresh automàtic)
```
Petició amb Authorization: Bearer <token>
        │
        └── Resposta 401?
              │
              ├── És la crida a /auth/refresh? → logout() → navigate('/login')
              │
              └── No → POST /auth/refresh  {withCredentials: true}
                          │
                          ├── 200 → nou access_token
                          │     authStore.setToken(nou_token)
                          │     Reintenta petició original
                          │
                          └── 401 → logout() → navigate('/login')
```

---

## Detall de cada fitxer

### `src/types/auth.ts`
```typescript
export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface AuthState {
  token: string | null;
  setToken: (token: string) => void;
  clearToken: () => void;
}
```

### `src/store/authStore.ts`
- Zustand store amb `token: string | null`
- `setToken(token)` i `clearToken()`
- **Mai `localStorage`** — estat en memòria pura

### `src/api/client.ts`
- `axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL, withCredentials: true })`
- `request interceptor`: afegeix `Authorization: Bearer <token>` si hi ha token al store
- `response interceptor`: si 401 → crida `/auth/refresh` → actualitza token → retry
  - Si el refresh també falla → `clearToken()` → `window.location.href = '/login'`

### `src/api/auth.ts`
- `login(email, password): Promise<TokenResponse>` → `POST /auth/login`
- `refreshToken(): Promise<TokenResponse>` → `POST /auth/refresh`

### `src/pages/LoginPage.tsx`
- Formulari: camp `email` (type email) + camp `password` (type password) + botó "Inicia sessió"
- Estat local: `email`, `password`, `error: string | null`, `loading: boolean`
- En submit: crida `api/auth.login()` → si OK guarda token i navega a `/`
- Error visible sota el formulari (missatge genèric, mai detalls interns)
- Disseny: centrat verticalment i horitzontalment, fons `gray-50`, targeta blanca amb ombra, logo/títol "SmartChrono IP" + icona `Timer` de Lucide

### `src/pages/DashboardPage.tsx`
- Placeholder mínim: missatge de benvinguda + botó "Tanca sessió"
- El logout crida `clearToken()` i navega a `/login`

### `src/components/ProtectedRoute.tsx`
- Si `token === null` → `<Navigate to="/login" replace />`
- Si `token` existeix → renderitza `<Outlet />`

### `src/App.tsx`
```
<Router>
  <Routes>
    <Route path="/login" element={<LoginPage />} />
    <Route element={<ProtectedRoute />}>
      <Route path="/" element={<DashboardPage />} />
    </Route>
    <Route path="*" element={<Navigate to="/" replace />} />
  </Routes>
</Router>
```

---

## Consideracions tècniques

### Cookie HttpOnly
- El Refresh Token arriba via `Set-Cookie` del servidor
- `withCredentials: true` a totes les crides fa que el navegador enviï la cookie automàticament
- JavaScript **no pot llegir** la cookie → és transparentament gestionada pel navegador

### Access Token
- Guardat únicament al store Zustand (memòria)
- Es perd en recarregar la pàgina → el interceptor cridarà `/auth/refresh` automàticament
- Mai a `localStorage` ni `sessionStorage`

### Evitar bucle infinit al refresh
- Quan el interceptor detecta 401, comprova si la URL original ja és `/auth/refresh`
- Si és així → no reintenta → logout directe

### Variables d'entorn Vite
- `import.meta.env.VITE_API_BASE_URL` per a l'URL base de l'API
- Vite exposa només les variables amb prefix `VITE_`

---

## Fases d'implementació

1. `package.json` + fitxers de config (`vite.config.ts`, `tailwind.config.ts`, `tsconfig.json`, `tsconfig.node.json`, `postcss.config.js`)
2. `index.html` + `src/index.css` + `src/main.tsx`
3. `src/types/auth.ts`
4. `src/store/authStore.ts`
5. `src/api/client.ts` + `src/api/auth.ts`
6. `src/components/ProtectedRoute.tsx`
7. `src/pages/LoginPage.tsx` + `src/pages/DashboardPage.tsx`
8. `src/App.tsx`
9. Crear el fitxer amb els endpoints documentat
10. Actualitzar `CLAUDE.md`
