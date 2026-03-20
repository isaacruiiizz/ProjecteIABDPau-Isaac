# Seguretat i BD

## 1. Seguretat i Autenticació (JWT)

- **Estratègia:** Doble token amb Access Token de vida curta i Refresh Token persistent.
    - **Access Token:** JWT signat (HS256) amb vida de 15 minuts. S'envia a cada petició via capçalera `Authorization: Bearer`. El payload conté `user_id`, `roles[]` i `team_id`. Mai conté dades sensibles.
    - **Refresh Token:** Token opac de 7 dies emmagatzemat en cookie `HttpOnly` + `Secure` + `SameSite=Strict`. JavaScript no hi pot accedir mai (protecció XSS).
    - **Refresh Token Rotation:** Cada crida a `/auth/refresh` invalida el token anterior i n'emet un de nou. Si un token robat intenta fer refresh un cop ja ha estat rotat, el sistema el detecta com a reutilització i invalida tota la sessió immediatament.
    - **Blacklist de sessions:** Els Refresh Tokens actius es registren a Redis amb TTL de 7 dies. En fer logout o revocar una sessió, el token s'afegeix a la blacklist i queda inutilitzable immediatament.
    - **Endpoints d'autenticació:**
        - `POST /auth/login` → emet ambdós tokens.
        - `POST /auth/refresh` → valida cookie, comprova blacklist Redis, invalida token antic, emet nou Refresh Token + nou Access Token.
        - `POST /auth/logout` → invalida Refresh Token a Redis.
    - **Frontend (React):** L'Access Token es guarda únicament en memòria (Zustand), mai a `localStorage`. El refresc és automàtic i transparent per a l'usuari mitjançant un interceptor d'Axios.

- **Secret de signatura JWT:**
    - `JWT_SECRET` és una variable d'entorn injectada en temps d'execució. Mai no pot estar hardcoded al codi ni commited al repositori.
    - Mínim 256 bits d'entropia (32 bytes aleatoris, generats amb `openssl rand -hex 32`).
    - Gestió: variable en fitxer `.env` per a entorns locals (exclòs de `.gitignore`), i Docker Secret o variable d'entorn xifrada per a producció.
    - Exemple de variables d'entorn requerides a `sc-api-gateway`:
        ```env
        JWT_SECRET=<mínim 32 bytes aleatoris>
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
        JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
        ```

- **Autenticació entre microserveis (Internal API Key):**
    - Els serveis interns (`sc-logic-aggregator`, `sc-inference-worker`, `sc-video-manager`, `sc-active-learner`) no accepten peticions sense autenticació, ni tan sols des de la xarxa Docker interna.
    - Cada servei intern valida una capçalera `X-Internal-API-Key` a totes les peticions rebudes.
    - La clau es comparteix entre serveis via variable d'entorn `INTERNAL_API_KEY`, generada amb el mateix procediment que `JWT_SECRET`.
    - `sc-api-gateway` és l'únic servei autoritzat a cridar els serveis interns directament. La resta de serveis es comuniquen exclusivament a través de Redis (cues) o MinIO (fitxers), sense cridades HTTP directes entre ells.
    - Exemple de capçalera per a comunicació interna:
        ```
        X-Internal-API-Key: <INTERNAL_API_KEY>
        ```

## 2. Esquema de Base de Dades (MongoDB)
 
El sistema utilitza 4 col·leccions. La instal·lació és per club, però el sistema suporta múltiples categories (Aleví A, Infantil B...). Els documents players i matches inclouen team_id per separar les dades per categoria.

### Col·lecció `players` — Plantilla de jugadors
 
Document persistent i reutilitzable entre partits. Es crea i manté manualment des del frontend.
 
```json
{
  "_id":        "ObjectId",
  "team_id":    "ObjectId",
  "dorsal":     9,
  "name":       "Pau Garcia",
  "position":   "base",
  "active":     true,
  "created_at": "ISODate"
}
```

- `team_id` és una FK lògica cap a `teams._id` de `sc-app-db`. Obligatori.
- `position` accepta els valors: `base`, `aler`, `pivot`.
- `active: false` desactiva el jugador sense eliminar-lo (historial preservat).
- **Índex:** compost `team_id + dorsal` (unique) — un dorsal és únic dins d'un equip, però dos equips poden tenir el mateix dorsal.

### Col·lecció `matches` — Partits
 
Document central de cada sessió de processament. Conté tota la metadata del vídeo i la configuració de la sessió.
 
```json
{
  "_id":          "ObjectId",
  "team_id":      "ObjectId",
  "title":        "Lliga J12 vs Joventut",
  "date":         "ISODate",
  "status":       "processing",
  "video_raw":    "a3f1c2d4/original.mp4",
  "video_output": "a3f1c2d4/output.mp4",
  "fps":          25,
  "start_frame":  1500,
  "end_frame":    138000,
  "roi_polygon":  [
    { "x": 120, "y": 80 },
    { "x": 1800, "y": 80 },
    { "x": 1800, "y": 1000 },
    { "x": 120, "y": 1000 }
  ],
  "created_at":   "ISODate",
  "updated_at":   "ISODate"
}
```

- `team_id` és una FK lògica cap a `teams._id`. Obligatori.
- `status` accepta els valors: `pending`, `processing`, `frames_ready`, `done`, `error`.
  - `pending` — el partit s'ha creat però el vídeo encara no s'ha processat.
  - `processing` — `sc-video-manager` ha rebut el job i està extraient frames.
  - `frames_ready` — tots els frames estan a MinIO i `sc-inference-worker` els està processant.
  - `done` — el pipeline ha finalitzat i el vídeo de sortida està disponible.
  - `error` — ha fallat algun pas del pipeline.
- `video_raw` i `video_output` contenen la clau MinIO (sense bucket), no una URL absoluta.
- `roi_polygon` és l'array de 4 punts `{x, y}` en píxels que defineix la zona activa de joc.
- `start_frame` i `end_frame` marquen el rang de frames a processar (descarten escalfament i descans).
- **Índexos:** `team_id`, `team_id + status`, `team_id + date` (desc), `status`, `date`.

### Col·lecció `match_players` — Jugadors per partit
 
Intersecció entre un partit i un jugador. Aquí viuen els minuts jugats, la confiança de detecció i l'historial de presència en pista. Es crea un document per cada jugador actiu quan comença el processament d'un partit.
 
```json
{
  "_id":            "ObjectId",
  "match_id":       "ObjectId",
  "player_id":      "ObjectId",
  "seconds_played": 1842,
  "confidence_avg": 0.87,
  "track_ids":      [3, 7, 15],
  "intervals": [
    { "in": 1500,  "out": 45000, "src": "auto" },
    { "in": 47200, "out": 98000, "src": "auto" },
    { "in": 98500, "out": 99000, "src": "manual" }
  ],
  "status":           "IN_GAME",
  "last_seen_frame":  99000,
  "updated_at":       "ISODate"
}
```
 
- `seconds_played` és el valor acumulat final. Es calcula sumant la durada de tots els `intervals`.
- `track_ids` conté tots els IDs de ByteTrack que el sistema ha associat a aquest jugador al llarg del partit (un jugador pot tenir múltiples track_ids per re-identificacions).
- `intervals` és l'historial complet de períodes en pista: `in` i `out` són números de frame, `src` indica si l'interval l'ha generat el sistema automàticament (`auto`) o l'ha corregit un usuari (`manual`).
- `status` reflecteix l'estat en temps real durant el processament: `IN_GAME` o `OFF_COURT`.
- **Índexos:** `match_id`, compost `match_id + player_id` (unique).

### Col·lecció `events` — Esdeveniments del partit
 
Registre immutable de tots els esdeveniments discrets: entrades i sortides de pista, substitucions i frames marcats per al feedback del model. És el log d'auditoria del partit.
 
```json
{
  "_id":          "ObjectId",
  "match_id":     "ObjectId",
  "player_id":    "ObjectId",
  "type":         "enter",
  "frame_number": 1500,
  "timestamp_s":  60.0,
  "confidence":   0.91,
  "source":       "auto",
  "created_at":   "ISODate"
}
```
 
- `type` accepta els valors: `enter`, `exit`, `substitution`, `feedback_flagged`, `low_confidence`.
- `player_id` és nullable: els esdeveniments de tipus `low_confidence` o `feedback_flagged` poden no tenir jugador associat si el dorsal no s'ha pogut identificar.
- `timestamp_s` és el segon de vídeo relatiu a `start_frame` (no al frame absolut).
- `source` indica si l'esdeveniment l'ha generat el sistema (`auto`) o l'usuari (`manual`).
- **Índexos:** `match_id + type`, `match_id + frame_number`.

### Resum de relacions
 
| Relació | Tipus | Nota |
| :--- | :--- | :--- |
| `players` → `match_players` | 1 : N | Un jugador pot aparèixer a molts partits |
| `matches` → `match_players` | 1 : N | Un partit té un document per cada jugador actiu |
| `matches` → `events` | 1 : N | Un partit genera múltiples esdeveniments |
| `players` → `events` | 0..1 : N | Un esdeveniment pot no tenir jugador identificat |

## 3. Gestió d'Usuaris — Arquitectura de Doble Base de Dades
 
El sistema separa **identitat** i **negoci** en dues bases de dades MongoDB lògicament independents dins del mateix contenidor `sc-mongodb`. Si `sc-app-db` es veiés compromesa, un atacant obtindria dades esportives però cap credencial. Si `sc-auth-db` es veiés compromesa, no tindria accés a cap dada de partits ni jugadors.

| Base de dades | Contingut | Qui hi accedeix |
| :--- | :--- | :--- |
| `sc-auth-db` | Credencials, rols, sessions (refresh tokens) | Únicament `sc-api-gateway` en login/refresh/logout |
| `sc-app-db` | Equips, perfils, partits, jugadors, estadístiques | `sc-api-gateway` i serveis interns (via Redis/MinIO) |

L'API Gateway utilitza dos clients Motor independents: `auth_db` i `app_db`. Cap servei té accés creuat entre les dues bases de dades.

### `sc-auth-db` — Col·lecció `users`
 
```json
{
  "_id":           "ObjectId",
  "email":         "pau.garcia@clubbadalona.cat",
  "password_hash": "$2b$12$...",
  "role":          "coach",
  "active":        true,
  "created_at":    "ISODate",
  "last_login":    "ISODate"
}
```
 
- `role` accepta: `admin`, `coach`, `assistant`, `player`.
- `password_hash` utilitza bcrypt amb cost factor 12 mínim.
- Cap camp d'aquest document fa referència a `sc-app-db`.
- **Índex:** `email` (unique).

### `sc-auth-db` — Col·lecció `refresh_tokens`
 
```json
{
  "_id":        "ObjectId",
  "user_id":    "ObjectId",
  "token_hash": "sha256:abc123...",
  "expires_at": "ISODate"
}
```
 
- `token_hash` emmagatzema el hash SHA-256 del token opac, mai el token en clar.
- `expires_at` té un **TTL index** de MongoDB que elimina automàticament els tokens expirats.
- Amb la Refresh Token Rotation (vegeu punt 2.10), cada crida a `/auth/refresh` elimina el document anterior i en crea un de nou.

### `sc-app-db` — Col·lecció `teams` (categories)
 
Representa cada categoria del club (Aleví A, Infantil B, etc.).
 
```json
{
  "_id":       "ObjectId",
  "name":      "Aleví A",
  "category":  "aleví",
  "coach_id":  "ObjectId",
  "active":    true
}
```
 
- `coach_id` és una **FK lògica** cap a `users._id` de `sc-auth-db`. MongoDB no enforça aquesta referència; la consistència la gestiona l'API Gateway.
- Un entrenador (`coach_id`) pot aparèixer a múltiples documents `teams`.
- Una categoria té exactament un entrenador.
- **Índexos:** `coach_id`, `category`.

### `sc-app-db` — Col·lecció `user_profiles`
 
Estén cada usuari amb les dades de negoci necessàries per al control d'accés i la navegació.
 
```json
{
  "_id":          "ObjectId",
  "user_id":      "ObjectId",
  "display_name": "Pau Garcia",
  "team_ids":     ["ObjectId", "ObjectId"],
  "player_id":    null
}
```
 
- `user_id` és la FK lògica que connecta amb `users._id` de `sc-auth-db`.
- `team_ids` conté els equips als quals l'usuari té accés:
  - **admin:** array amb tots els `team_ids` del club (o `[]` per indicar accés total).
  - **coach:** els equips que entrena.
  - **assistant:** els equips als quals ha estat assignat.
  - **player:** l'equip al qual pertany.
- `player_id` és nullable i només s'omple per al rol `player`, apuntant al document `players` de `sc-app-db`.
- **Índex:** `user_id` (unique).

### Flux d'autenticació i autorització
 
1. **Login:** L'API consulta `sc-auth-db` per verificar email i password (bcrypt). Si és correcte, carrega el `user_profile` de `sc-app-db` per obtenir `team_ids` i `player_id`.
2. **Emissió del JWT:** El payload del token inclou `user_id`, `role` i `team_ids`. D'aquesta manera, cada petició posterior és autocontinguda.
3. **Peticions normals:** L'API valida el JWT localment (sense consultar cap base de dades) i filtra les queries de `sc-app-db` pels `team_ids` extrets del token.
4. **`sc-auth-db` en repòs:** Un cop emès el JWT, `sc-auth-db` no es torna a consultar fins al proper login, refresh o logout.

### Regles d'accés per rol
 
| Rol | Veu partits de | Pot crear partits | Pot editar jugadors |
| :--- | :--- | :--- | :--- |
| `admin` | Tots els equips | Sí | Sí |
| `coach` | Els seus equips | Sí | Sí (els seus equips) |
| `assistant` | Els seus equips | No | No |
| `player` | El seu equip | No | No |

Amb `team_id` a `players` i `matches`, les queries de l'API ara filtren per `team_id` en lloc de confiar únicament en els `team_ids` del JWT:
 
- **Coach/assistant/player:** `{ team_id: { $in: token.team_ids } }` — veu només els jugadors i partits dels seus equips.
- **Admin:** sense filtre de `team_id` — veu tot.
 
Aquesta és la regla que `sc-api-gateway` ha d'aplicar a tots els endpoints de `players` i `matches`.