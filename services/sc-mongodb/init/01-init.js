// SmartChrono IP — MongoDB init script
// S'executa automàticament en el primer arranc del contenidor.
// Crea índexos a sc-auth-db i sc-app-db, i fa seed de l'usuari admin.
// Spec: docs/specs/07-estructura.md

// ── sc-auth-db ────────────────────────────────────────────────────────────────
const authDb = db.getSiblingDB('sc-auth-db');

authDb.users.createIndex({ email: 1 }, { unique: true });
authDb.refresh_tokens.createIndex({ expires_at: 1 }, { expireAfterSeconds: 0 });
authDb.refresh_tokens.createIndex({ user_id: 1 });

// ── sc-app-db ─────────────────────────────────────────────────────────────────
const appDb = db.getSiblingDB('sc-app-db');

// players — índex compost team_id + dorsal (unique per equip)
appDb.players.createIndex({ team_id: 1, dorsal: 1 }, { unique: true });

// matches
appDb.matches.createIndex({ team_id: 1 });
appDb.matches.createIndex({ team_id: 1, status: 1 });
appDb.matches.createIndex({ team_id: 1, date: -1 });
appDb.matches.createIndex({ status: 1 });
appDb.matches.createIndex({ date: -1 });

// match_players
appDb.match_players.createIndex({ match_id: 1 });
appDb.match_players.createIndex({ match_id: 1, player_id: 1 }, { unique: true });

// events
appDb.events.createIndex({ match_id: 1, type: 1 });
appDb.events.createIndex({ match_id: 1, frame_number: 1 });

// teams
appDb.teams.createIndex({ coach_id: 1 });
appDb.teams.createIndex({ category: 1 });

// user_profiles
appDb.user_profiles.createIndex({ user_id: 1 }, { unique: true });

// ── Seed: primer usuari admin ─────────────────────────────────────────────────
// Les variables d'entorn s'injecten via docker-compose (env_file: sc-mongodb/.env)
// password_hash conté el password EN CLAR amb force_reset: true.
// En el primer POST /auth/login, sc-api-gateway fa el hash bcrypt i elimina el flag.
const adminEmail = process.env.ADMIN_EMAIL;
const adminPassword = process.env.ADMIN_PASSWORD;
const adminDisplayName = process.env.ADMIN_DISPLAY_NAME || 'Administrador';

const existingAdmin = authDb.users.findOne({ email: adminEmail });
if (!existingAdmin) {
  authDb.users.insertOne({
    email: adminEmail,
    password_hash: adminPassword,
    force_reset: true,
    role: 'admin',
    active: true,
    created_at: new Date(),
    last_login: null,
  });

  const adminUser = authDb.users.findOne({ email: adminEmail });

  appDb.user_profiles.insertOne({
    user_id: adminUser._id,
    display_name: adminDisplayName,
    team_ids: [],
    player_id: null,
  });

  print('Seed: usuari admin creat → ' + adminEmail);
} else {
  print('Seed: usuari admin ja existeix, omès.');
}
