# ── Etapa 1: Build ─────────────────────────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

# Les variables VITE_* s'han de passar en temps de build perquè Vite les inclou
# directament al bundle JS estàtic (no es llegeix en runtime).
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

ARG VITE_LABEL_STUDIO_URL
ENV VITE_LABEL_STUDIO_URL=$VITE_LABEL_STUDIO_URL

COPY package*.json .
RUN npm install

COPY . .
RUN npm run build

# ── Etapa 2: Runtime ───────────────────────────────────────────────────────
FROM node:22-alpine

WORKDIR /app

RUN npm install -g serve

COPY --from=builder /app/dist ./dist

EXPOSE 3000

CMD ["serve", "-s", "dist", "-l", "3000"]
