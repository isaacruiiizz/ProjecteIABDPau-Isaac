#!/usr/bin/env sh
set -e

ALIAS="minio"
ENDPOINT="http://sc-object-storage:9000"

# 1. Connectar amb credencials root
mc alias set "$ALIAS" "$ENDPOINT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"

# 2. Crear els 9 buckets
for bucket in raw-videos pending-frames processed-frames processed-videos \
              feedback-data models labeling-videos labeling-frames datasets; do
  mc mb --ignore-existing "$ALIAS/$bucket"
done

# 3. Lifecycle — buckets amb retenció de 30 dies
for bucket in raw-videos labeling-videos labeling-frames; do
  mc ilm add --expiry-days 30 "$ALIAS/$bucket" || true
done

# 4. Lifecycle — buckets amb retenció de 7 dies
for bucket in pending-frames processed-frames; do
  mc ilm add --expiry-days 7 "$ALIAS/$bucket" || true
done

# 5. Crear/actualitzar usuaris IAM per servei (add crea, passwd actualitza si ja existeix)
create_or_update_user() {
  USER=$1
  PASS=$2
  mc admin user add "$ALIAS" "$USER" "$PASS" 2>/dev/null || \
  mc admin user passwd "$ALIAS" "$USER" "$PASS"
}
create_or_update_user sc-api-gateway      "$SC_API_GATEWAY_MINIO_PASSWORD"
create_or_update_user sc-video-manager    "$SC_VIDEO_MANAGER_MINIO_PASSWORD"
create_or_update_user sc-inference-worker "$SC_INFERENCE_WORKER_MINIO_PASSWORD"
create_or_update_user sc-active-learner   "$SC_ACTIVE_LEARNER_MINIO_PASSWORD"
create_or_update_user sc-label-studio     "$SC_LABEL_STUDIO_MINIO_PASSWORD"

# 6. Carregar i assignar polítiques (elimina primer per garantir que s'aplica la versió actual)
for service in sc-api-gateway sc-video-manager sc-inference-worker sc-active-learner sc-label-studio; do
  mc admin policy rm "$ALIAS" "policy-$service" 2>/dev/null || true
  mc admin policy create "$ALIAS" "policy-$service" "/init/policies/$service.json"
  mc admin policy attach "$ALIAS" "policy-$service" --user "$service"
done

# 7. Política pública de lectura a labeling-frames perquè Label Studio
#    pugui servir les imatges directament al navegador sense presign tokens.
mc anonymous set download "$ALIAS/labeling-frames"

echo "MinIO init completat: 9 buckets + lifecycle + 5 usuaris IAM"
