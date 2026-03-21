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
  mc ilm add --expiry-days 30 "$ALIAS/$bucket"
done

# 4. Lifecycle — buckets amb retenció de 7 dies
for bucket in pending-frames processed-frames; do
  mc ilm add --expiry-days 7 "$ALIAS/$bucket"
done

# 5. Crear usuaris IAM per servei (|| true: idempotent si ja existeixen)
mc admin user add "$ALIAS" sc-api-gateway      "$SC_API_GATEWAY_MINIO_PASSWORD"      || true
mc admin user add "$ALIAS" sc-video-manager    "$SC_VIDEO_MANAGER_MINIO_PASSWORD"    || true
mc admin user add "$ALIAS" sc-inference-worker "$SC_INFERENCE_WORKER_MINIO_PASSWORD" || true
mc admin user add "$ALIAS" sc-active-learner   "$SC_ACTIVE_LEARNER_MINIO_PASSWORD"   || true
mc admin user add "$ALIAS" sc-label-studio     "$SC_LABEL_STUDIO_MINIO_PASSWORD"     || true

# 6. Carregar i assignar polítiques
for service in sc-api-gateway sc-video-manager sc-inference-worker sc-active-learner sc-label-studio; do
  mc admin policy create "$ALIAS" "policy-$service" "/init/policies/$service.json" || true
  mc admin policy attach "$ALIAS" "policy-$service" --user "$service"
done

echo "MinIO init completat: 9 buckets + lifecycle + 5 usuaris IAM"
