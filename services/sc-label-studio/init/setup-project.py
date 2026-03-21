#!/usr/bin/env python3
"""
setup-project.py — Inicialització del projecte Label Studio per a SmartChrono IP.

Executat per sc-label-studio-init un cop Label Studio és healthy.
Idempotent: si el projecte ja existeix, no el recrea.

Variables d'entorn llegides:
  LABEL_STUDIO_URL        — ex: http://sc-label-studio:8081
  LABEL_STUDIO_USERNAME   — email de l'admin
  LABEL_STUDIO_PASSWORD   — password de l'admin
  MINIO_ENDPOINT          — ex: http://sc-object-storage:9000
  MINIO_ACCESS_KEY
  MINIO_SECRET_KEY
  MINIO_BUCKET_FRAMES     — labeling-frames (source)
  MINIO_BUCKET_DATASETS   — datasets (export)
"""

import os
import sys
import time
import requests

# ── Variables d'entorn ────────────────────────────────────────────────────────

LS_URL      = os.environ["LABEL_STUDIO_URL"].rstrip("/")
LS_USER     = os.environ["LABEL_STUDIO_USERNAME"]
LS_PASS     = os.environ["LABEL_STUDIO_PASSWORD"]
MINIO_URL   = os.environ["MINIO_ENDPOINT"].rstrip("/")
MINIO_KEY   = os.environ["MINIO_ACCESS_KEY"]
MINIO_SEC   = os.environ["MINIO_SECRET_KEY"]
BUCKET_SRC  = os.environ["MINIO_BUCKET_FRAMES"]
BUCKET_DST  = os.environ["MINIO_BUCKET_DATASETS"]

PROJECT_NAME = "SmartChrono — Etiquetatge de Jugadors"

LABEL_CONFIG = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="player_own" background="#ffb700"/>
    <Label value="other" background="#0074D9"/>
  </RectangleLabels>
</View>"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def wait_for_label_studio(max_retries: int = 30, delay: int = 10) -> None:
    """Espera fins que Label Studio respon a /health."""
    url = f"{LS_URL}/health"
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"Label Studio disponible ({url})")
                return
        except requests.exceptions.ConnectionError:
            pass
        print(f"Esperant Label Studio... intent {attempt}/{max_retries}")
        time.sleep(delay)
    print("ERROR: Label Studio no ha arrancat a temps", file=sys.stderr)
    sys.exit(1)


def get_token(session: requests.Session) -> str:
    """Autentica i retorna el token d'API."""
    r = session.post(
        f"{LS_URL}/api/auth/login/",
        json={"email": LS_USER, "password": LS_PASS},
    )
    r.raise_for_status()
    token = r.json().get("token")
    if not token:
        print("ERROR: no s'ha pogut obtenir el token d'API", file=sys.stderr)
        sys.exit(1)
    return token


def find_project(session: requests.Session) -> int | None:
    """Retorna l'id del projecte si ja existeix, o None."""
    r = session.get(f"{LS_URL}/api/projects/")
    r.raise_for_status()
    for project in r.json().get("results", []):
        if project["title"] == PROJECT_NAME:
            return project["id"]
    return None


def create_project(session: requests.Session) -> int:
    """Crea el projecte i retorna el seu id."""
    r = session.post(
        f"{LS_URL}/api/projects/",
        json={"title": PROJECT_NAME, "label_config": LABEL_CONFIG},
    )
    r.raise_for_status()
    project_id = r.json()["id"]
    print(f"Projecte creat: id={project_id}")
    return project_id


def configure_source_storage(session: requests.Session, project_id: int) -> None:
    """Configura el Source Storage S3 (labeling-frames → tasques d'etiquetatge)."""
    payload = {
        "project": project_id,
        "title": "MinIO — labeling-frames",
        "type": "s3",
        "bucket": BUCKET_SRC,
        "prefix": "",
        "aws_access_key_id": MINIO_KEY,
        "aws_secret_access_key": MINIO_SEC,
        "endpoint_url": MINIO_URL,
        "region_name": "us-east-1",
        "use_blob_urls": True,
        "recursive_scan": True,
    }
    r = session.post(f"{LS_URL}/api/storages/s3/", json=payload)
    r.raise_for_status()
    storage_id = r.json()["id"]
    print(f"Source Storage configurat: id={storage_id} bucket={BUCKET_SRC}")

    # Sincronitza per importar els frames existents
    r2 = session.post(f"{LS_URL}/api/storages/s3/{storage_id}/sync")
    r2.raise_for_status()
    print("Sincronització de Source Storage iniciada")


def configure_export_storage(session: requests.Session, project_id: int) -> None:
    """Configura l'Export Storage S3 (datasets → exportació d'anotacions)."""
    payload = {
        "project": project_id,
        "title": "MinIO — datasets",
        "type": "s3",
        "bucket": BUCKET_DST,
        "prefix": "yolo/v1/",
        "aws_access_key_id": MINIO_KEY,
        "aws_secret_access_key": MINIO_SEC,
        "endpoint_url": MINIO_URL,
        "region_name": "us-east-1",
        "can_delete_objects": False,
    }
    r = session.post(f"{LS_URL}/api/storages/s3export/", json=payload)
    r.raise_for_status()
    print(f"Export Storage configurat: bucket={BUCKET_DST} prefix=yolo/v1/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    wait_for_label_studio()

    session = requests.Session()
    token = get_token(session)
    session.headers.update({"Authorization": f"Token {token}"})

    project_id = find_project(session)
    if project_id is not None:
        print(f"Projecte ja existeix (id={project_id}), res a fer.")
        return

    project_id = create_project(session)
    configure_source_storage(session, project_id)
    configure_export_storage(session, project_id)
    print("Inicialització completada correctament.")


if __name__ == "__main__":
    main()
