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
import boto3
from botocore.exceptions import ClientError

# ── Variables d'entorn ────────────────────────────────────────────────────────

LS_URL      = os.environ["LABEL_STUDIO_URL"].rstrip("/")
LS_USER     = os.environ["LABEL_STUDIO_USERNAME"]
LS_PASS     = os.environ["LABEL_STUDIO_PASSWORD"]
MINIO_URL   = os.environ["MINIO_ENDPOINT"].rstrip("/")
MINIO_KEY   = os.environ["MINIO_ACCESS_KEY"]
MINIO_SEC   = os.environ["MINIO_SECRET_KEY"]
BUCKET_SRC  = os.environ["MINIO_BUCKET_FRAMES"]
BUCKET_DST  = os.environ["MINIO_BUCKET_DATASETS"]
# URL accessible des del navegador per a les URLs pre-signades de MinIO.
# host.docker.internal:9000 funciona tant des de contenidors Docker com des del navegador
# (Docker Desktop el mapeja a 127.0.0.1 al fitxer hosts de Windows).
MINIO_PRESIGN_ENDPOINT = os.environ.get("MINIO_PRESIGN_ENDPOINT", "http://host.docker.internal:9000").rstrip("/")

PROJECT_NAME = "SmartChrono — Etiquetatge de Jugadors"

LABEL_CONFIG = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="player_own" background="#ffb700"/>
    <Label value="other" background="#0074D9"/>
  </RectangleLabels>
</View>"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def test_minio_connection() -> None:
    """Verifica la connexió a MinIO amb boto3 abans de cridar Label Studio."""
    print(f"Testant connexió MinIO: endpoint={MINIO_URL} user={MINIO_KEY} bucket={BUCKET_SRC}")
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_KEY,
            aws_secret_access_key=MINIO_SEC,
            region_name="us-east-1",
        )
        s3.list_objects_v2(Bucket=BUCKET_SRC, MaxKeys=1)
        print(f"MinIO OK: {BUCKET_SRC} accessible.")
    except ClientError as e:
        print(f"ERROR MinIO boto3: {e}", file=sys.stderr)
        sys.exit(1)


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


def web_login(session: requests.Session, max_retries: int = 20, delay: int = 10) -> None:
    """Login via web (CSRF) i obté el token d'API per a les crides autenticades."""
    for attempt in range(1, max_retries + 1):
        try:
            # 1. GET pàgina login per obtenir CSRF cookie
            session.get(f"{LS_URL}/user/login/", timeout=10)
            csrf = session.cookies.get("csrftoken")
            if not csrf:
                print(f"CSRF no disponible (intent {attempt}/{max_retries}), esperant...")
                time.sleep(delay)
                continue

            # 2. POST login
            r = session.post(
                f"{LS_URL}/user/login/",
                data={"email": LS_USER, "password": LS_PASS, "csrfmiddlewaretoken": csrf},
                headers={"Referer": f"{LS_URL}/user/login/"},
                allow_redirects=True,
                timeout=10,
            )
            # Login OK si redirigeix a / o retorna 200 sense tornar al login
            if "/user/login" not in r.url and r.status_code == 200:
                # Afegir CSRF a totes les peticions POST/PUT/DELETE
                session.headers.update({"X-CSRFToken": session.cookies.get("csrftoken", csrf)})
                print("Login web correcte, sessió activa.")

                # Obtenir token d'API i afegir-lo com a capçalera Authorization
                r_token = session.get(f"{LS_URL}/api/current-user/token", timeout=10)
                if r_token.ok:
                    api_token = r_token.json().get("token")
                    if api_token:
                        session.headers.update({"Authorization": f"Token {api_token}"})
                        print("Token d'API obtingut correctament.")
                    else:
                        print("AVÍS: resposta de token buida, continuant amb sessió.", file=sys.stderr)
                else:
                    print(f"AVÍS: no s'ha pogut obtenir el token d'API ({r_token.status_code}), continuant amb sessió.", file=sys.stderr)
                return
            print(f"Login fallat (url={r.url}, status={r.status_code}), reintentant... ({attempt}/{max_retries})")
        except requests.exceptions.RequestException as exc:
            print(f"Error: {exc}, reintentant... ({attempt}/{max_retries})")
        time.sleep(delay)

    print("ERROR: no s'ha pogut fer login a Label Studio", file=sys.stderr)
    sys.exit(1)


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
        "s3_endpoint": MINIO_PRESIGN_ENDPOINT,
        "region_name": "us-east-1",
        "use_blob_urls": True,
        "recursive_scan": True,
        "presign_ttl": 300,
    }
    r = session.post(f"{LS_URL}/api/storages/s3/", json=payload)
    if not r.ok:
        print(f"ERROR source storage: {r.status_code} {r.text}", file=sys.stderr)
    r.raise_for_status()
    storage_id = r.json()["id"]
    print(f"Source Storage configurat: id={storage_id} bucket={BUCKET_SRC}")

    # Sincronitza per importar els frames existents
    r2 = session.post(f"{LS_URL}/api/storages/s3/{storage_id}/sync")
    r2.raise_for_status()
    print("Sincronització de Source Storage iniciada")


def ensure_export_prefix() -> None:
    """Crea el prefix yolo/v1/ al bucket datasets si no existeix."""
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_KEY,
        aws_secret_access_key=MINIO_SEC,
        region_name="us-east-1",
    )
    try:
        s3.put_object(Bucket=BUCKET_DST, Key="yolo/v1/.keep", Body=b"")
        print(f"Prefix yolo/v1/ creat a {BUCKET_DST}")
    except ClientError as e:
        print(f"AVÍS: no s'ha pogut crear el prefix: {e}", file=sys.stderr)


def configure_export_storage(session: requests.Session, project_id: int) -> None:
    """Configura l'Export Storage S3 (datasets → exportació d'anotacions)."""
    ensure_export_prefix()

    endpoint = f"{LS_URL}/api/storages/export/s3"
    payload = {
        "project": project_id,
        "title": "MinIO — datasets",
        "bucket": BUCKET_DST,
        "prefix": "yolo/v1/",
        "aws_access_key_id": MINIO_KEY,
        "aws_secret_access_key": MINIO_SEC,
        "endpoint_url": MINIO_URL,
        "region_name": "us-east-1",
        "can_delete_objects": False,
    }
    r = session.post(endpoint, json=payload)
    if not r.ok:
        print(f"ERROR export storage: {r.status_code} {r.text}", file=sys.stderr)
    r.raise_for_status()
    print(f"Export Storage configurat: bucket={BUCKET_DST} prefix=yolo/v1/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    test_minio_connection()
    wait_for_label_studio()

    session = requests.Session()
    web_login(session)

    project_id = find_project(session)
    if project_id is None:
        project_id = create_project(session)

    # Comprova si ja hi ha storages configurats
    def storage_exists(url: str) -> bool:
        r = session.get(url)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        data = r.json()
        items = data if isinstance(data, list) else data.get("results", [])
        return len(items) > 0

    source_exists = storage_exists(f"{LS_URL}/api/storages/s3/?project={project_id}")
    export_exists = storage_exists(f"{LS_URL}/api/storages/export/s3?project={project_id}")

    if not source_exists:
        configure_source_storage(session, project_id)
    else:
        print("Source Storage ja configurat, saltant.")

    if not export_exists:
        configure_export_storage(session, project_id)
    else:
        print("Export Storage ja configurat, saltant.")

    print("Inicialització completada correctament.")


if __name__ == "__main__":
    main()
