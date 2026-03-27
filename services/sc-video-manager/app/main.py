import json
import logging
import sys

import redis
import sentry_sdk
from minio import Minio
from sentry_sdk.integrations.logging import LoggingIntegration

from app.config import settings
from app.services.labeling_service import process_labeling

logger = logging.getLogger(__name__)


def setup_logging(service_name: str, sentry_dsn: str = "") -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format=(
            '{"time": "%(asctime)s", "service": "' + service_name + '", '
            '"level": "%(levelname)s", "message": "%(message)s"}'
        ),
    )
    if sentry_dsn:
        sentry_logging = LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR,
        )
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[sentry_logging],
            traces_sample_rate=1.0,
        )


def main() -> None:
    setup_logging("sc-video-manager", settings.SENTRY_DSN)
    logger.info("sc-video-manager starting")

    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD or None,
        decode_responses=True,
    )
    minio_client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_USE_SSL,
        region="us-east-1",
    )

    queue = settings.REDIS_QUEUE_VIDEO
    logger.info('{"event":"listening","queue":"%s"}', queue)

    while True:
        try:
            _, raw = redis_client.blpop(queue, timeout=0)
        except Exception as exc:
            logger.error('{"event":"redis_error","error":"%s"}', str(exc))
            continue

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error('{"event":"parse_error","raw":"%s","error":"%s"}', raw[:200], str(exc))
            continue

        job_type = payload.get("job_type")

        if job_type == "process_labeling":
            try:
                process_labeling(
                    payload,
                    minio_client,
                    settings.MINIO_BUCKET_LABELING_FRAMES,
                    redis_client=redis_client,
                    redis_queue_labeling="labeling_frames_to_infer",
                    label_studio_url=settings.LABEL_STUDIO_URL,
                    label_studio_api_token=settings.LABEL_STUDIO_API_TOKEN,
                    label_studio_source_storage_id=settings.LABEL_STUDIO_SOURCE_STORAGE_ID,
                    minio_public_url=settings.MINIO_PUBLIC_URL,
                )
            except Exception:
                # L'error ja ha estat logat i enviat a Sentry dins process_labeling
                pass

        elif job_type == "process_match":
            # Implementació pendent (ticket independent)
            logger.warning('{"event":"job_type_not_implemented","job_type":"process_match"}')

        else:
            logger.warning('{"event":"unknown_job_type","job_type":"%s"}', job_type)


if __name__ == "__main__":
    main()
