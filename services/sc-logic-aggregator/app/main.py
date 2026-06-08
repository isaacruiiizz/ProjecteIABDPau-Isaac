import json
import logging
import sys

import redis
import sentry_sdk
from minio import Minio
from pymongo import MongoClient
from sentry_sdk.integrations.logging import LoggingIntegration

from app.config import settings
from app.services.aggregator_service import handle_frame_result

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
        sentry_logging = LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)
        sentry_sdk.init(dsn=sentry_dsn, integrations=[sentry_logging], traces_sample_rate=1.0)


def main() -> None:
    setup_logging("sc-logic-aggregator", settings.SENTRY_DSN)
    logger.info("sc-logic-aggregator starting")

    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD or None,
        decode_responses=True,
    )
    mongo_client = MongoClient(settings.MONGO_APP_URI)
    db = mongo_client[settings.MONGO_APP_URI.rsplit("/", 1)[-1].split("?")[0]]

    minio_client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_USE_SSL,
        region="us-east-1",
    )

    queue = settings.REDIS_QUEUE_RESULTS
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
            logger.error('{"event":"parse_error","error":"%s"}', str(exc))
            continue

        try:
            handle_frame_result(payload, redis_client, minio_client, db, settings)
        except Exception:
            logger.exception('{"event":"handle_error","payload_keys":"%s"}',
                             ",".join(payload.keys()))


if __name__ == "__main__":
    main()
