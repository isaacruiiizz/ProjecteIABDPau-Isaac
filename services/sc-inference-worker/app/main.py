"""
Entrypoint del sc-inference-worker.

Inicia dos threads:
  - labeling_worker: pre-anotació per al pipeline d'etiquetatge
  - inference_worker: inferència en producció (placeholder, no implementat encara)

Gestió de senyals SIGTERM/SIGINT per a shutdown net.
"""
import logging
import signal
import threading

import sentry_sdk

from app.config import settings
from app.workers import inference_worker, labeling_worker


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":%(message)s}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    if settings.SENTRY_DSN:
        sentry_sdk.init(dsn=settings.SENTRY_DSN, traces_sample_rate=0.1)

    logger.info('{"event":"sc_inference_worker_starting"}')

    stop_event = threading.Event()

    def _shutdown(signum, frame):  # noqa: ARG001
        logger.info('{"event":"shutdown_signal","signum":%d}', signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Thread de pre-anotació d'etiquetatge
    labeling_thread = threading.Thread(
        target=labeling_worker.run,
        args=(stop_event,),
        name="labeling-worker",
        daemon=True,
    )
    labeling_thread.start()
    logger.info('{"event":"labeling_worker_thread_started"}')

    # Thread de inferència en producció
    inference_thread = threading.Thread(
        target=inference_worker.run,
        args=(stop_event,),
        name="inference-worker",
        daemon=True,
    )
    inference_thread.start()
    logger.info('{"event":"inference_worker_thread_started"}')

    # Espera fins que arribi el senyal de parada
    stop_event.wait()

    labeling_thread.join(timeout=10)
    logger.info('{"event":"sc_inference_worker_stopped"}')


if __name__ == "__main__":
    main()
