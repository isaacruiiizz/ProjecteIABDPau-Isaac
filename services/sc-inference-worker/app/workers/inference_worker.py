"""
Worker de inferència en producció — placeholder.
S'implementarà al ticket PJM-36 (pipeline de producció).

Escoltarà les cues: ['task_frames', 'model_promoted'] amb BLPOP.
"""
import logging

logger = logging.getLogger(__name__)


def run() -> None:
    logger.info('{"event":"inference_worker_placeholder","msg":"not yet implemented"}')
