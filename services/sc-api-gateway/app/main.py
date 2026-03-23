import logging
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from sentry_sdk.integrations.logging import LoggingIntegration

import app.dependencies as deps
from app.config import settings
from app.routers import auth, health


def setup_logging(service_name: str, sentry_dsn: str = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='{"time": "%(asctime)s", "service": "' + service_name
               + '", "level": "%(levelname)s", "message": "%(message)s"}',
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


# Primer de tot: logging + Sentry (captura errors d'arrencada)
setup_logging("sc-api-gateway", settings.SENTRY_DSN or None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    deps.auth_client = AsyncIOMotorClient(settings.MONGO_AUTH_URI)
    deps.app_client = AsyncIOMotorClient(settings.MONGO_APP_URI)
    yield
    deps.auth_client.close()
    deps.app_client.close()


app = FastAPI(title="SmartChrono IP — API Gateway", lifespan=lifespan)

# CORS només quan API_ENV=development (spec 03-infraestructura.md)
if settings.API_ENV == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(health.router)
app.include_router(auth.router)
