"""FastAPI Main Application Entrypoint for SPECTRA."""
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.database.session import init_db
from app.services.sample_generator import ensure_reference_samples


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_reference_samples()
    yield


app = FastAPI(
    title="SPECTRA | AI Audio Forensics & Fingerprint Analysis Laboratory",
    description=(
        "Local-first scientific audio forensics, C2PA provenance verification, "
        "neural-vocoder spectral fakeprint analysis, and experimental DSP laboratory."
    ),
    version=settings.SPECTRA_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "name": "SPECTRA | AI Audio Forensics & Fingerprint Analysis Laboratory",
        "version": settings.SPECTRA_VERSION,
        "docs": "/docs",
        "health": "/api/health",
    }
