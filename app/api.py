from __future__ import annotations

from threading import Lock, Thread
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.burn1_contract import (
    Burn1PortfolioConfig,
    run_burn1_nfl_portfolio_safe,
)
from run_burn1_live import SITE_CONFIG, load_players


app = FastAPI(
    title="BURN1 DFS API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Burn1RunRequest(BaseModel):
    site: str
    lineup_count: int = Field(default=20, ge=1, le=150)
    candidate_count: int = Field(default=60, ge=1)
    min_unique_players: int = Field(default=2, ge=1)
    candidate_gpp_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )


jobs = {}
jobs_lock = Lock()


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "application": "BURN1 DFS",
        "engine": "NFL V1",
        "status": "ready",
    }


def update_job(job_id: str, **values):
    with jobs_lock:
        jobs[job_id].update(values)


def run_job(
    job_id: str,
    request: Burn1RunRequest,
):
    try:
        site_key = request.site.lower()

        if site_key not in SITE_CONFIG:
            raise ValueError(
                "site must be 'dk' or 'fd'."
            )

        site = SITE_CONFIG[site_key]

        players = load_players(
            site["pool_file"],
            site["site_name"],
        )

        def status_callback(event):
            update_job(
                job_id,
                status=event.status,
                message=event.message,
            )

        config = Burn1PortfolioConfig(
            site=site["site_name"],
            lineup_count=request.lineup_count,
            candidate_count=request.candidate_count,
            min_unique_players=request.min_unique_players,
            candidate_gpp_fraction=(
                request.candidate_gpp_fraction
            ),
        )

        response = run_burn1_nfl_portfolio_safe(
            players,
            config,
            status_callback=status_callback,
        )

        payload = response.to_dict()

        if response.ok:
            update_job(
                job_id,
                status="complete",
                message=(
                    "BURN1 portfolio generation "
                    "completed successfully."
                ),
                result=payload["result"],
                error=None,
            )
        else:
            update_job(
                job_id,
                status="error",
                message=payload["error"]["message"],
                result=None,
                error=payload["error"],
            )

    except Exception as exc:
        update_job(
            job_id,
            status="error",
            message="BURN1 run failed.",
            result=None,
            error={
                "code": "API_RUN_ERROR",
                "stage": "api",
                "message": str(exc),
                "details": "",
            },
        )


@app.post("/api/runs")
def start_run(request: Burn1RunRequest):
    if request.candidate_count < request.lineup_count:
        raise HTTPException(
            status_code=400,
            detail=(
                "candidate_count must be at least "
                "lineup_count."
            ),
        )

    job_id = str(uuid4())

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "BURN1 run queued.",
            "result": None,
            "error": None,
        }

    thread = Thread(
        target=run_job,
        args=(job_id, request),
        daemon=True,
    )
    thread.start()

    return jobs[job_id]


@app.get("/api/runs/{job_id}")
def get_run(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="BURN1 run not found.",
            )

        return dict(job)
