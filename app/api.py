from __future__ import annotations

from threading import Lock, Thread
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.burn1_contract import (
    Burn1PortfolioConfig,
    run_burn1_nfl_portfolio_safe,
)
from run_burn1_live import SITE_CONFIG, load_players


from app.intelligence.foundation import (
    create_run_archive,
    readiness_report,
    update_run_archive,
)

app = FastAPI(
    title="BURN1 DFS API",
    version="1.1.0",
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


SITE_CAPS = {
    "dk": 50000,
    "fd": 60000,
}


class Burn1RunRequest(BaseModel):
    site: str
    lineup_count: int = Field(default=20, ge=1, le=150)
    candidate_count: int = Field(default=60, ge=1, le=500)
    min_unique_players: int = Field(default=2, ge=1, le=9)

    candidate_gpp_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    candidate_max_player_exposure: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
    )

    qb_stack_min: int = Field(default=1, ge=0, le=4)
    bring_back_min: int = Field(default=0, ge=0, le=4)
    rb_dst_stack: bool = False

    locked_player_ids: list[str] = Field(
        default_factory=list
    )

    excluded_player_ids: list[str] = Field(
        default_factory=list
    )

    min_player_exposures: dict[str, float] = Field(
        default_factory=dict
    )

    max_player_exposures: dict[str, float] = Field(
        default_factory=dict
    )

    min_strategy_exposures: dict[str, float] = Field(
        default_factory=dict
    )

    max_strategy_exposures: dict[str, float] = Field(
        default_factory=dict
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
        "api_version": "1.1.0",
    }


def _site_key(value: str) -> str:
    site_key = value.lower().strip()

    if site_key not in SITE_CONFIG:
        raise HTTPException(
            status_code=404,
            detail="site must be 'dk' or 'fd'.",
        )

    return site_key


@app.get("/api/player-pool/{site_key}")
def get_player_pool(site_key: str):
    site_key = _site_key(site_key)
    site = SITE_CONFIG[site_key]

    try:
        players = load_players(
            site["pool_file"],
            site["site_name"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not load player pool: {exc}"
            ),
        ) from exc

    rows = [
        {
            "player_id": player.player_id,
            "name": player.name,
            "position": player.position,
            "team": player.team,
            "opponent": player.opponent,
            "salary": player.salary,
            "status": player.status,
            "projection": round(
                float(player.projection),
                4,
            ),
            "roster_positions": list(
                player.roster_positions
            ),
        }
        for player in players
    ]

    rows.sort(
        key=lambda row: (
            row["position"],
            -row["projection"],
            row["name"],
        )
    )

    return {
        "site": site["site_name"],
        "salary_cap": SITE_CAPS[site_key],
        "player_count": len(rows),
        "players": rows,
    }



@app.get("/api/intelligence/readiness")
def intelligence_readiness():
    return readiness_report(
        SITE_CONFIG
    )

def update_job(job_id: str, **values):
    with jobs_lock:
        jobs[job_id].update(values)

    try:
        update_run_archive(
            job_id,
            status=values.get("status"),
            message=values.get("message"),
            result=values.get("result"),
            error=values.get("error"),
        )
    except Exception as exc:
        print(
            "[BURN1 Intelligence] "
            "Run archive update warning: "
            f"{exc}",
            flush=True,
        )


def run_job(
    job_id: str,
    request: Burn1RunRequest,
):
    started = perf_counter()

    previous_status = None
    previous_status_started = started

    timings = {}

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
            nonlocal previous_status
            nonlocal previous_status_started

            now = perf_counter()

            if previous_status is not None:
                timings[
                    f"{previous_status}_seconds"
                ] = round(
                    now - previous_status_started,
                    3,
                )

            previous_status = event.status
            previous_status_started = now

            progress_current = getattr(
                event,
                "progress_current",
                None,
            )

            progress_total = getattr(
                event,
                "progress_total",
                None,
            )

            update_job(
                job_id,
                status=event.status,
                message=event.message,
                progress_current=progress_current,
                progress_total=progress_total,
                elapsed_seconds=round(
                    now - started,
                    3,
                ),
                timings=dict(timings),
            )

        config = Burn1PortfolioConfig(
            site=site["site_name"],
            lineup_count=request.lineup_count,
            candidate_count=request.candidate_count,
            min_unique_players=(
                request.min_unique_players
            ),
            candidate_gpp_fraction=(
                request.candidate_gpp_fraction
            ),
            candidate_max_player_exposure=(
                request.candidate_max_player_exposure
            ),
            qb_stack_min=request.qb_stack_min,
            bring_back_min=request.bring_back_min,
            rb_dst_stack=request.rb_dst_stack,
            locked_player_ids=set(
                request.locked_player_ids
            ),
            excluded_player_ids=set(
                request.excluded_player_ids
            ),
            min_player_exposures=dict(
                request.min_player_exposures
            ),
            max_player_exposures=dict(
                request.max_player_exposures
            ),
            min_strategy_exposures=dict(
                request.min_strategy_exposures
            ),
            max_strategy_exposures=dict(
                request.max_strategy_exposures
            ),
        )

        response = run_burn1_nfl_portfolio_safe(
            players,
            config,
            status_callback=status_callback,
        )

        payload = response.to_dict()

        total_seconds = round(
            perf_counter() - started,
            3,
        )

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
                total_seconds=total_seconds,
                elapsed_seconds=total_seconds,
                timings=dict(timings),
            )

        else:
            update_job(
                job_id,
                status="error",
                message=payload["error"]["message"],
                result=None,
                error=payload["error"],
                total_seconds=total_seconds,
                elapsed_seconds=total_seconds,
                timings=dict(timings),
            )

    except Exception as exc:
        total_seconds = round(
            perf_counter() - started,
            3,
        )

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
            total_seconds=total_seconds,
            elapsed_seconds=total_seconds,
            timings=dict(timings),
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

    conflicts = (
        set(request.locked_player_ids)
        & set(request.excluded_player_ids)
    )

    if conflicts:
        conflict = sorted(conflicts)[0]

        raise HTTPException(
            status_code=400,
            detail=(
                f"Player {conflict} cannot be both "
                "locked and excluded."
            ),
        )

    job_id = str(uuid4())

    try:
        site_key = request.site.lower()

        if site_key in SITE_CONFIG:
            site = SITE_CONFIG[site_key]

            request_data = (
                request.model_dump(
                    mode="json"
                )
                if hasattr(
                    request,
                    "model_dump",
                )
                else request.dict()
            )

            create_run_archive(
                job_id=job_id,
                request_data=request_data,
                site_key=site_key,
                site_name=site["site_name"],
                pool_file=site["pool_file"],
            )
    except Exception as exc:
        print(
            "[BURN1 Intelligence] "
            "Run archive start warning: "
            f"{exc}",
            flush=True,
        )

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "BURN1 run queued.",
            "progress_current": 0,
            "progress_total": request.candidate_count,
            "result": None,
            "error": None,
            "elapsed_seconds": 0.0,
            "total_seconds": None,
            "timings": {},
        }

    thread = Thread(
        target=run_job,
        args=(job_id, request),
        daemon=True,
    )

    thread.start()

    return dict(jobs[job_id])


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
