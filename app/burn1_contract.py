from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from models.player import Player
from app.prelock_validation import (
    PreLockValidationError,
    validate_final_lineups,
    validate_player_pool,
)
from optimizer.portfolio_optimizer import (
    Stage2InfeasibleError,
    generate_nfl_candidates,
    select_nfl_portfolio,
)


SUPPORTED_SITES = {
    "DraftKings": 50000,
    "FanDuel": 60000,
}

SUPPORTED_STRATEGIES = {
    "stacked",
    "unstacked",
    "qb_stack_1_plus",
    "qb_stack_2_plus",
    "bring_back_1_plus",
    "rb_dst",
    "qb_vs_opposing_dst",
}

STATUS_VALIDATING = "validating"
STATUS_STAGE1 = "stage1_generating"
STATUS_STAGE2 = "stage2_optimizing"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"


@dataclass
class Burn1PortfolioConfig:
    site: str
    lineup_count: int = 20
    candidate_count: int = 200
    min_unique_players: int = 2
    candidate_gpp_fraction: float | None = None
    candidate_max_player_exposure: float | None = None

    gpp_mode: bool = False
    qb_stack_min: int = 1
    bring_back_min: int = 0
    rb_dst_stack: bool = False

    locked_player_ids: set[str] = field(default_factory=set)
    excluded_player_ids: set[str] = field(default_factory=set)
    min_player_exposures: dict[str, float] = field(default_factory=dict)
    max_player_exposures: dict[str, float] = field(default_factory=dict)
    min_strategy_exposures: dict[str, float] = field(default_factory=dict)
    max_strategy_exposures: dict[str, float] = field(default_factory=dict)


@dataclass
class Burn1StatusEvent:
    status: str
    message: str

    def to_dict(self):
        return asdict(self)


@dataclass
class Burn1RunResult:
    status: str
    site: str
    requested_lineups: int
    generated_lineups: int
    requested_candidates: int
    generated_candidates: int
    solver_status: str
    total_projection: float
    lineups: list[dict[str, Any]]
    exposures: list[dict[str, Any]]
    strategy_summary: dict[str, dict[str, Any]]
    message: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class Burn1ErrorResult:
    code: str
    stage: str
    message: str
    details: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class Burn1ApplicationResponse:
    ok: bool
    status: str
    result: Burn1RunResult | None = None
    error: Burn1ErrorResult | None = None

    def to_dict(self):
        return asdict(self)


class Burn1ApplicationError(ValueError):
    def __init__(
        self,
        code,
        stage,
        message,
        details="",
    ):
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.message = message
        self.details = details


def _emit_status(
    status_callback: Callable[[Burn1StatusEvent], None] | None,
    status,
    message,
):
    if status_callback is None:
        return

    status_callback(
        Burn1StatusEvent(
            status=status,
            message=message,
        )
    )


def _validate_fraction_map(values, label):
    for key, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{label} for {key} must be between 0.0 and 1.0."
            )


def _validate_config(config):
    if config.site not in SUPPORTED_SITES:
        raise ValueError(
            "site must be 'DraftKings' or 'FanDuel'."
        )

    if config.lineup_count < 1:
        raise ValueError(
            "lineup_count must be at least 1."
        )

    if config.candidate_count < config.lineup_count:
        raise ValueError(
            "candidate_count must be at least lineup_count."
        )

    if config.min_unique_players < 1:
        raise ValueError(
            "min_unique_players must be at least 1."
        )

    if (
        config.candidate_gpp_fraction is not None
        and not 0.0 <= config.candidate_gpp_fraction <= 1.0
    ):
        raise ValueError(
            "candidate_gpp_fraction must be between 0.0 and 1.0."
        )

    if (
        config.candidate_max_player_exposure is not None
        and not 0.0
        < config.candidate_max_player_exposure
        <= 1.0
    ):
        raise ValueError(
            "candidate_max_player_exposure must be "
            "greater than 0.0 and at most 1.0."
        )

    if config.qb_stack_min < 0:
        raise ValueError(
            "qb_stack_min cannot be negative."
        )

    if config.bring_back_min < 0:
        raise ValueError(
            "bring_back_min cannot be negative."
        )

    conflicts = (
        set(config.locked_player_ids)
        & set(config.excluded_player_ids)
    )

    if conflicts:
        conflict = sorted(conflicts)[0]
        raise ValueError(
            f"Player {conflict} cannot be both locked and excluded."
        )

    exposure_maps = (
        (config.min_player_exposures, "Minimum player exposure"),
        (config.max_player_exposures, "Maximum player exposure"),
        (config.min_strategy_exposures, "Minimum strategy exposure"),
        (config.max_strategy_exposures, "Maximum strategy exposure"),
    )

    for values, label in exposure_maps:
        _validate_fraction_map(values, label)

    player_ids = (
        set(config.min_player_exposures)
        | set(config.max_player_exposures)
    )

    for player_id in player_ids:
        minimum = config.min_player_exposures.get(
            player_id,
            0.0,
        )
        maximum = config.max_player_exposures.get(
            player_id,
            1.0,
        )

        if minimum > maximum:
            raise ValueError(
                f"Minimum player exposure for {player_id} "
                "cannot exceed maximum exposure."
            )

    strategy_names = (
        set(config.min_strategy_exposures)
        | set(config.max_strategy_exposures)
    )

    unknown_strategies = (
        strategy_names - SUPPORTED_STRATEGIES
    )

    if unknown_strategies:
        unknown = sorted(unknown_strategies)[0]
        raise ValueError(
            f"Unknown BURN1 strategy: {unknown}"
        )

    for strategy_name in strategy_names:
        minimum = config.min_strategy_exposures.get(
            strategy_name,
            0.0,
        )
        maximum = config.max_strategy_exposures.get(
            strategy_name,
            1.0,
        )

        if minimum > maximum:
            raise ValueError(
                "Minimum strategy exposure for "
                f"{strategy_name} cannot exceed maximum exposure."
            )


def _strategy_matches(profile, strategy_name):
    if strategy_name == "stacked":
        return not profile.is_unstacked
    if strategy_name == "unstacked":
        return profile.is_unstacked
    if strategy_name == "qb_stack_1_plus":
        return profile.qb_stack_size >= 1
    if strategy_name == "qb_stack_2_plus":
        return profile.qb_stack_size >= 2
    if strategy_name == "bring_back_1_plus":
        return profile.bring_back_size >= 1
    if strategy_name == "rb_dst":
        return profile.has_rb_dst
    if strategy_name == "qb_vs_opposing_dst":
        return profile.has_qb_vs_opposing_dst

    raise ValueError(
        f"Unknown BURN1 strategy: {strategy_name}"
    )


def _serialize_lineup(lineup, profile):
    players = []

    for entry in lineup:
        player = entry.player

        players.append(
            {
                "roster_slot": entry.roster_slot,
                "player_id": player.player_id,
                "name": player.name,
                "position": player.position,
                "team": player.team,
                "opponent": player.opponent,
                "salary": player.salary,
                "projection": round(
                    float(player.projection),
                    4,
                ),
            }
        )

    return {
        "lineup_number": profile.lineup_number,
        "total_salary": profile.total_salary,
        "total_projection": round(
            float(profile.total_projection),
            4,
        ),
        "qb_stack_size": profile.qb_stack_size,
        "bring_back_size": profile.bring_back_size,
        "has_rb_dst": profile.has_rb_dst,
        "has_qb_vs_opposing_dst": (
            profile.has_qb_vs_opposing_dst
        ),
        "is_unstacked": profile.is_unstacked,
        "players": players,
    }


def _build_exposure_rows(result, player_lookup):
    rows = []

    for player_id, count in result.player_counts.items():
        player = player_lookup.get(player_id)

        rows.append(
            {
                "player_id": player_id,
                "name": (
                    player.name
                    if player is not None
                    else ""
                ),
                "position": (
                    player.position
                    if player is not None
                    else ""
                ),
                "team": (
                    player.team
                    if player is not None
                    else ""
                ),
                "count": count,
                "exposure": round(
                    float(result.exposures[player_id]),
                    6,
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            -row["exposure"],
            row["name"],
            row["player_id"],
        )
    )

    return rows


def _build_strategy_summary(profiles):
    lineup_count = len(profiles)
    summary = {}

    for strategy_name in sorted(SUPPORTED_STRATEGIES):
        count = sum(
            1
            for profile in profiles
            if _strategy_matches(
                profile,
                strategy_name,
            )
        )

        summary[strategy_name] = {
            "count": count,
            "exposure": (
                round(count / lineup_count, 6)
                if lineup_count
                else 0.0
            ),
        }

    return summary


def run_burn1_nfl_portfolio(
    players: list[Player],
    config: Burn1PortfolioConfig,
    status_callback: Callable[[Burn1StatusEvent], None] | None = None,
):
    _emit_status(
        status_callback,
        STATUS_VALIDATING,
        "Validating BURN1 portfolio configuration.",
    )

    try:
        _validate_config(config)
    except ValueError as exc:
        raise Burn1ApplicationError(
            code="INVALID_CONFIGURATION",
            stage=STATUS_VALIDATING,
            message=str(exc),
        ) from exc

    if not players:
        raise Burn1ApplicationError(
            code="EMPTY_PLAYER_POOL",
            stage=STATUS_VALIDATING,
            message="The player pool is empty.",
        )

    salary_cap = SUPPORTED_SITES[config.site]

    try:
        validate_player_pool(
            players,
            site=config.site,
            salary_cap=salary_cap,
            locked_player_ids=config.locked_player_ids,
            excluded_player_ids=config.excluded_player_ids,
            min_player_exposures=config.min_player_exposures,
            max_player_exposures=config.max_player_exposures,
        )
    except PreLockValidationError as exc:
        raise Burn1ApplicationError(
            code="PRELOCK_VALIDATION_FAILED",
            stage=STATUS_VALIDATING,
            message=str(exc),
        ) from exc

    _emit_status(
        status_callback,
        STATUS_STAGE1,
        "Generating Stage 1 candidate lineups.",
    )

    # Excluded players must never consume Stage 1 candidate slots.
    # Stage 2 still enforces the exclusion as a second safety layer.
    stage1_players = [
        player
        for player in players
        if player.player_id
        not in config.excluded_player_ids
    ]

    def candidate_progress_callback(
        current,
        total,
    ):
        if status_callback is None:
            return

        event = Burn1StatusEvent(
            status=STATUS_STAGE1,
            message="GENERATING",
        )

        event.progress_current = int(current)
        event.progress_total = int(total)

        status_callback(event)

    candidate_pool = generate_nfl_candidates(
        stage1_players,
        salary_cap=salary_cap,
        candidate_count=config.candidate_count,
        min_unique_players=config.min_unique_players,
        gpp_mode=config.gpp_mode,
        qb_stack_min=config.qb_stack_min,
        bring_back_min=config.bring_back_min,
        rb_dst_stack=config.rb_dst_stack,
        candidate_gpp_fraction=(
            config.candidate_gpp_fraction
        ),
        candidate_max_player_exposure=(
            config.candidate_max_player_exposure
        ),
        progress_callback=(
            candidate_progress_callback
        ),
    )

    if candidate_pool.generated_count < config.lineup_count:
        raise Burn1ApplicationError(
            code="STAGE1_INSUFFICIENT_CANDIDATES",
            stage=STATUS_STAGE1,
            message=(
                "Stage 1 generated only "
                f"{candidate_pool.generated_count} candidates, "
                "which is fewer than the requested "
                f"{config.lineup_count} final lineups."
            ),
        )

    _emit_status(
        status_callback,
        STATUS_STAGE2,
        "Selecting the final Stage 2 portfolio.",
    )

    try:
        selection = select_nfl_portfolio(
            candidate_pool,
            lineup_count=config.lineup_count,
            min_exposures=config.min_player_exposures,
            max_exposures=config.max_player_exposures,
            locked_player_ids=config.locked_player_ids,
            excluded_player_ids=config.excluded_player_ids,
            min_strategy_exposures=config.min_strategy_exposures,
            max_strategy_exposures=config.max_strategy_exposures,
        )
    except Stage2InfeasibleError as exc:
        raise Burn1ApplicationError(
            code="STAGE2_INFEASIBLE",
            stage=STATUS_STAGE2,
            message=str(exc),
        ) from exc
    except ValueError as exc:
        raise Burn1ApplicationError(
            code="INVALID_STAGE2_CONSTRAINTS",
            stage=STATUS_STAGE2,
            message=str(exc),
        ) from exc

    player_lookup = {
        player.player_id: player
        for player in players
    }

    # Stage 2 selects candidates created during Stage 1.
    # profile.lineup_number is therefore the Stage 1 candidate ID,
    # not the final portfolio position.
    #
    # Preserve the candidate ID for debugging/backtesting, but expose
    # clean final portfolio numbers 1..N to the application/UI/export.
    lineups = []

    for final_lineup_number, (lineup, profile) in enumerate(
        zip(
            selection.lineups,
            selection.profiles,
        ),
        start=1,
    ):
        serialized_lineup = _serialize_lineup(
            lineup,
            profile,
        )

        serialized_lineup["candidate_lineup_number"] = (
            serialized_lineup["lineup_number"]
        )

        serialized_lineup["lineup_number"] = (
            final_lineup_number
        )

        lineups.append(serialized_lineup)

    try:
        validate_final_lineups(
            lineups,
            site=config.site,
            salary_cap=salary_cap,
            expected_lineup_count=config.lineup_count,
        )
    except PreLockValidationError as exc:
        raise Burn1ApplicationError(
            code="FINAL_LINEUP_VALIDATION_FAILED",
            stage=STATUS_STAGE2,
            message=str(exc),
        ) from exc

    result = Burn1RunResult(
        status=STATUS_COMPLETE,
        site=config.site,
        requested_lineups=config.lineup_count,
        generated_lineups=selection.selected_lineup_count,
        requested_candidates=config.candidate_count,
        generated_candidates=candidate_pool.generated_count,
        solver_status=selection.solver_status,
        total_projection=round(
            float(selection.total_projection),
            4,
        ),
        lineups=lineups,
        exposures=_build_exposure_rows(
            selection,
            player_lookup,
        ),
        strategy_summary=_build_strategy_summary(
            selection.profiles
        ),
        message="BURN1 portfolio generation completed successfully.",
    )

    _emit_status(
        status_callback,
        STATUS_COMPLETE,
        result.message,
    )

    return result


def run_burn1_nfl_portfolio_safe(
    players: list[Player],
    config: Burn1PortfolioConfig,
    status_callback: Callable[[Burn1StatusEvent], None] | None = None,
):
    try:
        result = run_burn1_nfl_portfolio(
            players,
            config,
            status_callback=status_callback,
        )

        return Burn1ApplicationResponse(
            ok=True,
            status=STATUS_COMPLETE,
            result=result,
            error=None,
        )

    except Burn1ApplicationError as exc:
        _emit_status(
            status_callback,
            STATUS_ERROR,
            exc.message,
        )

        return Burn1ApplicationResponse(
            ok=False,
            status=STATUS_ERROR,
            result=None,
            error=Burn1ErrorResult(
                code=exc.code,
                stage=exc.stage,
                message=exc.message,
                details=exc.details,
            ),
        )

    except Exception as exc:
        message = (
            "BURN1 encountered an unexpected internal error."
        )

        _emit_status(
            status_callback,
            STATUS_ERROR,
            message,
        )

        return Burn1ApplicationResponse(
            ok=False,
            status=STATUS_ERROR,
            result=None,
            error=Burn1ErrorResult(
                code="INTERNAL_ERROR",
                stage="internal",
                message=message,
                details=str(exc),
            ),
        )
