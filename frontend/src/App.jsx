import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";
const RUN_HISTORY_KEY = "burn1RunHistoryV1";
const RUN_HISTORY_LIMIT = 10;

const STRATEGIES = [
  ["stacked", "Stacked Lineups"],
  ["unstacked", "Unstacked Lineups"],
  ["qb_stack_1_plus", "QB Stack (1+)"],
  ["qb_stack_2_plus", "QB Stack (2+)"],
  ["bring_back_1_plus", "Bring Back (1+)"],
  ["rb_dst", "RB + DST"],
  ["qb_vs_opposing_dst", "QB vs Opp DST"],
];

const VISIBLE_STRATEGIES = [
  ["stacked", "Stacked Lineups"],
  ["unstacked", "Unstacked Lineups"],
  ["qb_stack_1_plus", "QB Stack (1+)"],
  ["bring_back_1_plus", "Bring Back (1+)"],
  ["rb_dst", "RB + DST"],
];

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function siteName(site) {
  return site === "dk" ? "DraftKings" : "FanDuel";
}

function siteCode(site) {
  return site === "dk" ? "DK" : "FD";
}

function salaryCap(site) {
  return site === "dk" ? 50000 : 60000;
}

function readJsonStorage(storage, key, fallback) {
  try {
    const raw = storage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function readRunHistory() {
  const value = readJsonStorage(localStorage, RUN_HISTORY_KEY, []);
  return Array.isArray(value) ? value.slice(0, RUN_HISTORY_LIMIT) : [];
}

function formatRunTime(value) {
  try {
    return new Date(value).toLocaleString([], {
      month: "numeric",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function buildFractionMap(percentMap, defaultValue) {
  const output = {};

  Object.entries(percentMap).forEach(([key, rawValue]) => {
    const value = clamp(Number(rawValue), 0, 100);
    if (value !== defaultValue) {
      output[key] = value / 100;
    }
  });

  return output;
}

function verifyCompletedRun(result, config) {
  if (!result) {
    throw new Error("BURN1 returned an empty result.");
  }

  const requestedLineups = Number(config.lineup_count);
  const cap = salaryCap(config.site);

  if (result.site && result.site !== siteName(config.site)) {
    throw new Error(
      `Site verification failed: expected ${siteName(config.site)}, received ${result.site}.`
    );
  }

  if (Number(result.generated_lineups) !== requestedLineups) {
    throw new Error(
      `Portfolio verification failed: expected ${requestedLineups} lineups but received ${result.generated_lineups}.`
    );
  }

  if (!Array.isArray(result.lineups) || result.lineups.length !== requestedLineups) {
    throw new Error("Portfolio lineup payload does not match the requested count.");
  }

  if (!["OPTIMAL", "FEASIBLE"].includes(String(result.solver_status).toUpperCase())) {
    throw new Error(`Unexpected solver status: ${result.solver_status}.`);
  }

  const exposureCounts = {};
  const expectedSlots = {
    QB: 1,
    RB: 2,
    WR: 3,
    TE: 1,
    FLEX: 1,
    DST: 1,
  };

  result.lineups.forEach((lineup) => {
    if (!Array.isArray(lineup.players) || lineup.players.length !== 9) {
      throw new Error(`Lineup ${lineup.lineup_number} failed roster-size verification.`);
    }

    const playerIds = lineup.players.map((player) => String(player.player_id));
    if (new Set(playerIds).size !== playerIds.length) {
      throw new Error(`Lineup ${lineup.lineup_number} contains a duplicate player.`);
    }

    const totalSalary = Number(lineup.total_salary);
    if (!Number.isFinite(totalSalary) || totalSalary > cap) {
      throw new Error(`Lineup ${lineup.lineup_number} exceeds the ${siteName(config.site)} salary cap.`);
    }

    const slotCounts = {};
    lineup.players.forEach((player) => {
      const slot = String(player.roster_slot || "").toUpperCase();
      slotCounts[slot] = (slotCounts[slot] || 0) + 1;
      exposureCounts[String(player.player_id)] =
        (exposureCounts[String(player.player_id)] || 0) + 1;
    });

    Object.entries(expectedSlots).forEach(([slot, expected]) => {
      if ((slotCounts[slot] || 0) !== expected) {
        throw new Error(`Lineup ${lineup.lineup_number} failed ${slot} roster-slot verification.`);
      }
    });

    (config.locked_player_ids || []).forEach((playerId) => {
      if (!playerIds.includes(String(playerId))) {
        throw new Error(`Lock verification failed for player ${playerId}.`);
      }
    });

    (config.excluded_player_ids || []).forEach((playerId) => {
      if (playerIds.includes(String(playerId))) {
        throw new Error(`Exclude verification failed for player ${playerId}.`);
      }
    });
  });

  const minPlayer = config.min_player_exposures || {};
  const maxPlayer = config.max_player_exposures || {};
  const constrainedPlayerIds = new Set([
    ...Object.keys(minPlayer),
    ...Object.keys(maxPlayer),
  ]);

  constrainedPlayerIds.forEach((playerId) => {
    const actual = (exposureCounts[playerId] || 0) / requestedLineups;
    const minimum = Number(minPlayer[playerId] ?? 0);
    const maximum = Number(maxPlayer[playerId] ?? 1);

    if (actual + 1e-9 < minimum || actual - 1e-9 > maximum) {
      throw new Error(`Exposure verification failed for player ${playerId}.`);
    }
  });

  const strategySummary = result.strategy_summary || {};
  const minStrategy = config.min_strategy_exposures || {};
  const maxStrategy = config.max_strategy_exposures || {};
  const constrainedStrategies = new Set([
    ...Object.keys(minStrategy),
    ...Object.keys(maxStrategy),
  ]);

  constrainedStrategies.forEach((strategy) => {
    const actual = Number(strategySummary[strategy]?.exposure ?? 0);
    const minimum = Number(minStrategy[strategy] ?? 0);
    const maximum = Number(maxStrategy[strategy] ?? 1);

    if (actual + 1e-9 < minimum || actual - 1e-9 > maximum) {
      throw new Error(`Strategy verification failed for ${strategy}.`);
    }
  });

  return true;
}

function App() {
  const savedRun = readJsonStorage(sessionStorage, "burn1LastRun", null);
  const initialSite = savedRun?.site === "FanDuel" ? "fd" : "dk";

  const [site, setSite] = useState(initialSite);
  const [view, setView] = useState("optimizer");
  const [lineupsUnread, setLineupsUnread] = useState(false);
  const [lineupCount, setLineupCount] = useState(20);
  const [candidateCount, setCandidateCount] = useState(60);
  const [minUnique, setMinUnique] = useState(2);
  const [gppMix, setGppMix] = useState(50);
  const [qbStackMin, setQbStackMin] = useState(1);
  const [bringBackMin, setBringBackMin] = useState(0);
  const [rbDstStack, setRbDstStack] = useState(false);

  const [backendStatus, setBackendStatus] = useState("CONNECTING");
  const [backendDetail, setBackendDetail] = useState("Connecting to BURN1 API...");

  const [apiMeta, setApiMeta] = useState({
    version: "N/A",
    engine: "N/A",
    latencyMs: null,
    lastChecked: null,
  });

  const [systemCheckRunning, setSystemCheckRunning] = useState(false);

  const [playerPool, setPlayerPool] = useState([]);
  const [playerPoolMeta, setPlayerPoolMeta] = useState(null);
  const [playerPoolLoading, setPlayerPoolLoading] = useState(true);
  const [playerPoolError, setPlayerPoolError] = useState("");
  const [playerSearch, setPlayerSearch] = useState("");
  const [positionFilter, setPositionFilter] = useState("ALL");
  const [playerControlTab, setPlayerControlTab] = useState("locks");

  const [lockedPlayerIds, setLockedPlayerIds] = useState([]);
  const [excludedPlayerIds, setExcludedPlayerIds] = useState([]);
  const [minPlayerExposures, setMinPlayerExposures] = useState({});
  const [maxPlayerExposures, setMaxPlayerExposures] = useState({});
  const [minStrategyExposures, setMinStrategyExposures] = useState({});
  const [maxStrategyExposures, setMaxStrategyExposures] = useState({});
  const [focusedStrategy, setFocusedStrategy] = useState(null);

  const [runStatus, setRunStatus] = useState(savedRun ? "complete" : "ready");
  const [runMessage, setRunMessage] = useState(savedRun ? "Last completed portfolio restored." : "");
  const [runError, setRunError] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(savedRun);

  const [showPortfolioReady, setShowPortfolioReady] = useState(false);
const [runHistory, setRunHistory] = useState(readRunHistory);
  const [activeRunId, setActiveRunId] = useState(null);
  const [candidateProgress, setCandidateProgress] = useState({
    current: 0,
    total: 0,
  });

  const [soundEnabled, setSoundEnabled] = useState(() => {
    return localStorage.getItem("burn1SoundEnabled") !== "false";
  });
  const [toast, setToast] = useState(null);

  const toastTimerRef = useRef(null);
  const audioContextRef = useRef(null);

  const notify = (message, type = "info") => {
    setToast({ message, type });
    if (toastTimerRef.current) {
      window.clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = window.setTimeout(() => setToast(null), 2600);
  };

  function playTone(frequency = 620, duration = 0.06) {
    if (!soundEnabled) return;

    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
      }
      const context = audioContextRef.current;
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.value = frequency;
      gain.gain.value = 0.022;
      gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + duration);
      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start();
      oscillator.stop(context.currentTime + duration);
    } catch {
      // Sound is optional; never let it affect optimizer operation.
    }
  }

  async function checkBackend(showToast = false) {
    try {
      const healthStarted = performance.now();
      const response = await fetch(`${API}/api/health`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (!data.ok || data.status !== "ready") throw new Error("API not ready");
      const latencyMs = Math.max(
        0,
        Math.round(performance.now() - healthStarted)
      );

      setBackendStatus("READY");
      setBackendDetail(`BURN1 API ${data.api_version || "unknown"} ready`);

      setApiMeta({
        version: data.api_version || "unknown",
        engine: data.engine || "NFL V1",
        latencyMs,
        lastChecked: new Date().toISOString(),
      });
      if (showToast) notify("BURN1 API connection verified.", "success");
      return true;
    } catch (error) {
      setBackendStatus("OFFLINE");
      setBackendDetail(`API unavailable: ${error.message}`);

      setApiMeta((current) => ({
        ...current,
        latencyMs: null,
        lastChecked: new Date().toISOString(),
      }));
      if (showToast) notify("BURN1 API is offline.", "error");
      return false;
    }
  }

  async function loadPlayerPool(targetSite = site, showToast = false) {
    setPlayerPoolLoading(true);
    setPlayerPoolError("");

    try {
      const response = await fetch(`${API}/api/player-pool/${targetSite}`, {
        cache: "no-store",
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail || `HTTP ${response.status}`);
      }
      const data = await response.json();
      const players = Array.isArray(data.players) ? data.players : [];
      if (!players.length) throw new Error("Player pool is empty.");
      setPlayerPool(players);
      setPlayerPoolMeta(data);
      setPlayerPoolLoading(false);
      if (showToast) {
        notify(`${siteCode(targetSite)} player pool refreshed: ${players.length} players.`, "success");
      }
      return true;
    } catch (error) {
      setPlayerPool([]);
      setPlayerPoolMeta(null);
      setPlayerPoolError(error.message);
      setPlayerPoolLoading(false);
      if (showToast) notify(`Player pool error: ${error.message}`, "error");
      return false;
    }
  }

  useEffect(() => {
    checkBackend(false);
    const timer = window.setInterval(() => checkBackend(false), 15000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    loadPlayerPool(site, false);
  }, [site]);

  useEffect(() => {
    localStorage.setItem("burn1SoundEnabled", String(soundEnabled));
  }, [soundEnabled]);

  useEffect(() => {
    return () => {
      if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    };
  }, []);

  const playerById = (playerId) =>
    playerPool.find((player) => String(player.player_id) === String(playerId));

  const filteredPlayers = useMemo(() => {
    const query = playerSearch.trim().toLowerCase();

    return playerPool
      .filter((player) => {
        if (positionFilter !== "ALL" && player.position !== positionFilter) return false;
        if (!query) return true;
        return [player.name, player.team, player.opponent, player.position]
          .some((value) => String(value || "").toLowerCase().includes(query));
      })
      .sort((a, b) => Number(b.projection) - Number(a.projection));
  }, [playerPool, playerSearch, positionFilter]);

  const quickSearchPlayers = useMemo(() => {
    if (!playerSearch.trim()) return [];
    return filteredPlayers.slice(0, 6);
  }, [filteredPlayers, playerSearch]);

  const controlledExposureCount = useMemo(() => {
    const ids = new Set([
      ...Object.keys(minPlayerExposures),
      ...Object.keys(maxPlayerExposures),
    ]);
    return ids.size;
  }, [minPlayerExposures, maxPlayerExposures]);

  const topExposures = result?.exposures?.slice(0, 10) ?? [];
  const strategySummary = result?.strategy_summary ?? {};

  function clearSiteControls() {
    setLockedPlayerIds([]);
    setExcludedPlayerIds([]);
    setMinPlayerExposures({});
    setMaxPlayerExposures({});
    setMinStrategyExposures({});
    setMaxStrategyExposures({});
    setPlayerSearch("");
    setPositionFilter("ALL");
    setPlayerControlTab("locks");
    setFocusedStrategy(null);
  }

  function changeSite(nextSite) {
    if (running || nextSite === site) return;
    playTone(540);
    setSite(nextSite);
    setResult(null);
    setActiveRunId(null);
    setRunStatus("ready");
    setRunMessage("");
    setRunError("");
    sessionStorage.removeItem("burn1LastRun");
    clearSiteControls();
    notify(`${siteCode(nextSite)} selected. Site-specific controls reset for safety.`, "success");
  }

  function switchView(nextView) {

    if (nextView !== "optimizer") {
      setShowPortfolioReady(false);
    }

playTone(610);
    setView(nextView);

    if (nextView === "lineups") {
      setLineupsUnread(false);
    }

    if (nextView !== "optimizer") {
      setPlayerSearch("");
    }
  }

  function toggleLock(playerId) {
    const id = String(playerId);
    setLockedPlayerIds((current) =>
      current.includes(id) ? current.filter((value) => value !== id) : [...current, id]
    );
    setExcludedPlayerIds((current) => current.filter((value) => value !== id));
    playTone(720);
  }

  function toggleExclude(playerId) {
    const id = String(playerId);
    setExcludedPlayerIds((current) =>
      current.includes(id) ? current.filter((value) => value !== id) : [...current, id]
    );
    setLockedPlayerIds((current) => current.filter((value) => value !== id));
    playTone(260);
  }

  function updatePlayerExposure(playerId, type, rawValue) {
    const id = String(playerId);
    const value = clamp(Number(rawValue), 0, 100);

    if (type === "min") {
      const currentMax = Number(maxPlayerExposures[id] ?? 100);
      setMinPlayerExposures((current) => ({
        ...current,
        [id]: Math.min(value, currentMax),
      }));
    } else {
      const currentMin = Number(minPlayerExposures[id] ?? 0);
      setMaxPlayerExposures((current) => ({
        ...current,
        [id]: Math.max(value, currentMin),
      }));
    }
  }

  function updateStrategy(strategy, type, rawValue) {
    const value = clamp(Number(rawValue), 0, 100);

    if (type === "min") {
      const currentMax = Number(maxStrategyExposures[strategy] ?? 100);
      setMinStrategyExposures((current) => ({
        ...current,
        [strategy]: Math.min(value, currentMax),
      }));
    } else {
      const currentMin = Number(minStrategyExposures[strategy] ?? 0);
      setMaxStrategyExposures((current) => ({
        ...current,
        [strategy]: Math.max(value, currentMin),
      }));
    }
  }

  function resetOptimizerControls() {
    setLineupCount(20);
    setCandidateCount(60);
    setMinUnique(2);
    setGppMix(50);
    setQbStackMin(1);
    setBringBackMin(0);
    setRbDstStack(false);
    clearSiteControls();
    notify("Optimizer controls reset to proven baseline.", "success");
  }

  function getRunConfig() {
    return {
      site,
      lineup_count: Number(lineupCount),
      candidate_count: Math.max(Number(candidateCount), Number(lineupCount)),
      min_unique_players: Number(minUnique),
      candidate_gpp_fraction: Number(gppMix) / 100,
      qb_stack_min: Number(qbStackMin),
      bring_back_min: Number(bringBackMin),
      rb_dst_stack: Boolean(rbDstStack),
      locked_player_ids: [...lockedPlayerIds],
      excluded_player_ids: [...excludedPlayerIds],
      min_player_exposures: buildFractionMap(minPlayerExposures, 0),
      max_player_exposures: buildFractionMap(maxPlayerExposures, 100),
      min_strategy_exposures: buildFractionMap(minStrategyExposures, 0),
      max_strategy_exposures: buildFractionMap(maxStrategyExposures, 100),
    };
  }

  function validateBeforeRun(config) {
    if (backendStatus !== "READY") {
      throw new Error("BURN1 API is not ready.");
    }
    if (playerPoolLoading) {
      throw new Error("Player pool is still loading.");
    }
    if (playerPoolError || !playerPool.length) {
      throw new Error(`Player pool is unavailable: ${playerPoolError || "empty pool"}`);
    }
    if (config.candidate_count < config.lineup_count) {
      throw new Error("Candidate count must be at least the lineup count.");
    }
    if (config.locked_player_ids.length > 9) {
      throw new Error("An NFL lineup cannot contain more than 9 locked players.");
    }

    const conflicts = config.locked_player_ids.filter((id) =>
      config.excluded_player_ids.includes(id)
    );
    if (conflicts.length) {
      throw new Error(`${playerById(conflicts[0])?.name || conflicts[0]} cannot be both locked and excluded.`);
    }

    const validIds = new Set(playerPool.map((player) => String(player.player_id)));
    [...config.locked_player_ids, ...config.excluded_player_ids,
      ...Object.keys(config.min_player_exposures), ...Object.keys(config.max_player_exposures)]
      .forEach((id) => {
        if (!validIds.has(String(id))) {
          throw new Error(`A tournament control references a player not in the current ${siteCode(site)} pool.`);
        }
      });

    const playerIds = new Set([
      ...Object.keys(config.min_player_exposures),
      ...Object.keys(config.max_player_exposures),
    ]);

    playerIds.forEach((id) => {
      const minimum = Number(config.min_player_exposures[id] ?? 0);
      const maximum = Number(config.max_player_exposures[id] ?? 1);
      if (minimum > maximum) {
        throw new Error(`Minimum exposure exceeds maximum exposure for ${playerById(id)?.name || id}.`);
      }
      if (config.locked_player_ids.includes(id) && maximum < 1) {
        throw new Error(`Locked player ${playerById(id)?.name || id} must have 100% maximum exposure.`);
      }
      if (config.excluded_player_ids.includes(id) && minimum > 0) {
        throw new Error(`Excluded player ${playerById(id)?.name || id} cannot have a positive minimum exposure.`);
      }
    });

    const mins = config.min_strategy_exposures;
    const maxes = config.max_strategy_exposures;
    const strategyNames = new Set([...Object.keys(mins), ...Object.keys(maxes)]);
    strategyNames.forEach((name) => {
      if (Number(mins[name] ?? 0) > Number(maxes[name] ?? 1)) {
        throw new Error(`Minimum strategy exposure exceeds maximum for ${name}.`);
      }
    });

    if (Number(mins.stacked ?? 0) + Number(mins.unstacked ?? 0) > 1 + 1e-9) {
      throw new Error("Stacked and unstacked minimums cannot total more than 100%.");
    }
    if (Number(mins.qb_stack_1_plus ?? 0) + Number(mins.unstacked ?? 0) > 1 + 1e-9) {
      throw new Error("QB Stack 1+ and unstacked minimums cannot total more than 100%.");
    }
    if (Number(mins.qb_stack_2_plus ?? 0) > Number(maxes.qb_stack_1_plus ?? 1) + 1e-9) {
      throw new Error("QB Stack 2+ minimum cannot exceed the QB Stack 1+ maximum.");
    }

    return true;
  }

  function saveSuccessfulRun(completedResult, config) {
    const now = new Date();
    const entry = {
      id: `${now.getTime()}-${config.site}`,
      completed_at: now.toISOString(),
      site_key: config.site,
      site_name: siteName(config.site),
      slate: "Week 3 - Main",
      lineup_count: Number(config.lineup_count),
      candidate_count: Number(completedResult.generated_candidates ?? config.candidate_count),
      solver_status: completedResult.solver_status,
      total_projection: Number(completedResult.total_projection),
      config,
      result: completedResult,
    };

    setRunHistory((current) => {
      const next = [entry, ...current.filter((item) => item.id !== entry.id)]
        .slice(0, RUN_HISTORY_LIMIT);
      localStorage.setItem(RUN_HISTORY_KEY, JSON.stringify(next));
      return next;
    });

    return entry;
  }

  function loadHistoricalRun(entry) {
    if (running || !entry?.result) return;

    const config = entry.config || {
      site: entry.site_key,
      lineup_count: entry.lineup_count,
      locked_player_ids: [],
      excluded_player_ids: [],
      min_player_exposures: {},
      max_player_exposures: {},
      min_strategy_exposures: {},
      max_strategy_exposures: {},
    };

    verifyCompletedRun(entry.result, config);

    setSite(entry.site_key);
    setLineupCount(Number(entry.lineup_count || 20));
    setCandidateCount(Number(entry.config?.candidate_count || entry.candidate_count || 60));
    setMinUnique(Number(entry.config?.min_unique_players || 2));
    setGppMix(Math.round(Number(entry.config?.candidate_gpp_fraction ?? 0.5) * 100));
    setQbStackMin(Number(entry.config?.qb_stack_min ?? 1));
    setBringBackMin(Number(entry.config?.bring_back_min ?? 0));
    setRbDstStack(Boolean(entry.config?.rb_dst_stack));
    setLockedPlayerIds([...(entry.config?.locked_player_ids || [])]);
    setExcludedPlayerIds([...(entry.config?.excluded_player_ids || [])]);

    const minPlayer = {};
    Object.entries(entry.config?.min_player_exposures || {}).forEach(([key, value]) => {
      minPlayer[key] = Math.round(Number(value) * 100);
    });
    const maxPlayer = {};
    Object.entries(entry.config?.max_player_exposures || {}).forEach(([key, value]) => {
      maxPlayer[key] = Math.round(Number(value) * 100);
    });
    const minStrategy = {};
    Object.entries(entry.config?.min_strategy_exposures || {}).forEach(([key, value]) => {
      minStrategy[key] = Math.round(Number(value) * 100);
    });
    const maxStrategy = {};
    Object.entries(entry.config?.max_strategy_exposures || {}).forEach(([key, value]) => {
      maxStrategy[key] = Math.round(Number(value) * 100);
    });

    setMinPlayerExposures(minPlayer);
    setMaxPlayerExposures(maxPlayer);
    setMinStrategyExposures(minStrategy);
    setMaxStrategyExposures(maxStrategy);
    setResult(entry.result);
    setRunStatus(entry.result.solver_status || "COMPLETE");
    setRunMessage(`Loaded saved ${entry.site_name} portfolio.`);
    setRunError("");
    setActiveRunId(entry.id);
    sessionStorage.setItem("burn1LastRun", JSON.stringify(entry.result));
    notify(`${siteCode(entry.site_key)} portfolio restored from history.`, "success");
  }

  async function ignite() {
    if (running) return;

    const config = getRunConfig();

    try {
      validateBeforeRun(config);
    } catch (error) {
      setRunError(error.message);
      notify(error.message, "error");
      return;
    }

    setRunning(true);
    setShowPortfolioReady(false);
    setActiveRunId(null);
    setRunError("");
    setRunMessage("Ignition sequence started.");
    setRunStatus("queued");
    setCandidateProgress({
      current: 0,
      total: Number(config.candidate_count || 0),
    });
    sessionStorage.removeItem("burn1LastRun");
    playTone(180, 0.12);

    try {
      const startResponse = await fetch(`${API}/api/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });

      if (!startResponse.ok) {
        const payload = await startResponse.json().catch(() => null);
        throw new Error(payload?.detail || `BURN1 API returned HTTP ${startResponse.status}.`);
      }

      const started = await startResponse.json();
      if (!started.job_id) throw new Error("BURN1 API did not return a job ID.");

      while (true) {
        await new Promise((resolve) => window.setTimeout(resolve, 650));

        const response = await fetch(`${API}/api/runs/${started.job_id}`, { cache: "no-store" });
        if (!response.ok) throw new Error("Could not retrieve BURN1 run status.");

        const job = await response.json();
        setRunStatus(job.status || "running");
        setRunMessage(job.message || "");

        setCandidateProgress({
          current: Number(job.progress_current || 0),
          total: Number(
            job.progress_total ||
            config.candidate_count ||
            0
          ),
        });

        if (job.status === "complete") {
          verifyCompletedRun(job.result, config);
          const saved = saveSuccessfulRun(job.result, config);
          setResult(job.result);
          setShowPortfolioReady(true);
          setLineupsUnread(true);
          setActiveRunId(saved.id);
          setRunning(false);
          sessionStorage.setItem("burn1LastRun", JSON.stringify(job.result));
          playTone(820, 0.18);
          notify(`${siteCode(site)} portfolio verified and saved.`, "success");
          return;
        }

        if (job.status === "error") {
          throw new Error(job.error?.message || job.message || "BURN1 run failed.");
        }
      }
    } catch (error) {
      setRunStatus("error");
      setRunMessage("BURN1 run failed.");
      setRunError(error.message);
      setRunning(false);
      playTone(150, 0.2);
      notify(error.message, "error");
    }
  }

  function exportCsv() {
    if (!result?.lineups?.length) {
      notify("There is no completed portfolio to export.", "error");
      return;
    }

    const rows = [[
      "lineup_number", "roster_slot", "player_id", "name", "position",
      "team", "opponent", "salary", "projection", "lineup_salary", "lineup_projection",
    ]];

    result.lineups.forEach((lineup) => {
      lineup.players.forEach((player) => {
        rows.push([
          lineup.lineup_number, player.roster_slot, player.player_id, player.name,
          player.position, player.team, player.opponent, player.salary, player.projection,
          lineup.total_salary, lineup.total_projection,
        ]);
      });
    });

    const csv = rows
      .map((row) => row.map((value) => `"${String(value ?? "").replaceAll('"', '""')}"`).join(","))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `burn1_${site}_${lineupCount}_lineups.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
    notify("Portfolio CSV exported.", "success");
  }

  function useQuickPlayer(player) {
    if (playerControlTab === "locks") {
      toggleLock(player.player_id);
      notify(`${player.name} ${lockedPlayerIds.includes(String(player.player_id)) ? "unlocked" : "locked"}.`, "success");
    } else if (playerControlTab === "excludes") {
      toggleExclude(player.player_id);
      notify(`${player.name} ${excludedPlayerIds.includes(String(player.player_id)) ? "restored" : "excluded"}.`, "success");
    } else {
      const id = String(player.player_id);
      if (minPlayerExposures[id] === undefined) {
        setMinPlayerExposures((current) => ({ ...current, [id]: 0 }));
      }
      if (maxPlayerExposures[id] === undefined) {
        setMaxPlayerExposures((current) => ({ ...current, [id]: 100 }));
      }
      setView("playerPool");
      setPlayerSearch(player.name);
      notify(`Exposure controls opened for ${player.name}.`, "success");
    }
  }

  function formatSystemTime(value) {
    if (!value) {
      return "Not checked";
    }

    try {
      return new Date(value).toLocaleTimeString([], {
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
      });
    } catch {
      return "Unknown";
    }
  }

  async function runPlatformCheck() {
    if (systemCheckRunning) {
      return;
    }

    setSystemCheckRunning(true);

    try {
      const started = performance.now();

      const [healthResponse, poolResponse] =
        await Promise.all([
          fetch(`${API}/api/health`, {
            cache: "no-store",
          }),

          fetch(`${API}/api/player-pool/${site}`, {
            cache: "no-store",
          }),
        ]);

      if (!healthResponse.ok) {
        throw new Error(
          `Health check returned HTTP ${healthResponse.status}.`
        );
      }

      if (!poolResponse.ok) {
        throw new Error(
          `${siteCode(site)} pool returned HTTP ${poolResponse.status}.`
        );
      }

      const health = await healthResponse.json();
      const pool = await poolResponse.json();

      if (!health.ok || health.status !== "ready") {
        throw new Error(
          "BURN1 API did not report READY."
        );
      }

      if (Number(pool.player_count || 0) < 1) {
        throw new Error(
          `${siteCode(site)} player pool is empty.`
        );
      }

      const expectedCap =
        site === "dk"
          ? 50000
          : 60000;

      if (
        Number(pool.salary_cap) !==
        expectedCap
      ) {
        throw new Error(
          `${siteCode(site)} salary cap verification failed.`
        );
      }

      const latencyMs = Math.max(
        0,
        Math.round(
          performance.now() -
          started
        )
      );

      setBackendStatus("READY");

      setBackendDetail(
        `BURN1 API ${health.api_version || "unknown"} ready`
      );

      setApiMeta({
        version:
          health.api_version ||
          "unknown",

        engine:
          health.engine ||
          "NFL V1",

        latencyMs,

        lastChecked:
          new Date().toISOString(),
      });

      await loadPlayerPool(
        site,
        false
      );

      notify(
        `Platform check passed: ${siteCode(site)} ${pool.player_count} players.`,
        "success"
      );
    } catch (error) {
      setBackendStatus("OFFLINE");
      setBackendDetail(error.message);

      setApiMeta((current) => ({
        ...current,
        latencyMs: null,
        lastChecked:
          new Date().toISOString(),
      }));

      notify(
        `Platform check failed: ${error.message}`,
        "error"
      );
    } finally {
      setSystemCheckRunning(false);
    }
  }
  function openStrategy(strategy) {
    setFocusedStrategy(strategy);
    setView("settings");
    notify(`${STRATEGIES.find(([key]) => key === strategy)?.[1] || strategy} controls opened.`, "success");
  }

  const totalProjection = result ? Number(result.total_projection || 0).toFixed(2) : "â€”";
  const averageProjection = result?.generated_lineups
    ? (Number(result.total_projection || 0) / Number(result.generated_lineups)).toFixed(2)
    : "â€”";

  return (
    <div className="burn1-root">
      <div className={`master status-${runStatus}`}>
        <img className="master-image" src="/BURN1_UI_clean.png?v=20260927_195856" alt="BURN1 DFS Optimizer" />

        {/* BURN1 PROFESSIONAL JUMBOTRON START */}
        {view === "optimizer" && (
          <div
            className={`burn1-pro-jumbo status-${runStatus}`}
            aria-live="polite"
          >

            {/* LEFT ? SYSTEM MONITOR */}
            <section className="burn1-pro-system">
              <strong className="pro-screen-title">
                SYSTEM READY
              </strong>

              <span className="pro-screen-subtitle">
                {runStatus === "stage1_generating"
                  ? "GENERATING CANDIDATES"
                  : runStatus === "stage2_optimizing" ||
                    runStatus === "complete"
                    ? "CANDIDATES GENERATED"
                    : "CANDIDATE ENGINE READY"}
              </span>

              <div className="pro-candidate-row">
                <svg
                  className="pro-users-icon"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  aria-hidden="true"
                >
                  <path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5s-3 1.34-3 3 1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z" />
                </svg>

                <strong className="pro-candidate-count">
                  {runStatus === "stage1_generating"
                    ? `${candidateProgress.current || 0} / ${candidateProgress.total || candidateCount}`
                    : runStatus === "stage2_optimizing" ||
                      runStatus === "complete"
                      ? `${candidateProgress.total || result?.generated_candidates || candidateCount} / ${candidateProgress.total || result?.generated_candidates || candidateCount}`
                      : `${candidateCount} / ${candidateCount}`}
                </strong>
              </div>

              <div className="pro-segmented-bar">
                {[...Array(8)].map((_, i) => (
                  <span
                    key={i}
                    className={`bar-seg ${
                      runStatus === "complete" ||
                      runStatus === "stage2_optimizing" ||
                      (
                        runStatus === "stage1_generating" &&
                        (i / 8) <= (
                          (candidateProgress.current || 0) /
                          Math.max(
                            1,
                            candidateProgress.total || candidateCount
                          )
                        )
                      )
                        ? "active"
                        : ""
                    }`}
                  />
                ))}
              </div>
            </section>


            {/* CENTER ? PROCESS MONITOR */}
            <section className="burn1-pro-process">
              <div className="pro-process-track">

                <div
                  className={`pro-process-step step-1 ${
                    runStatus === "stage1_generating"
                      ? "active"
                      : runStatus === "stage2_optimizing" ||
                        runStatus === "complete"
                        ? "done"
                        : ""
                  }`}
                >
                  <div className="step-circle">
                    {runStatus === "stage2_optimizing" ||
                    runStatus === "complete" ? (
                      <svg
                        className="step-check"
                        viewBox="0 0 16 16"
                        fill="currentColor"
                        aria-hidden="true"
                      >
                        <path d="M13.485 3.515a1 1 0 010 1.414l-7 7a1 1 0 01-1.414 0l-3-3a1 1 0 011.414-1.414L6 10.086l6.293-6.293a1 1 0 011.414 0z" />
                      </svg>
                    ) : (
                      <span>1</span>
                    )}
                  </div>

                  <span>
                    CANDIDATES
                    <br />
                    GENERATED
                  </span>
                </div>

                <div
                  className={`pro-process-line line-1 ${
                    runStatus === "stage2_optimizing" ||
                    runStatus === "complete"
                      ? "active"
                      : ""
                  }`}
                />


                <div
                  className={`pro-process-step step-2 ${
                    runStatus === "stage2_optimizing"
                      ? "active"
                      : runStatus === "complete"
                        ? "done"
                        : ""
                  }`}
                >
                  <div className="step-circle">
                    {runStatus === "complete" ? (
                      <svg
                        className="step-check"
                        viewBox="0 0 16 16"
                        fill="currentColor"
                        aria-hidden="true"
                      >
                        <path d="M13.485 3.515a1 1 0 010 1.414l-7 7a1 1 0 01-1.414 0l-3-3a1 1 0 011.414-1.414L6 10.086l6.293-6.293a1 1 0 011.414 0z" />
                      </svg>
                    ) : (
                      <span>2</span>
                    )}
                  </div>

                  <span>
                    OPTIMIZATION
                    <br />
                    COMPLETE
                  </span>
                </div>

                <div
                  className={`pro-process-line line-2 ${
                    runStatus === "complete"
                      ? "active"
                      : ""
                  }`}
                />


                <div
                  className={`pro-process-step step-3 ${
                    runStatus === "complete"
                      ? "complete"
                      : ""
                  }`}
                >
                  <div className="step-circle">
                    {runStatus === "complete" ? (
                      <svg
                        className="step-check"
                        viewBox="0 0 16 16"
                        fill="currentColor"
                        aria-hidden="true"
                      >
                        <path d="M13.485 3.515a1 1 0 010 1.414l-7 7a1 1 0 01-1.414 0l-3-3a1 1 0 011.414-1.414L6 10.086l6.293-6.293a1 1 0 011.414 0z" />
                      </svg>
                    ) : (
                      <span>3</span>
                    )}
                  </div>

                  <span>
                    LINEUPS
                    <br />
                    CREATED
                  </span>
                </div>

              </div>
            </section>


            {/* RIGHT ? ENGINE MONITOR */}
            <section className="burn1-pro-engine">

              <strong className="pro-engine-title">
                ENGINE READY
              </strong>

              <div className="pro-engine-row ready">
                <span className="check-pip ready">?</span>
                <span>API READY</span>
              </div>

              <div className="pro-engine-row ready">
                <span className="check-pip ready">?</span>
                <span>PLAYER POOL READY</span>
              </div>

              <div
                className={`pro-engine-row ${
                  runStatus === "stage1_generating"
                    ? "active"
                    : runStatus === "stage2_optimizing" ||
                      runStatus === "complete"
                      ? "ready"
                      : ""
                }`}
              >
                <span
                  className={`check-pip ${
                    runStatus === "stage1_generating"
                      ? "active"
                      : runStatus === "stage2_optimizing" ||
                        runStatus === "complete"
                        ? "ready"
                        : "pending"
                  }`}
                >
                  {runStatus === "stage2_optimizing" ||
                  runStatus === "complete"
                    ? "?"
                    : "?"}
                </span>

                <span>
                  {candidateProgress.total ||
                    result?.generated_candidates ||
                    candidateCount} CANDIDATES
                </span>
              </div>

              <div
                className={`pro-engine-row ${
                  runStatus === "stage2_optimizing"
                    ? "active"
                    : runStatus === "complete"
                      ? "ready"
                      : ""
                }`}
              >
                <span
                  className={`check-pip ${
                    runStatus === "stage2_optimizing"
                      ? "active"
                      : runStatus === "complete"
                        ? "ready"
                        : "pending"
                  }`}
                >
                  {runStatus === "complete"
                    ? "?"
                    : "?"}
                </span>

                <span>FINALIZING</span>
              </div>

              <div
                className={`pro-engine-row optimized ${
                  runStatus === "complete"
                    ? "completed"
                    : ""
                }`}
              >
                <span
                  className={`check-pip ${
                    runStatus === "complete"
                      ? "completed"
                      : "pending"
                  }`}
                >
                  {runStatus === "complete"
                    ? "?"
                    : "?"}
                </span>

                <span>LINEUPS OPTIMIZED</span>
              </div>

            </section>

          </div>
        )}
        {/* BURN1 PROFESSIONAL JUMBOTRON END */}


        

        



        <div className={`connection-orb ${backendStatus.toLowerCase()}`} title={backendDetail} />

        {[
          ["optimizer", "nav-hit nav-optimizer"],
          ["lineups", "nav-hit nav-lineups"],
          ["exposures", "nav-hit nav-exposures"],
          ["stacks", "nav-hit nav-stacks"],
          ["playerPool", "nav-hit nav-playerpool"],
          ["settings", "nav-hit nav-settings"],
        ].map(([key, className]) => (
          <button
            key={key}
            className={`${className} ${
              view === key ? "active" : ""
            } ${
              key === "lineups" && lineupsUnread
                ? "has-new-lineups"
                : ""
            }`}
            aria-label={key}
            onClick={() => switchView(key)}
          />
        ))}
        
        {/* BURN1 DEFINITIVE NAVIGATION LAYER START */}
        <nav className="burn1-real-nav" aria-label="BURN1 main navigation">
          <button
            type="button"
            className={view === "optimizer" ? "active" : ""}
            onClick={() => switchView("optimizer")}
          >
            OPTIMIZER
          </button>

          <button
            type="button"
            className={view === "lineups" ? "active" : ""}
            onClick={() => switchView("lineups")}
          >
            LINEUPS
          </button>

          <button
            type="button"
            className={view === "exposures" ? "active" : ""}
            onClick={() => switchView("exposures")}
          >
            EXPOSURES
          </button>

          <button
            type="button"
            className={view === "stacks" ? "active" : ""}
            onClick={() => switchView("stacks")}
          >
            STACKS
          </button>

          <button
            type="button"
            className={view === "playerPool" ? "active" : ""}
            onClick={() => switchView("playerPool")}
          >
            PLAYER POOL
          </button>

          <button
            type="button"
            className={view === "settings" ? "active" : ""}
            onClick={() => switchView("settings")}
          >
            SETTINGS
          </button>
        </nav>
        {/* BURN1 DEFINITIVE NAVIGATION LAYER END */}

<button className="top-hit top-sport" onClick={() => { setView("settings"); notify("NFL V1 is the active sport engine."); }} aria-label="Sport" />
        <button className="top-hit top-slate" onClick={() => { setView("settings"); notify("Week 3 - Main is the active loaded slate."); }} aria-label="Slate" />
        <button className="top-hit top-gear" onClick={() => switchView("settings")} aria-label="Settings" />
        {/* BURN1 OPTIMIZER ONLY CONTROLS START */}
        {view === "optimizer" && (
          <>
        <button className="left-field-hit left-sport-hit" onClick={() => { setView("settings"); notify("NFL V1 is the active sport engine."); }} aria-label="Left sport selector" />
        <button className="left-field-hit left-slate-hit" onClick={() => { setView("settings"); notify("Week 3 - Main is the active loaded slate."); }} aria-label="Left slate selector" />
        <div
          className="burn1-left-clean-underlay"
          aria-hidden="true"
        >
          <span className="clean-mask clean-site-mask" />
          <span className="clean-mask clean-lineup-mask" />
          <span className="clean-mask clean-player-search-mask" />

          <span className="clean-mask clean-strategy-value sv-1" />
          <span className="clean-mask clean-strategy-value sv-2" />
          <span className="clean-mask clean-strategy-value sv-3" />
          <span className="clean-mask clean-strategy-value sv-4" />
          <span className="clean-mask clean-strategy-value sv-5" />
        </div>



        <button
          className={`site-button dk-button ${site === "dk" ? "selected" : ""}`}
          disabled={running}
          onClick={() => changeSite("dk")}
        >
          DraftKings
        </button>
        <button
          className={`site-button fd-button ${site === "fd" ? "selected" : ""}`}
          disabled={running}
          onClick={() => changeSite("fd")}
        >
          FanDuel
        </button>

        <div className="lineup-value">{lineupCount}</div>

        <button
          className={`player-tab-hit player-lock-tab ${playerControlTab === "locks" ? "selected" : ""}`}
          onClick={() => { setPlayerControlTab("locks"); setPlayerSearch(""); }}
          aria-label="Locks"
        />
        <button
          className={`player-tab-hit player-exclude-tab ${playerControlTab === "excludes" ? "selected" : ""}`}
          onClick={() => { setPlayerControlTab("excludes"); setPlayerSearch(""); }}
          aria-label="Excludes"
        />
        <button
          className={`player-tab-hit player-exposure-tab ${playerControlTab === "exposures" ? "selected" : ""}`}
          onClick={() => { setPlayerControlTab("exposures"); setPlayerSearch(""); }}
          aria-label="Exposures"
        />

        <div className="player-tab-count lock-count">{lockedPlayerIds.length}</div>
        <div className="player-tab-count exclude-count">{excludedPlayerIds.length}</div>
        <div className="player-tab-count exposure-count">{controlledExposureCount}</div>



        {VISIBLE_STRATEGIES.map(([strategy], index) => (
          <button
            key={strategy}
            className={`strategy-hit strategy-hit-${index + 1}`}
            onClick={() => openStrategy(strategy)}
            aria-label={`${strategy} controls`}
          />
        ))}


        <button
          className={`ignite-hit ${running ? "running" : ""}`}
          disabled={running || backendStatus !== "READY"}
          onClick={ignite}
          aria-label="Ignite BURN1"
          style={{
            "--ignite-progress":
              runStatus === "stage2_optimizing"
                ? 1
                : runStatus === "stage1_generating"
                  ? Math.min(
                      1,
                      Number(candidateProgress.current || 0) /
                        Math.max(
                          1,
                          Number(
                            candidateProgress.total ||
                            candidateCount ||
                            1
                          )
                        )
                    )
                  : runStatus === "validating"
                    ? 0.08
                    : runStatus === "queued"
                      ? 0.04
                      : 0,
          }}
        >
          <span className="ignite-label">
            {running
              ? runStatus === "stage2_optimizing"
                ? "SELECTING PORTFOLIO"
                : runStatus === "stage1_generating"
                  ? `GENERATING ${candidateProgress.current || 0}/${candidateProgress.total || candidateCount}`
                  : runStatus === "validating"
                    ? "VALIDATING"
                    : "STARTING"
              : "IGNITE"}
          </span>
        </button>

        <button
          className="burn1-home-hit"
          type="button"
          onClick={() => switchView("optimizer")}
          aria-label="BURN1 home"
          title="Return to BURN1 Optimizer"
        />

        <button className="view-all-hit" onClick={() => setView("exposures")} aria-label="View all exposures" />
          </>
        )}
        {/* BURN1 OPTIMIZER ONLY CONTROLS END */}


        {view === "optimizer" && (
          <>
            {(running || showPortfolioReady) && (
              <div
                className={`burn1-jumbotron status-${runStatus}`}
                aria-live="polite"
              >
              <section className="jumbo-screen jumbo-left">
                <div className="jumbo-screen-glass" />

                <div className="jumbo-content">
                  {runStatus === "stage1_generating" ? (
                    <>
                      <small>GENERATING</small>
                      <span>CANDIDATES</span>

                      <strong>
                        {candidateProgress.current || 0}
                        <b>/</b>
                        {candidateProgress.total || candidateCount}
                      </strong>

                      <div className="jumbo-progress-track">
                        <i
                          style={{
                            width: `${
                              Math.min(
                                1,
                                Number(candidateProgress.current || 0) /
                                  Math.max(
                                    1,
                                    Number(
                                      candidateProgress.total ||
                                      candidateCount ||
                                      1
                                    )
                                  )
                              ) * 100
                            }%`,
                          }}
                        />
                      </div>
                    </>
                  ) : runStatus === "stage2_optimizing" ? (
                    <>
                      <small>CANDIDATES</small>
                      <span>COMPLETE</span>

                      <strong>
                        {candidateProgress.total || candidateCount}
                        <b>/</b>
                        {candidateProgress.total || candidateCount}
                      </strong>

                      <div className="jumbo-progress-track complete">
                        <i style={{ width: "100%" }} />
                      </div>
                    </>
                  ) : runStatus === "complete" && showPortfolioReady && result ? (
                    <>
                      <small>PORTFOLIO</small>
                      <span>READY</span>

                      <strong>
                        {result.generated_lineups}
                        <b className="jumbo-unit"> LINEUPS</b>
                      </strong>

                      <div className="jumbo-progress-track complete">
                        <i style={{ width: "100%" }} />
                      </div>
                    </>
                  ) : (
                    <>
                      <small>BURN1</small>
                      <span>READY</span>

                      <strong className="jumbo-ready">
                        IGNITE
                      </strong>

                      <div className="jumbo-progress-track">
                        <i style={{ width: "0%" }} />
                      </div>
                    </>
                  )}
                </div>
              </section>


              <section className="jumbo-screen jumbo-right">
                <div className="jumbo-screen-glass" />

                <div className="jumbo-status-content">
                  <small>
                    {runStatus === "complete"
                      ? "STAGE 3 OF 3"
                      : runStatus === "stage2_optimizing"
                        ? "STAGE 2 OF 3"
                        : running
                          ? "STAGE 1 OF 3"
                          : "SYSTEM READY"}
                  </small>

                  {runStatus === "stage1_generating" ? (
                    <>
                      <div className="jumbo-status-row done">
                        <i /> PLAYER POOL READY
                      </div>

                      <div className="jumbo-status-row active">
                        <i /> BUILDING CANDIDATES
                      </div>

                      <div className="jumbo-status-row">
                        <i /> {candidateProgress.current || 0}/
                        {candidateProgress.total || candidateCount}
                      </div>

                      <div className="jumbo-status-row">
                        <i /> TARGET {lineupCount} LINEUPS
                      </div>
                    </>
                  ) : runStatus === "stage2_optimizing" ? (
                    <>
                      <div className="jumbo-status-row done">
                        <i /> CANDIDATES COMPLETE
                      </div>

                      <div className="jumbo-status-row active">
                        <i /> SELECTING PORTFOLIO
                      </div>

                      <div className="jumbo-status-row">
                        <i /> TARGET {lineupCount} LINEUPS
                      </div>

                      <div className="jumbo-status-row">
                        <i /> CP-SAT OPTIMIZER
                      </div>
                    </>
                  ) : runStatus === "complete" && showPortfolioReady && result ? (
                    <>
                      <div className="jumbo-complete-title">
                        PORTFOLIO COMPLETE
                      </div>

                      <div className="jumbo-result-grid">
                        <div>
                          <span>LINEUPS</span>
                          <strong>{result.generated_lineups}</strong>
                        </div>

                        <div>
                          <span>TOTAL</span>
                          <strong>{totalProjection}</strong>
                        </div>

                        <div>
                          <span>AVG</span>
                          <strong>{averageProjection}</strong>
                        </div>

                        <div>
                          <span>STATUS</span>
                          <strong>{result.solver_status}</strong>
                        </div>
                      </div>

                      <div className="jumbo-lineups-hint">
                        LINEUPS TAB READY
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="jumbo-status-row done">
                        <i /> API READY
                      </div>

                      <div className="jumbo-status-row done">
                        <i /> {siteCode(site)} PLAYER POOL
                      </div>

                      <div className="jumbo-status-row">
                        <i /> {candidateCount} CANDIDATES
                      </div>

                      <div className="jumbo-status-row">
                        <i /> {lineupCount} FINAL LINEUPS
                      </div>
                    </>
                  )}
                </div>
              </section>


              <div
                className={`jumbo-stage-lights status-${runStatus}`}
                aria-hidden="true"
              >
                <span className="jumbo-stage-light jumbo-stage-1" />
                <span className="jumbo-stage-light jumbo-stage-2" />
                <span className="jumbo-stage-light jumbo-stage-3" />
              </div>
            </div>
            )}



            <div
              className={`burn1-live-stage-label status-${runStatus}`}
              aria-live="polite"
            >
              {runStatus === "stage1_generating"
                ? `GENERATING CANDIDATES ${candidateProgress.current || 0}/${candidateProgress.total || 0}`
                : runStatus === "stage2_optimizing"
                  ? "SELECTING PORTFOLIO"
                  : runStatus === "complete"
                    ? "PORTFOLIO READY"
                    : runStatus === "validating"
                      ? "VALIDATING RUN"
                      : "BURN1 OPTIMIZATION ENGINE"}
            </div>

            <div
              className={`burn1-stage-overlay status-${runStatus}`}
              aria-hidden="true"
            >
              <span className="burn1-stage-node stage-node-1">
                <i>
                  {runStatus === "stage2_optimizing" ||
                  runStatus === "complete"
                    ? "✓"
                    : ""}
                </i>
              </span>

              <span className="burn1-stage-node stage-node-2">
                <i>
                  {runStatus === "complete"
                    ? "✓"
                    : ""}
                </i>
              </span>

              <span className="burn1-stage-node stage-node-3">
                <i>
                  {runStatus === "complete"
                    ? "✓"
                    : ""}
                </i>
              </span>
            </div>

            <div
              className={`burn1-progress-engine status-${runStatus} ${
                running ? "running" : ""
              }`}
              style={{
                "--burn1-progress-scale":
                  runStatus === "complete"
                    ? 1
                    : runStatus === "stage2_optimizing"
                      ? 0.90
                      : runStatus === "stage1_generating"
                        ? 0.12 + (
                            0.68 *
                            Math.min(
                              1,
                              Number(candidateProgress.current || 0) /
                              Math.max(
                                1,
                                Number(candidateProgress.total || 1)
                              )
                            )
                          )
                        : runStatus === "validating"
                          ? 0.12
                          : runStatus === "queued"
                            ? 0.06
                            : 0,
              }}
              aria-hidden="true"
            >
              <div className="burn1-progress-track">
                <div className="burn1-progress-fill">
                  <span className="burn1-progress-light" />
                </div>
              </div>
            </div>

            {running && (
              <div
                className={`burn1-fire-stage status-${runStatus}`}
                aria-hidden="true"
              >
                <span className="burn1-flame flame-1" />
                <span className="burn1-flame flame-2" />
                <span className="burn1-flame flame-3" />
                <span className="burn1-flame flame-4" />
                <span className="burn1-heat-wave" />
              </div>
            )}

            {running && (
              <svg
                className={`burn1-route-field status-${runStatus}`}
                viewBox="0 0 1983 793"
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                <defs>
                  <filter
                    id="burn1-blue-glow"
                    x="-100%"
                    y="-100%"
                    width="300%"
                    height="300%"
                  >
                    <feGaussianBlur
                      stdDeviation="6"
                      result="blur"
                    />
                    <feMerge>
                      <feMergeNode in="blur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>

                  <filter
                    id="burn1-orange-glow"
                    x="-100%"
                    y="-100%"
                    width="300%"
                    height="300%"
                  >
                    <feGaussianBlur
                      stdDeviation="7"
                      result="blur"
                    />
                    <feMerge>
                      <feMergeNode in="blur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                </defs>

                <g className="burn1-blue-routes">

                  {/* QB -> RB */}
                  <path
                    id="burn1-blue-route-1"
                    className="burn1-route blue"
                    d="M575 454 C630 452 684 448 738 445"
                  />

                  {/* QB -> WR1 */}
                  <path
                    id="burn1-blue-route-2"
                    className="burn1-route blue"
                    d="M575 454 C646 424 710 357 790 364"
                  />

                  {/* QB -> TE */}
                  <path
                    id="burn1-blue-route-3"
                    className="burn1-route blue"
                    d="M575 454 C675 440 775 430 886 437"
                  />

                  {/* QB -> WR2 */}
                  <path
                    id="burn1-blue-route-4"
                    className="burn1-route blue"
                    d="M575 454 C700 413 855 383 1009 388"
                  />

                  <circle
                    className="burn1-tracker blue"
                    r="7"
                  >
                    <animateMotion
                      dur="1.35s"
                      repeatCount="indefinite"
                    >
                      <mpath href="#burn1-blue-route-1" />
                    </animateMotion>
                  </circle>

                  <circle
                    className="burn1-tracker blue secondary"
                    r="6"
                  >
                    <animateMotion
                      dur="1.60s"
                      repeatCount="indefinite"
                    >
                      <mpath href="#burn1-blue-route-2" />
                    </animateMotion>
                  </circle>

                  <circle
                    className="burn1-tracker blue tertiary"
                    r="6"
                  >
                    <animateMotion
                      dur="1.82s"
                      repeatCount="indefinite"
                    >
                      <mpath href="#burn1-blue-route-3" />
                    </animateMotion>
                  </circle>

                  <circle
                    className="burn1-tracker blue quaternary"
                    r="6"
                  >
                    <animateMotion
                      dur="2.02s"
                      repeatCount="indefinite"
                    >
                      <mpath href="#burn1-blue-route-4" />
                    </animateMotion>
                  </circle>

                </g>

                <g className="burn1-orange-routes">

                  {/* QB -> RB */}
                  <path
                    id="burn1-orange-route-1"
                    className="burn1-route orange"
                    d="M1643 454 C1575 451 1513 448 1451 445"
                  />

                  {/* QB -> WR1 */}
                  <path
                    id="burn1-orange-route-2"
                    className="burn1-route orange"
                    d="M1643 454 C1572 420 1505 358 1431 364"
                  />

                  {/* QB -> TE */}
                  <path
                    id="burn1-orange-route-3"
                    className="burn1-route orange"
                    d="M1643 454 C1540 440 1420 431 1303 445"
                  />

                  {/* QB -> WR2 */}
                  <path
                    id="burn1-orange-route-4"
                    className="burn1-route orange"
                    d="M1643 454 C1488 418 1300 400 1139 409"
                  />

                  <circle
                    className="burn1-tracker orange"
                    r="7"
                  >
                    <animateMotion
                      dur="1.35s"
                      repeatCount="indefinite"
                    >
                      <mpath href="#burn1-orange-route-1" />
                    </animateMotion>
                  </circle>

                  <circle
                    className="burn1-tracker orange secondary"
                    r="6"
                  >
                    <animateMotion
                      dur="1.60s"
                      repeatCount="indefinite"
                    >
                      <mpath href="#burn1-orange-route-2" />
                    </animateMotion>
                  </circle>

                  <circle
                    className="burn1-tracker orange tertiary"
                    r="6"
                  >
                    <animateMotion
                      dur="1.82s"
                      repeatCount="indefinite"
                    >
                      <mpath href="#burn1-orange-route-3" />
                    </animateMotion>
                  </circle>

                  <circle
                    className="burn1-tracker orange quaternary"
                    r="6"
                  >
                    <animateMotion
                      dur="2.02s"
                      repeatCount="indefinite"
                    >
                      <mpath href="#burn1-orange-route-4" />
                    </animateMotion>
                  </circle>

                </g>
              </svg>
            )}
          </>
        )}

        {view === "optimizer" && running && (
          <section className={`burn1-run-hud status-${runStatus}`}>
            <div className="hud-kicker">
              {runStatus === "stage2_optimizing"
                ? "SELECTING PORTFOLIO"
                : runStatus === "validating"
                  ? "VALIDATING"
                  : "GENERATING CANDIDATES"}
            </div>

            <div className="hud-main">
              <strong>
                {runStatus === "stage1_generating"
                  ? `${candidateProgress.current || 0}/${candidateProgress.total || candidateCount}`
                  : runStatus === "stage2_optimizing"
                    ? `${candidateProgress.total || candidateCount} CANDIDATES`
                    : siteCode(site)}
              </strong>

              <span>
                {siteCode(site)} · WEEK 3 MAIN
              </span>
            </div>

            <div className="hud-grid">
              <div>
                <span>FINAL</span>
                <strong>{lineupCount}</strong>
              </div>

              <div>
                <span>ENGINE</span>
                <strong>CP-SAT</strong>
              </div>

              <div>
                <span>STATUS</span>
                <strong>
                  {runStatus === "stage2_optimizing"
                    ? "SELECTING"
                    : runStatus === "validating"
                      ? "VALIDATING"
                      : "GENERATING"}
                </strong>
              </div>
            </div>
          </section>
        )}

        {view === "optimizer" && showPortfolioReady && result && !running && (
          <section className="burn1-ready-hud">
            <div className="ready-copy">
              <small>PORTFOLIO READY</small>

              <strong>
                {result.generated_lineups} LINEUPS
              </strong>

              <span>
                {siteCode(site)} · {result.solver_status}
              </span>
            </div>

            <div className="ready-stats">
              <div>
                <span>TOTAL PROJ</span>
                <strong>{totalProjection}</strong>
              </div>

              <div>
                <span>AVG PROJ</span>
                <strong>{averageProjection}</strong>
              </div>
            </div>

            <button
              type="button"
              onClick={() => switchView("lineups")}
            >
              VIEW LINEUPS
              <b>→</b>
            </button>
          </section>
        )}

        {result && !running && (
          <>
            <section className="live-summary">
              <div><strong>{result.generated_lineups}</strong><span>Lineups</span></div>
              <div><strong>{totalProjection}</strong><span>Total Projection</span></div>
              <div><strong>{averageProjection}</strong><span>Avg Projection</span></div>
              <div><strong className="accent">{result.solver_status}</strong><span>Solver Status</span></div>
            </section>

            <section className="live-exposures">
              {topExposures.map((player) => {
                const exposure = Math.round(Number(player.exposure || 0) * 100);
                return (
                  <div className="live-exposure-row" key={player.player_id}>
                    <span className="exp-name">{player.name}</span>
                    <span>{player.position}</span>
                    <span>{player.team}</span>
                    <div className="exp-track"><i style={{ width: `${exposure}%` }} /></div>
                    <strong>{exposure}%</strong>
                  </div>
                );
              })}
            </section>
          </>
        )}

        {runError && view === "optimizer" && <div className="run-error">{runError}</div>}
        {running && runMessage && view === "optimizer" && <div className="run-message">{runMessage}</div>}

        {view !== "optimizer" && (
          <section className="detail-view">
            <header className="detail-header">
              <div className="detail-title">
                <div className="eyebrow">BURN1 DFS  |  {siteCode(site)}  |  WEEK 3 MAIN</div>
                <h1>{view === "playerPool" ? "PLAYER POOL" : view.toUpperCase()}</h1>
              </div>
              <div className="detail-header-actions">
                <div className={`api-pill ${backendStatus.toLowerCase()}`}><i /> API {backendStatus}</div>
                <button onClick={() => setView("optimizer")}>RETURN TO OPTIMIZER</button>
              </div>
            </header>

            {view === "lineups" && (
              <div className="detail-content">
                <section className="current-portfolio-bar">
                  <strong>{siteCode(site)}</strong>
                  <span>Week 3 - Main</span>
                  <span>{result ? `${result.generated_lineups} Lineups` : "No active portfolio"}</span>
                  {result && <span className="accent portfolio-status">{result.solver_status}</span>}
                  <button disabled={!result} onClick={exportCsv}>EXPORT CSV</button>
                </section>

                <section className="history-panel">
                  <div className="section-heading"><strong>RECENT PORTFOLIOS</strong><span>{runHistory.length}/{RUN_HISTORY_LIMIT} SAVED</span></div>
                  {runHistory.length === 0 ? (
                    <div className="empty-state">Completed DK and FD portfolios will be retained here automatically.</div>
                  ) : (
                    <div className="history-grid">
                      {runHistory.map((entry) => (
                        <button key={entry.id} className={`history-card ${activeRunId === entry.id ? "active" : ""}`} onClick={() => loadHistoricalRun(entry)}>
                          <div><b className={entry.site_key === "dk" ? "dk-badge" : "fd-badge"}>{siteCode(entry.site_key)}</b><span>{formatRunTime(entry.completed_at)}</span></div>
                          <strong>{entry.lineup_count} LINEUPS</strong>
                          <small>{Number(entry.total_projection).toFixed(2)} TOTAL PROJ  |  {entry.solver_status}</small>
                        </button>
                      ))}
                    </div>
                  )}
                </section>

                {!result ? (
                  <div className="empty-state large">No completed portfolio loaded. Return to OPTIMIZER and IGNITE BURN1.</div>
                ) : (
                  <div className="lineup-grid">
                    {result.lineups.map((lineup) => (
                      <article className="lineup-card" key={lineup.lineup_number}>
                        <header><strong>LINEUP {lineup.lineup_number}</strong><span>${Number(lineup.total_salary).toLocaleString()}  |  {Number(lineup.total_projection).toFixed(2)}</span></header>
                        {lineup.players.map((player) => (
                          <div className="lineup-player" key={`${lineup.lineup_number}-${player.player_id}-${player.roster_slot}`}>
                            <strong>{player.roster_slot}</strong><span>{player.name}</span><span>{player.team}{player.opponent ? ` vs ${player.opponent}` : ""}</span><span>${Number(player.salary).toLocaleString()}</span><b>{Number(player.projection).toFixed(2)}</b>
                          </div>
                        ))}
                      </article>
                    ))}
                  </div>
                )}
              </div>
            )}

            {view === "exposures" && (
              <div className="detail-content">
                {!result ? (
                  <div className="empty-state large">No completed portfolio loaded.</div>
                ) : (
                  <div className="exposure-table">
                    <div className="table-head"><span>Player</span><span>Pos</span><span>Team</span><span>Exposure</span><span>Count</span></div>
                    {result.exposures.map((player) => {
                      const exposure = Math.round(Number(player.exposure || 0) * 100);
                      return (
                        <div className="exposure-table-row" key={player.player_id}>
                          <strong>{player.name}</strong><span>{player.position}</span><span>{player.team}</span>
                          <div className="wide-track"><i style={{ width: `${exposure}%` }} /><b>{exposure}%</b></div>
                          <span>{player.count}/{result.generated_lineups}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {view === "stacks" && (
              <div className="detail-content">
                {!result ? (
                  <div className="empty-state large">No completed portfolio loaded.</div>
                ) : (
                  <div className="strategy-report-grid">
                    {STRATEGIES.map(([key, label]) => {
                      const metrics = strategySummary[key] || { count: 0, exposure: 0 };
                      return (
                        <article key={key} className="strategy-report-card">
                          <span>{label}</span>
                          <strong>{metrics.count}</strong>
                          <small>{Math.round(Number(metrics.exposure || 0) * 100)}% OF PORTFOLIO</small>
                        </article>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {view === "playerPool" && (
              <div className="detail-content">
                <div className="pool-toolbar">
                  <input value={playerSearch} onChange={(event) => setPlayerSearch(event.target.value)} placeholder="Search player, team, opponent or position..." />
                  <select value={positionFilter} onChange={(event) => setPositionFilter(event.target.value)}>
                    {['ALL', 'QB', 'RB', 'WR', 'TE', 'DST'].map((position) => <option key={position}>{position}</option>)}
                  </select>
                  <button onClick={() => loadPlayerPool(site, true)}>REFRESH POOL</button>
                  <strong>{playerPoolLoading ? "LOADING..." : `${filteredPlayers.length} PLAYERS`}</strong>
                </div>

                {playerPoolError ? <div className="error-banner">{playerPoolError}</div> : null}

                <div className="player-pool-table">
                  <div className="player-pool-head">
                    <span>Player</span><span>Pos</span><span>Matchup</span><span>Salary</span><span>Proj</span><span>Lock</span><span>Exclude</span><span>Min%</span><span>Max%</span>
                  </div>
                  {filteredPlayers.map((player) => {
                    const id = String(player.player_id);
                    const locked = lockedPlayerIds.includes(id);
                    const excluded = excludedPlayerIds.includes(id);
                    return (
                      <div className="player-pool-row" key={id}>
                        <strong>{player.name}</strong><span>{player.position}</span><span>{player.team}{player.opponent ? ` vs ${player.opponent}` : ""}</span><span>${Number(player.salary).toLocaleString()}</span><b>{Number(player.projection).toFixed(2)}</b>
                        <button className={locked ? "toggle active" : "toggle"} onClick={() => toggleLock(id)}>{locked ? "LOCKED" : "LOCK"}</button>
                        <button className={excluded ? "toggle danger active" : "toggle danger"} onClick={() => toggleExclude(id)}>{excluded ? "OUT" : "EXCLUDE"}</button>
                        <input type="number" min="0" max="100" value={minPlayerExposures[id] ?? 0} onChange={(event) => updatePlayerExposure(id, "min", event.target.value)} />
                        <input type="number" min="0" max="100" value={maxPlayerExposures[id] ?? 100} onChange={(event) => updatePlayerExposure(id, "max", event.target.value)} />
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {view === "settings" && (
              <div className="detail-content settings-grid">
                <section className="settings-card strategy-settings-card">
                  <div className="section-heading"><strong>FINAL PORTFOLIO STRATEGY LIMITS</strong><span>Stage 2</span></div>
                  {STRATEGIES.map(([key, label]) => (
                    <div key={key} className={`strategy-editor ${focusedStrategy === key ? "focused" : ""}`}>
                      <div className="strategy-editor-head"><strong>{label}</strong><span>{minStrategyExposures[key] ?? 0}% - {maxStrategyExposures[key] ?? 100}%</span></div>
                      <div className="dual-slider-row">
                        <label>MIN<input type="range" min="0" max="100" step="5" value={minStrategyExposures[key] ?? 0} onChange={(event) => updateStrategy(key, "min", event.target.value)} /></label>
                        <input className="number-box" type="number" min="0" max="100" value={minStrategyExposures[key] ?? 0} onChange={(event) => updateStrategy(key, "min", event.target.value)} />
                        <label>MAX<input type="range" min="0" max="100" step="5" value={maxStrategyExposures[key] ?? 100} onChange={(event) => updateStrategy(key, "max", event.target.value)} /></label>
                        <input className="number-box" type="number" min="0" max="100" value={maxStrategyExposures[key] ?? 100} onChange={(event) => updateStrategy(key, "max", event.target.value)} />
                      </div>
                    </div>
                  ))}
                </section>

                <section className="settings-card">
                  <div className="section-heading"><strong>PORTFOLIO ENGINE</strong><span>Stage 1</span></div>
                  <label className="setting-row"><span>Number of Lineups<small>Final portfolio size</small></span><input type="number" min="1" max="50" value={lineupCount} onChange={(event) => { const value = clamp(Number(event.target.value), 1, 50); setLineupCount(value); setCandidateCount((current) => Math.max(current, value)); }} /></label>
                  <label className="setting-row"><span>Candidate Pool<small>Stage 1 candidate lineups</small></span><input type="number" min={lineupCount} max="500" value={candidateCount} onChange={(event) => setCandidateCount(clamp(Number(event.target.value), lineupCount, 500))} /></label>
                  <label className="setting-row"><span>Minimum Unique<small>Players different between candidates</small></span><input type="number" min="1" max="9" value={minUnique} onChange={(event) => setMinUnique(clamp(Number(event.target.value), 1, 9))} /></label>
                  <label className="setting-row"><span>GPP Candidate Mix<small>Percent of Stage 1 generated in GPP mode</small></span><input type="number" min="0" max="100" value={gppMix} onChange={(event) => setGppMix(clamp(Number(event.target.value), 0, 100))} /></label>
                  <label className="setting-row"><span>QB Stack Minimum<small>When GPP candidate mode is active</small></span><input type="number" min="0" max="4" value={qbStackMin} onChange={(event) => setQbStackMin(clamp(Number(event.target.value), 0, 4))} /></label>
                  <label className="setting-row"><span>Bring-Back Minimum<small>Opposing skill players in GPP candidate mode</small></span><input type="number" min="0" max="4" value={bringBackMin} onChange={(event) => setBringBackMin(clamp(Number(event.target.value), 0, 4))} /></label>
                  <label className="setting-switch"><span>Require RB + DST<small>Applied to GPP candidate generation</small></span><input type="checkbox" checked={rbDstStack} onChange={(event) => setRbDstStack(event.target.checked)} /></label>
                </section>

                <section className="settings-card">
                  <div className="section-heading"><strong>PLATFORM STATUS</strong><span>Live connection</span></div>
                  <div className="status-row"><span>BURN1 API</span><strong className={backendStatus === "READY" ? "good" : "bad"}>{backendStatus}</strong></div>
                  <div className="status-row"><span>Active Site</span><strong>{siteName(site)}</strong></div>
                  <div className="status-row"><span>Sport</span><strong>NFL V1</strong></div>
                  <div className="status-row"><span>Slate</span><strong>Week 3 - Main</strong></div>
                  <div className="status-row"><span>Player Pool</span><strong>{playerPoolMeta?.player_count ?? playerPool.length}</strong></div>
                  <div className="status-row"><span>Salary Cap</span><strong>${Number(playerPoolMeta?.salary_cap ?? salaryCap(site)).toLocaleString()}</strong></div>
                  <div className="settings-actions">
                    <button onClick={() => checkBackend(true)}>CHECK API</button>
                    <button onClick={() => loadPlayerPool(site, true)}>REFRESH POOL</button>
                  </div>
                </section>

                <section className="settings-card">
                  <div className="section-heading"><strong>INTERFACE</strong><span>System</span></div>
                  <label className="setting-switch"><span>Completion / UI Sound<small>Never affects optimizer operation</small></span><input type="checkbox" checked={soundEnabled} onChange={(event) => setSoundEnabled(event.target.checked)} /></label>
                  <div className="status-row"><span>Locks</span><strong>{lockedPlayerIds.length}</strong></div>
                  <div className="status-row"><span>Excludes</span><strong>{excludedPlayerIds.length}</strong></div>
                  <div className="status-row"><span>Player Exposure Controls</span><strong>{controlledExposureCount}</strong></div>
                  <div className="settings-actions"><button className="danger-button" onClick={resetOptimizerControls}>RESET OPTIMIZER CONTROLS</button></div>
                </section>
              </div>
            )}
          </section>
        )}

        {toast && <div className={`burn1-toast ${toast.type}`}>{toast.message}</div>}
      </div>
    </div>
  );
}

export default App;
