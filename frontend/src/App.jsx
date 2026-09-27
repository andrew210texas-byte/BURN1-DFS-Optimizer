import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const [site, setSite] = useState("DraftKings");
  const [lineups, setLineups] = useState(20);
  const [gppMix, setGppMix] = useState(50);

  const [backendStatus, setBackendStatus] = useState("CONNECTING");
  const [runStatus, setRunStatus] = useState("ready");
  const [runMessage, setRunMessage] = useState("");
  const [runResult, setRunResult] = useState(null);
  const [runError, setRunError] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);

  const audioContextRef = useRef(null);
  const previousStatusRef = useRef("");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/health")
      .then((response) => {
        if (!response.ok) {
          throw new Error("BURN1 API health check failed");
        }

        return response.json();
      })
      .then((data) => {
        setBackendStatus(
          data.ok && data.status === "ready"
            ? "READY"
            : "ERROR"
        );
      })
      .catch(() => {
        setBackendStatus("OFFLINE");
      });
  }, []);

  function playTone(frequency, duration, type = "sine", volume = 0.04) {
    if (!soundEnabled) {
      return;
    }

    const AudioContext =
      window.AudioContext || window.webkitAudioContext;

    if (!AudioContext) {
      return;
    }

    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }

    const context = audioContextRef.current;
    const oscillator = context.createOscillator();
    const gain = context.createGain();

    oscillator.type = type;
    oscillator.frequency.value = frequency;

    gain.gain.setValueAtTime(volume, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(
      0.001,
      context.currentTime + duration
    );

    oscillator.connect(gain);
    gain.connect(context.destination);

    oscillator.start();
    oscillator.stop(context.currentTime + duration);
  }

  function playStatusSound(status) {
    if (status === previousStatusRef.current) {
      return;
    }

    previousStatusRef.current = status;

    if (status === "validating") {
      playTone(260, 0.12, "sine");
    } else if (status === "stage1_generating") {
      playTone(360, 0.18, "triangle");
    } else if (status === "stage2_optimizing") {
      playTone(470, 0.22, "triangle");
    } else if (status === "complete") {
      playTone(620, 0.18, "sine");
      setTimeout(() => {
        playTone(820, 0.28, "sine");
      }, 120);
    } else if (status === "error") {
      playTone(145, 0.35, "sawtooth", 0.03);
    }
  }

  async function igniteBurn1() {
    if (isRunning) {
      return;
    }

    setIsRunning(true);
    setRunResult(null);
    setRunError("");
    setRunMessage("Starting BURN1...");
    setRunStatus("queued");
    previousStatusRef.current = "";

    playTone(180, 0.12, "sawtooth", 0.035);

    try {
      const startResponse = await fetch(
        "http://127.0.0.1:8000/api/runs",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            site: site === "DraftKings" ? "dk" : "fd",
            lineup_count: lineups,
            candidate_count: 60,
            min_unique_players: 2,
            candidate_gpp_fraction: gppMix / 100,
          }),
        }
      );

      if (!startResponse.ok) {
        throw new Error(
          `BURN1 API returned ${startResponse.status}`
        );
      }

      const startedJob = await startResponse.json();
      const jobId = startedJob.job_id;

      while (true) {
        await new Promise((resolve) =>
          setTimeout(resolve, 750)
        );

        const statusResponse = await fetch(
          `http://127.0.0.1:8000/api/runs/${jobId}`
        );

        if (!statusResponse.ok) {
          throw new Error(
            "Could not retrieve BURN1 run status."
          );
        }

        const job = await statusResponse.json();

        setRunStatus(job.status);
        setRunMessage(job.message || "");
        playStatusSound(job.status);

        if (job.status === "complete") {
          setRunResult(job.result);
          setIsRunning(false);
          break;
        }

        if (job.status === "error") {
          setRunError(
            job.error?.message ||
              job.message ||
              "BURN1 run failed."
          );
          setIsRunning(false);
          break;
        }
      }
    } catch (error) {
      setRunStatus("error");
      setRunError(error.message);
      setRunMessage("BURN1 run failed.");
      setIsRunning(false);
      playStatusSound("error");
    }
  }

  function stageClass(stage) {
    const statusOrder = {
      ready: 0,
      queued: 0,
      validating: 0,
      stage1_generating: 1,
      stage2_optimizing: 2,
      complete: 3,
      error: -1,
    };

    const current = statusOrder[runStatus] ?? 0;

    if (runStatus === "complete") {
      return "stage complete";
    }

    if (current > stage) {
      return "stage complete";
    }

    if (current === stage) {
      return "stage active";
    }

    return "stage";
  }

  const resultLineups =
    runResult?.generated_lineups ?? "—";

  const totalProjection =
    runResult?.total_projection != null
      ? Number(runResult.total_projection).toFixed(2)
      : "—";

  const averageProjection =
    runResult?.total_projection != null &&
    runResult?.generated_lineups
      ? (
          runResult.total_projection /
          runResult.generated_lineups
        ).toFixed(2)
      : "—";

  const solverStatus =
    runResult?.solver_status ?? "—";

  const exposures =
    runResult?.exposures?.slice(0, 5) ?? [];

  const strategy =
    runResult?.strategy_summary ?? {};

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-main">
            BURN<span>1</span>
          </div>
          <div className="brand-sub">DFS OPTIMIZER</div>
        </div>

        <nav className="main-nav">
          <button className="active">OPTIMIZER</button>
          <button>LINEUPS</button>
          <button>EXPOSURES</button>
          <button>STACKS</button>
          <button>PLAYER POOL</button>
          <button>SETTINGS</button>
        </nav>

        <div className="top-controls">
          <button
            className="sound-toggle"
            onClick={() => setSoundEnabled(!soundEnabled)}
          >
            SOUND {soundEnabled ? "ON" : "OFF"}
          </button>

          <select defaultValue="NFL">
            <option>NFL</option>
          </select>

          <select defaultValue="Week 3 - Main">
            <option>Week 3 - Main</option>
          </select>
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <section className="panel">
            <h2>SLATE & SITE</h2>

            <div className="label">Site</div>

            <div className="site-toggle">
              <button
                className={
                  site === "DraftKings"
                    ? "selected"
                    : ""
                }
                onClick={() =>
                  setSite("DraftKings")
                }
                disabled={isRunning}
              >
                DraftKings
              </button>

              <button
                className={
                  site === "FanDuel"
                    ? "selected"
                    : ""
                }
                onClick={() =>
                  setSite("FanDuel")
                }
                disabled={isRunning}
              >
                FanDuel
              </button>
            </div>

            <div className="control-row">
              <span>Sport</span>
              <strong>NFL</strong>
            </div>

            <div className="control-row">
              <span>Slate</span>
              <strong>Week 3 - Main</strong>
            </div>
          </section>

          <section className="panel">
            <h2>PORTFOLIO SETTINGS</h2>

            <div className="slider-head">
              <span>Number of Lineups</span>
              <strong>{lineups}</strong>
            </div>

            <input
              type="range"
              min="1"
              max="50"
              value={lineups}
              disabled={isRunning}
              onChange={(event) =>
                setLineups(
                  Number(event.target.value)
                )
              }
            />

            <div className="slider-head">
              <span>Stage 1 GPP Mix</span>
              <strong>{gppMix}%</strong>
            </div>

            <input
              type="range"
              min="0"
              max="100"
              value={gppMix}
              disabled={isRunning}
              onChange={(event) =>
                setGppMix(
                  Number(event.target.value)
                )
              }
            />
          </section>

          <section className="panel">
            <h2>PLAYER CONTROLS</h2>

            <div className="mini-tabs">
              <button className="selected">
                Locks (0)
              </button>
              <button>Excludes (0)</button>
              <button>Exposures (0)</button>
            </div>

            <input
              className="search"
              placeholder="Search player by name or team..."
            />

            <div className="empty-box">
              No active manual overrides
            </div>
          </section>

          <section className="panel strategy-panel">
            <h2>STRATEGY CONTROLS</h2>

            <div className="strategy-row">
              <span>Candidate GPP Mix</span>
              <strong>{gppMix}%</strong>
            </div>

            <div className="strategy-row">
              <span>Minimum Unique</span>
              <strong>2</strong>
            </div>

            <div className="strategy-row">
              <span>Candidate Pool</span>
              <strong>60</strong>
            </div>
          </section>

          <button
            className={`ignite ${
              isRunning ? "solving" : ""
            }`}
            disabled={isRunning}
            onClick={igniteBurn1}
          >
            <span className="flame">▲</span>
            {isRunning
              ? "BURN1 SOLVING..."
              : "IGNITE TWO-STAGE SOLVER"}
          </button>
        </aside>

        <main className="main-area">
          {runError && (
            <div className="run-error">
              {runError}
            </div>
          )}

          <section className="solver-panel">
            <div className="solver-title">
              BURN1 TWO-STAGE PORTFOLIO ENGINE
            </div>

            <div className="pipeline">
              <div className={stageClass(0)}>
                <div className="stage-node">
                  {runStatus === "validating"
                    ? "•"
                    : "✓"}
                </div>
                <strong>VALIDATE</strong>
                <span>Player pool ready</span>
              </div>

              <div className="pipeline-line"></div>

              <div className={stageClass(1)}>
                <div className="stage-node">1</div>
                <strong>STAGE 1</strong>
                <span>Generate candidates</span>
              </div>

              <div className="pipeline-line"></div>

              <div className={stageClass(2)}>
                <div className="stage-node">2</div>
                <strong>STAGE 2</strong>
                <span>Select portfolio</span>
              </div>

              <div className="pipeline-line"></div>

              <div className={stageClass(3)}>
                <div className="stage-node">3</div>
                <strong>COMPLETE</strong>
                <span>Portfolio output</span>
              </div>
            </div>

            <div className="arena">
              <div className="field-visual">
                <div className="node cyan n1">QB</div>
                <div className="node cyan n2">WR</div>
                <div className="node cyan n3">TE</div>
                <div className="node orange n4">RB</div>
                <div className="node orange n5">WR</div>

                <div className="arc a1"></div>
                <div className="arc a2"></div>
                <div className="arc a3"></div>
              </div>

              <div className="telemetry">
                <div>
                  <span>Site</span>
                  <strong>{site}</strong>
                </div>

                <div>
                  <span>Final Lineups</span>
                  <strong>{lineups}</strong>
                </div>

                <div>
                  <span>Candidate GPP Mix</span>
                  <strong>{gppMix}%</strong>
                </div>

                <div>
                  <span>Solver</span>
                  <strong>CP-SAT</strong>
                </div>

                <div>
                  <span>Backend</span>
                  <strong>
                    {backendStatus}
                  </strong>
                </div>

                <div>
                  <span>Status</span>
                  <strong className="orange-text">
                    {runStatus.toUpperCase()}
                  </strong>
                </div>
              </div>
            </div>

            {runMessage && (
              <div className="run-message">
                {runMessage}
              </div>
            )}
          </section>

          <section className="results-grid">
            <div className="panel results-card">
              <h2>PORTFOLIO SUMMARY</h2>

              <div className="metric-grid">
                <div>
                  <strong>{resultLineups}</strong>
                  <span>Lineups</span>
                </div>

                <div>
                  <strong>{totalProjection}</strong>
                  <span>Total Projection</span>
                </div>

                <div>
                  <strong>{averageProjection}</strong>
                  <span>Avg Projection</span>
                </div>

                <div>
                  <strong className="green">
                    {solverStatus}
                  </strong>
                  <span>Solver Status</span>
                </div>
              </div>
            </div>

            <div className="panel results-card">
              <h2>LINEUP STRATEGY MIX</h2>

              <div className="strategy-summary">
                <div>
                  <span>Stacked</span>
                  <strong>
                    {strategy.stacked?.count ?? "—"} /{" "}
                    {resultLineups}
                  </strong>
                </div>

                <div>
                  <span>Unstacked</span>
                  <strong>
                    {strategy.unstacked?.count ?? "—"} /{" "}
                    {resultLineups}
                  </strong>
                </div>

                <div>
                  <span>QB Stack 1+</span>
                  <strong>
                    {strategy.qb_stack_1_plus?.count ??
                      "—"}{" "}
                    / {resultLineups}
                  </strong>
                </div>

                <div>
                  <span>QB Stack 2+</span>
                  <strong>
                    {strategy.qb_stack_2_plus?.count ??
                      "—"}{" "}
                    / {resultLineups}
                  </strong>
                </div>
              </div>
            </div>

            <div className="panel results-card">
              <h2>TOP PLAYER EXPOSURES</h2>

              <div className="exposure-table">
                {exposures.length === 0 ? (
                  <div className="empty-box">
                    Ignite BURN1 to populate real exposures.
                  </div>
                ) : (
                  exposures.map((player) => (
                    <div
                      className="exposure-row"
                      key={player.player_id}
                    >
                      <span className="player-name">
                        {player.name}
                      </span>
                      <span>{player.position}</span>
                      <span>{player.team}</span>
                      <strong>
                        {Math.round(
                          player.exposure * 100
                        )}
                        %
                      </strong>
                    </div>
                  ))
                )}
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

export default App;
