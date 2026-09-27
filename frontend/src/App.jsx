import { useMemo, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [site, setSite] = useState("dk");
  const [lineupCount, setLineupCount] = useState(20);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("READY");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [view, setView] = useState("optimizer");

  const siteName =
    site === "dk" ? "DraftKings" : "FanDuel";

  const topExposures = useMemo(
    () => result?.exposures?.slice(0, 10) ?? [],
    [result]
  );

  async function ignite() {
    if (running) return;

    setRunning(true);
    setResult(null);
    setError("");
    setMessage("Starting BURN1...");
    setStatus("QUEUED");

    try {
      const startResponse = await fetch(
        `${API}/api/runs`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            site,
            lineup_count: lineupCount,
            candidate_count: 60,
            min_unique_players: 2,
            candidate_gpp_fraction: 0.50,

            locked_player_ids: [],
            excluded_player_ids: [],

            min_player_exposures: {},
            max_player_exposures: {},

            min_strategy_exposures: {},
            max_strategy_exposures: {},
          }),
        }
      );

      if (!startResponse.ok) {
        throw new Error(
          `BURN1 returned HTTP ${startResponse.status}`
        );
      }

      const started = await startResponse.json();

      if (!started.job_id) {
        throw new Error("BURN1 did not return a job ID.");
      }

      while (true) {
        await new Promise((resolve) =>
          setTimeout(resolve, 750)
        );

        const response = await fetch(
          `${API}/api/runs/${started.job_id}`
        );

        if (!response.ok) {
          throw new Error(
            "Could not retrieve BURN1 run status."
          );
        }

        const job = await response.json();

        setStatus(
          String(job.status || "running")
            .replaceAll("_", " ")
            .toUpperCase()
        );

        setMessage(job.message || "");

        if (job.status === "complete") {
          setResult(job.result);
          setRunning(false);

          sessionStorage.setItem(
            "burn1LastRun",
            JSON.stringify(job.result)
          );

          return;
        }

        if (job.status === "error") {
          throw new Error(
            job.error?.message ||
            job.message ||
            "BURN1 run failed."
          );
        }
      }
    } catch (err) {
      setStatus("ERROR");
      setError(err.message);
      setRunning(false);
    }
  }

  function changeSite(nextSite) {
    if (running) return;

    setSite(nextSite);
    setResult(null);
    setStatus("READY");
    setMessage("");
    setError("");
  }

  function exportCsv() {
    if (!result?.lineups?.length) return;

    const rows = [[
      "lineup",
      "slot",
      "player_id",
      "name",
      "position",
      "team",
      "opponent",
      "salary",
      "projection",
    ]];

    for (const lineup of result.lineups) {
      for (const player of lineup.players) {
        rows.push([
          lineup.lineup_number,
          player.roster_slot,
          player.player_id,
          player.name,
          player.position,
          player.team,
          player.opponent,
          player.salary,
          player.projection,
        ]);
      }
    }

    const csv = rows
      .map((row) =>
        row.map((value) =>
          `"${String(value ?? "").replaceAll('"', '""')}"`
        ).join(",")
      )
      .join("\n");

    const blob = new Blob(
      [csv],
      { type: "text/csv;charset=utf-8" }
    );

    const url = URL.createObjectURL(blob);

    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download =
      `burn1_${site}_${lineupCount}_lineups.csv`;

    anchor.click();

    URL.revokeObjectURL(url);
  }

  return (
    <div className="app">

      <div className="master">

        {/* This IS the exact uploaded UI. */}
        <img
          className="master-image"
          src="/BURN1_UI.png"
          alt="BURN1 DFS Optimizer"
        />

        {/* TOP NAV */}
        <button
          className="hit nav optimizer"
          onClick={() => setView("optimizer")}
          aria-label="Optimizer"
        />

        <button
          className="hit nav lineups"
          onClick={() => setView("lineups")}
          aria-label="Lineups"
        />

        <button
          className="hit nav exposures"
          onClick={() => setView("exposures")}
          aria-label="Exposures"
        />

        <button
          className="hit nav stacks"
          onClick={() => setView("stacks")}
          aria-label="Stacks"
        />

        {/* SITE */}
        <button
          className="hit dk"
          disabled={running}
          onClick={() => changeSite("dk")}
          aria-label="DraftKings"
        />

        <button
          className="hit fd"
          disabled={running}
          onClick={() => changeSite("fd")}
          aria-label="FanDuel"
        />

        <div
          className={
            site === "dk"
              ? "site-overlay site-dk"
              : "site-overlay site-fd"
          }
        >
          {siteName}
        </div>

        {/* LINEUP COUNT */}
        <input
          className="lineup-slider"
          type="range"
          min="1"
          max="50"
          value={lineupCount}
          disabled={running}
          onChange={(event) =>
            setLineupCount(
              Number(event.target.value)
            )
          }
        />

        <div className="lineup-value">
          {lineupCount}
        </div>

        {/* IGNITE */}
        <button
          className="hit ignite"
          disabled={running}
          onClick={ignite}
          aria-label="Ignite BURN1"
        >
          {running && (
            <span>BURN1 SOLVING...</span>
          )}
        </button>

        {/* Replace sample telemetry with REAL telemetry */}
        {(running || result) && (
          <section className="telemetry">
            <div>
              <span>Candidates Generated</span>
              <strong>
                {result
                  ? `${result.generated_candidates}`
                  : "—"}
              </strong>
            </div>

            <div>
              <span>Valid Lineups</span>
              <strong>
                {result
                  ? `${result.generated_candidates}`
                  : "—"}
              </strong>
            </div>

            <div>
              <span>Portfolio Selection</span>
              <strong>
                {running
                  ? "In Progress..."
                  : "Complete"}
              </strong>
            </div>

            <div>
              <span>Solver</span>
              <strong>CP-SAT (OR-Tools)</strong>
            </div>

            <div>
              <span>Status</span>
              <strong className="orange">
                {result?.solver_status ?? status}
              </strong>
            </div>

            <div>
              <span>Final</span>
              <strong>
                {result
                  ? `${result.generated_lineups}/${result.requested_lineups}`
                  : `—/${lineupCount}`}
              </strong>
            </div>
          </section>
        )}

        {/* REAL SUMMARY */}
        {result && (
          <section className="summary">
            <div>
              <strong>
                {result.generated_lineups}
              </strong>
              <span>Lineups</span>
            </div>

            <div>
              <strong>
                {Number(
                  result.total_projection
                ).toFixed(2)}
              </strong>
              <span>Total Projection</span>
            </div>

            <div>
              <strong>
                {(
                  Number(result.total_projection) /
                  Number(result.generated_lineups)
                ).toFixed(2)}
              </strong>
              <span>Avg Projection</span>
            </div>

            <div>
              <strong className="orange">
                {result.solver_status}
              </strong>
              <span>Solver Status</span>
            </div>
          </section>
        )}

        {/* REAL TOP EXPOSURES */}
        {result && (
          <section className="top-exposures">
            {topExposures.map((player) => {
              const exposure =
                Math.round(
                  Number(player.exposure || 0) * 100
                );

              return (
                <div
                  className="exposure-row"
                  key={player.player_id}
                >
                  <span>{player.name}</span>
                  <span>{player.position}</span>
                  <span>{player.team}</span>

                  <div className="bar-track">
                    <i
                      style={{
                        width: `${exposure}%`,
                      }}
                    />
                  </div>

                  <strong>
                    {exposure}%
                  </strong>
                </div>
              );
            })}
          </section>
        )}

        {error && (
          <div className="error-box">
            {error}
          </div>
        )}

        {running && message && (
          <div className="run-box">
            {message}
          </div>
        )}

        {/* LINEUPS VIEW */}
        {view === "lineups" && (
          <section className="overlay-view">

            <header>
              <div>
                <small>BURN1 DFS</small>
                <h1>FINAL LINEUPS</h1>
              </div>

              <div className="overlay-actions">
                <button
                  disabled={!result}
                  onClick={exportCsv}
                >
                  EXPORT CSV
                </button>

                <button
                  onClick={() =>
                    setView("optimizer")
                  }
                >
                  RETURN
                </button>
              </div>
            </header>

            {!result ? (
              <div className="empty">
                Run BURN1 first.
              </div>
            ) : (
              <div className="lineup-grid">
                {result.lineups.map((lineup) => (
                  <article
                    key={lineup.lineup_number}
                  >
                    <header>
                      <strong>
                        LINEUP {lineup.lineup_number}
                      </strong>

                      <span>
                        $
                        {Number(
                          lineup.total_salary
                        ).toLocaleString()}
                        {" • "}
                        {Number(
                          lineup.total_projection
                        ).toFixed(2)}
                      </span>
                    </header>

                    {lineup.players.map(
                      (player) => (
                        <div
                          className="player-row"
                          key={
                            `${lineup.lineup_number}-` +
                            `${player.player_id}-` +
                            `${player.roster_slot}`
                          }
                        >
                          <strong>
                            {player.roster_slot}
                          </strong>

                          <span>
                            {player.name}
                          </span>

                          <span>
                            {player.team} vs{" "}
                            {player.opponent}
                          </span>

                          <span>
                            $
                            {Number(
                              player.salary
                            ).toLocaleString()}
                          </span>

                          <strong>
                            {Number(
                              player.projection
                            ).toFixed(2)}
                          </strong>
                        </div>
                      )
                    )}
                  </article>
                ))}
              </div>
            )}

          </section>
        )}

        {/* EXPOSURES VIEW */}
        {view === "exposures" && (
          <section className="overlay-view">

            <header>
              <div>
                <small>BURN1 DFS</small>
                <h1>PLAYER EXPOSURES</h1>
              </div>

              <button
                onClick={() =>
                  setView("optimizer")
                }
              >
                RETURN
              </button>
            </header>

            {!result ? (
              <div className="empty">
                Run BURN1 first.
              </div>
            ) : (
              <div className="exposure-list">
                {result.exposures.map(
                  (player) => {
                    const exposure =
                      Math.round(
                        Number(
                          player.exposure || 0
                        ) * 100
                      );

                    return (
                      <div
                        key={player.player_id}
                      >
                        <strong>
                          {player.name}
                        </strong>

                        <span>
                          {player.position}
                        </span>

                        <span>
                          {player.team}
                        </span>

                        <div className="wide-track">
                          <i
                            style={{
                              width:
                                `${exposure}%`,
                            }}
                          />
                        </div>

                        <strong>
                          {exposure}%
                        </strong>
                      </div>
                    );
                  }
                )}
              </div>
            )}

          </section>
        )}

        {/* STACKS VIEW */}
        {view === "stacks" && (
          <section className="overlay-view">

            <header>
              <div>
                <small>BURN1 DFS</small>
                <h1>STACKS</h1>
              </div>

              <button
                onClick={() =>
                  setView("optimizer")
                }
              >
                RETURN
              </button>
            </header>

            {!result ? (
              <div className="empty">
                Run BURN1 first.
              </div>
            ) : (
              <div className="stack-list">
                {Object.entries(
                  result.strategy_summary ?? {}
                ).map(([name, metrics]) => (
                  <div key={name}>
                    <span>
                      {name
                        .replaceAll("_", " ")
                        .toUpperCase()}
                    </span>

                    <strong>
                      {metrics.count}
                    </strong>

                    <strong>
                      {Math.round(
                        Number(
                          metrics.exposure || 0
                        ) * 100
                      )}
                      %
                    </strong>
                  </div>
                ))}
              </div>
            )}

          </section>
        )}

      </div>
    </div>
  );
}

export default App;