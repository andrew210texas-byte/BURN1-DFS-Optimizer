import { useState } from "react";
import "./App.css";

const mockExposures = [
  ["Patrick Mahomes", "QB", "KC", "60%"],
  ["Christian McCaffrey", "RB", "SF", "55%"],
  ["Derrick Henry", "RB", "BAL", "50%"],
  ["Adonai Mitchell", "WR", "NYJ", "45%"],
  ["George Kittle", "TE", "SF", "35%"],
];

function App() {
  const [site, setSite] = useState("DraftKings");
  const [lineups, setLineups] = useState(20);
  const [gppMix, setGppMix] = useState(50);

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
                className={site === "DraftKings" ? "selected" : ""}
                onClick={() => setSite("DraftKings")}
              >
                DraftKings
              </button>

              <button
                className={site === "FanDuel" ? "selected" : ""}
                onClick={() => setSite("FanDuel")}
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
              max="150"
              value={lineups}
              onChange={(e) => setLineups(Number(e.target.value))}
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
              onChange={(e) => setGppMix(Number(e.target.value))}
            />
          </section>

          <section className="panel">
            <h2>PLAYER CONTROLS</h2>

            <div className="mini-tabs">
              <button className="selected">Locks (0)</button>
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
              <span>Stacked</span>
              <strong>50%</strong>
            </div>

            <div className="strategy-row">
              <span>Unstacked</span>
              <strong>50%</strong>
            </div>

            <div className="strategy-row">
              <span>QB Stack (1+)</span>
              <strong>50%</strong>
            </div>

            <div className="strategy-row">
              <span>Bring Back (1+)</span>
              <strong>0%</strong>
            </div>

            <div className="strategy-row">
              <span>RB + DST</span>
              <strong>0%</strong>
            </div>
          </section>

          <button className="ignite">
            <span className="flame">▲</span>
            IGNITE TWO-STAGE SOLVER
          </button>
        </aside>

        <main className="main-area">
          <section className="solver-panel">
            <div className="solver-title">
              BURN1 TWO-STAGE PORTFOLIO ENGINE
            </div>

            <div className="pipeline">
              <div className="stage complete">
                <div className="stage-node">✓</div>
                <strong>VALIDATE</strong>
                <span>Player pool ready</span>
              </div>

              <div className="pipeline-line"></div>

              <div className="stage active">
                <div className="stage-node">1</div>
                <strong>STAGE 1</strong>
                <span>Generate candidates</span>
              </div>

              <div className="pipeline-line"></div>

              <div className="stage">
                <div className="stage-node">2</div>
                <strong>STAGE 2</strong>
                <span>Select portfolio</span>
              </div>

              <div className="pipeline-line"></div>

              <div className="stage">
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
                  <span>Status</span>
                  <strong className="orange-text">READY</strong>
                </div>
              </div>
            </div>
          </section>

          <section className="results-grid">
            <div className="panel results-card">
              <h2>PORTFOLIO SUMMARY</h2>

              <div className="metric-grid">
                <div>
                  <strong>{lineups}</strong>
                  <span>Lineups</span>
                </div>

                <div>
                  <strong>3065.72</strong>
                  <span>Total Projection</span>
                </div>

                <div>
                  <strong>153.29</strong>
                  <span>Avg Projection</span>
                </div>

                <div>
                  <strong className="green">OPTIMAL</strong>
                  <span>Solver Status</span>
                </div>
              </div>

              <div className="chart-placeholder">
                <div style={{ height: "65%" }}>QB</div>
                <div style={{ height: "75%" }}>RB</div>
                <div style={{ height: "90%" }}>WR</div>
                <div style={{ height: "55%" }}>TE</div>
                <div style={{ height: "70%" }}>FLEX</div>
                <div style={{ height: "45%" }}>DST</div>
              </div>
            </div>

            <div className="panel results-card">
              <h2>LINEUP STRATEGY MIX</h2>

              <div className="strategy-summary">
                <div>
                  <span>Stacked</span>
                  <strong>10 / 20</strong>
                </div>
                <div>
                  <span>Unstacked</span>
                  <strong>10 / 20</strong>
                </div>
                <div>
                  <span>QB Stack 1+</span>
                  <strong>10 / 20</strong>
                </div>
                <div>
                  <span>QB Stack 2+</span>
                  <strong>1 / 20</strong>
                </div>
              </div>
            </div>

            <div className="panel results-card">
              <h2>TOP PLAYER EXPOSURES</h2>

              <div className="exposure-table">
                {mockExposures.map(([name, pos, team, exposure]) => (
                  <div className="exposure-row" key={name}>
                    <span className="player-name">{name}</span>
                    <span>{pos}</span>
                    <span>{team}</span>
                    <strong>{exposure}</strong>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

export default App;
