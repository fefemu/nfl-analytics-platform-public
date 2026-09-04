"""Central visual tokens and CSS for the premium Streamlit shell."""

APP_CSS = """
<style>
:root {
  --canvas: #06111f; --sidebar: #071525; --card: #0b1b2e;
  --card-2: #0d2138; --border: #1d3551; --text: #f4f7fb;
  --muted: #96a6ba; --blue: #42a5ff; --green: #36e39a;
  --amber: #f4c84a; --red: #ff5964;
}
.stApp { background: radial-gradient(circle at 70% -10%, #102944 0, var(--canvas) 42%); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stAppDeployButton"], #MainMenu { display:none !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #071525 0%, #06111f 100%); border-right: 1px solid var(--border); min-width:285px; max-width:285px; }
[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
@media (min-width: 761px) {
  [data-testid="stSidebar"] {
    display:block !important; visibility:visible !important;
    position:relative !important; left:0 !important;
    width:285px !important; min-width:285px !important; max-width:285px !important;
    margin-left:0 !important; transform:none !important;
    flex:0 0 285px !important; overflow:visible !important;
  }
  [data-testid="stSidebar"] > div { display:block !important; width:285px !important; }
  [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"] { display:none !important; }
}
.block-container { max-width: 1540px; padding: 1.4rem 2rem 3rem; }
h1, h2, h3, p, label { color: var(--text); }
.nap-brand { display:flex; align-items:center; gap:.7rem; font-weight:800; font-size:1.2rem; padding:.5rem 0 1.2rem; }
.nap-mark { color:var(--green); font-size:1.45rem; }
.nap-eyebrow { color:var(--muted); letter-spacing:.12em; text-transform:uppercase; font-size:.72rem; font-weight:700; }
.nap-page-header { display:flex; justify-content:space-between; gap:2rem; align-items:flex-start; }
.nap-page-header > div:first-child { min-width:0; }
.nap-refresh { color:var(--muted); font-size:.76rem; white-space:nowrap; padding-top:.45rem; }
.nap-title { font-size:2.15rem; font-weight:800; letter-spacing:-.04em; margin:.1rem 0 .25rem; }
.nap-subtitle { color:var(--muted); margin-bottom:1.25rem; }
.nap-card { background:linear-gradient(145deg, rgba(13,33,56,.98), rgba(8,23,40,.98)); border:1px solid var(--border); border-radius:10px; padding:1rem 1.1rem; box-shadow:0 16px 40px rgba(0,0,0,.16); height:100%; }
.nap-metric-label { display:flex; align-items:center; justify-content:space-between; gap:.35rem; color:var(--muted); font-size:.78rem; }
.nap-metric-value { color:var(--text); font-size:1.55rem; font-weight:800; margin-top:.15rem; }
.nap-metric-value.green { color:var(--green); } .nap-metric-value.blue { color:var(--blue); } .nap-metric-value.amber { color:var(--amber); }
.nap-pill { display:inline-flex; align-items:center; padding:.24rem .55rem; border-radius:5px; font-size:.7rem; font-weight:800; letter-spacing:.04em; }
.nap-pill.ready { color:#87f4c4; background:rgba(28,130,88,.38); } .nap-pill.warning { color:#ffe589; background:rgba(139,107,8,.42); }
.nap-panel-title { font-weight:750; font-size:1rem; margin-bottom:.25rem; }
.nap-heading-with-help { display:flex; align-items:center; gap:.35rem; margin:0 0 .65rem; }
.nap-heading-with-help h3 { margin:0; }
.nap-muted { color:var(--muted); font-size:.86rem; }
.nap-divider { border-top:1px solid var(--border); margin:1rem 0; }
.nap-empty { padding:2.4rem 1rem; text-align:center; color:var(--muted); }
.nap-matchup-card { position:relative; z-index:1; overflow:visible; margin-bottom:1rem; }
.nap-matchup-card:hover,.nap-matchup-card:focus-within { z-index:10010; }
[data-testid="stColumn"]:has(.nap-matchup-card:hover),
[data-testid="stColumn"]:has(.nap-matchup-card:focus-within) { z-index:10010; }
.nap-matchup-link { color:inherit; display:block; text-decoration:none !important; }
.nap-matchup-link .nap-matchup-card { transition:border-color .16s ease, transform .16s ease, box-shadow .16s ease; }
.nap-matchup-link:hover .nap-matchup-card { border-color:rgba(66,165,255,.62); transform:translateY(-2px); box-shadow:0 18px 44px rgba(0,0,0,.24); }
.nap-matchup-action { color:var(--blue); font-size:.72rem; font-weight:750; margin-top:.65rem; text-align:right; }
.nap-team-identity { position:relative; display:inline-flex; flex:0 0 auto; align-items:center; justify-content:center; overflow:hidden; border:2px solid color-mix(in srgb,var(--nap-team-accent,#64748b) 72%,#dce5ef); border-radius:50%; background:linear-gradient(145deg,#f8fafc,#dce5ef); box-shadow:inset 0 0 0 1px rgba(255,255,255,.7),0 7px 18px rgba(0,0,0,.28); }
.nap-team-fallback { display:none; position:absolute; inset:0; align-items:center; justify-content:center; color:#fff; background:#132238; font-weight:850; }
.nap-team-logo { position:absolute; inset:8%; display:block; width:84%; height:84%; object-fit:contain; object-position:center; filter:drop-shadow(0 3px 3px rgba(0,0,0,.24)); }
.nap-matchup-meta { color:var(--muted); font-size:.7rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }
.nap-matchup-line { display:grid; grid-template-columns:1fr auto 1fr; gap:.65rem; align-items:center; margin:1rem 0; }
.nap-matchup-line > div:not(.nap-at) { display:grid; grid-template-columns:auto 1fr; gap:.25rem .65rem; align-items:center; }
.nap-matchup-line span { color:var(--blue); font-size:1.15rem; font-weight:800; grid-column:2; }
.nap-matchup-trends,.nap-game-trends { display:flex; justify-content:space-between; gap:.65rem; margin:.3rem 0 .75rem; }
.nap-matchup-trends > span:last-child,.nap-game-trends > span:last-child { text-align:right; }
.nap-probability-trend { display:inline-flex; align-items:center; gap:.12rem; font-size:.72rem; font-weight:750; }
.nap-probability-trend.increase { color:var(--green); }
.nap-probability-trend.decrease { color:var(--red); }
.nap-probability-trend.neutral,.nap-probability-trend.new { color:var(--muted); }
.nap-probability-trend .nap-tooltip-trigger { color:inherit; }
.nap-at { color:var(--muted); font-weight:800; }
.nap-scoreline { display:flex; justify-content:space-between; gap:.7rem; color:var(--muted); border-top:1px solid var(--border); padding-top:.75rem; font-size:.8rem; }
.nap-scoreline b { color:var(--text); }
.nap-candidate-card { position:relative; z-index:1; overflow:visible; margin-bottom:1rem; }
.nap-candidate-card:hover,.nap-candidate-card:focus-within { z-index:10010; }
.nap-candidate-teams { display:flex; align-items:center; justify-content:space-between; gap:.45rem; }
.nap-candidate-market { font-size:1rem; font-weight:800; margin:.9rem 0; }
.nap-candidate-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.45rem; color:var(--muted); font-size:.76rem; }
.nap-candidate-grid > span { background:rgba(255,255,255,.025); padding:.5rem; border-radius:6px; }
.nap-candidate-grid > span > b { display:block; color:var(--text); font-size:.95rem; margin-top:.15rem; }
.nap-candidate-grid .nap-positive { color:var(--green); }
.nap-candidate-footer { display:flex; justify-content:space-between; align-items:center; border-top:1px solid var(--border); padding-top:.75rem; margin-top:.8rem; color:var(--muted); font-size:.76rem; }
.nap-attribution { color:var(--muted); font-size:.72rem; line-height:1.45; }
.nap-attribution a { color:var(--blue); text-decoration:none; }
.nap-game-hero { padding:1.35rem 1.5rem; margin-bottom:1rem; }
.nap-game-teams { display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:1rem; }
.nap-game-team { display:flex; align-items:center; gap:.9rem; font-size:1.45rem; font-weight:850; }
.nap-game-team.home { justify-content:flex-end; text-align:right; }
.nap-game-context { color:var(--muted); text-align:center; font-size:.74rem; line-height:1.55; }
.nap-probability-labels { display:flex; justify-content:space-between; margin:.95rem 0 .35rem; color:var(--muted); font-size:.82rem; }
.nap-probability-labels b { color:var(--text); font-size:1.05rem; }
.nap-probability-bar { display:flex; height:12px; overflow:hidden; border-radius:99px; background:#15283d; }
.nap-probability-bar .away { background:linear-gradient(90deg,#2589e8,#47b3ff); }
.nap-probability-bar .home { background:linear-gradient(90deg,#f0646d,#e53f51); }
.nap-prediction-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.7rem; margin:1rem 0; }
.nap-prediction-tile { background:rgba(255,255,255,.025); border:1px solid var(--border); border-radius:8px; padding:.8rem; text-align:center; }
.nap-prediction-tile span { display:block; color:var(--muted); font-size:.72rem; }
.nap-prediction-tile b { display:block; color:var(--text); font-size:1.3rem; margin-top:.2rem; }
.nap-narrative { font-size:.94rem; line-height:1.65; color:#dce6f2; }
.nap-market-row { display:grid; grid-template-columns:1.2fr .8fr .8fr .8fr; gap:.5rem; padding:.65rem 0; border-bottom:1px solid var(--border); font-size:.78rem; }
.nap-market-row b { color:var(--text); }
.nap-market-card .nap-candidate-market { margin:.65rem 0 .25rem; }
.nap-market-edge { margin:.8rem 0 .15rem; font-size:2rem; font-weight:850; letter-spacing:-.04em; line-height:1; }
.nap-market-edge.positive,.nap-market-card .nap-positive { color:var(--green); }
.nap-market-edge.negative,.nap-market-card .nap-negative { color:var(--red); }
.nap-market-edge.neutral,.nap-market-card .nap-neutral { color:var(--muted); }
.nap-market-status { min-height:1.3rem; font-size:.78rem; font-weight:750; }
.nap-market-probabilities { display:flex; justify-content:space-between; gap:.65rem; margin-top:.9rem; color:var(--muted); font-size:.75rem; }
.nap-market-probabilities b { display:block; margin-top:.1rem; color:var(--text); font-size:.96rem; }
.nap-market-price { margin-top:.8rem; padding-top:.7rem; border-top:1px solid var(--border); color:var(--muted); font-size:.72rem; }
.nap-market-price b { color:var(--text); font-size:.82rem; }
.nap-market-book { margin-left:.35rem; color:var(--muted); }
.nap-sim-leaders { display:grid; grid-template-columns:repeat(5,1fr); gap:.65rem; margin:1rem 0; }
.nap-sim-leader { display:flex; align-items:center; gap:.65rem; min-width:0; }
.nap-sim-rank { color:var(--muted); font-size:.75rem; }
.nap-sim-leader b { display:block; font-size:1rem; }
.nap-sim-leader strong { color:var(--green); font-size:1.1rem; }
.nap-team-sim-header { display:flex; align-items:center; gap:1rem; margin-bottom:.8rem; }
.nap-team-sim-header h3 { margin:0; }
.nap-team-sim-header span { color:var(--muted); font-size:.8rem; }
.nap-roster-hero { display:flex; align-items:center; gap:1rem; margin-bottom:1rem; padding:.8rem 1rem; min-height:92px; }
.nap-roster-identity { flex:1; min-width:0; }
.nap-roster-hero h2 { margin:0 0 .1rem; font-size:1.35rem; }
.nap-roster-identity span,.nap-roster-identity small { display:block; color:var(--muted); font-size:.75rem; }
.nap-roster-kpi { border-left:1px solid var(--border); padding:.2rem .5rem .2rem 1rem; min-width:120px; }
.nap-roster-kpi span,.nap-roster-kpi small { color:var(--muted); font-size:.68rem; }
.nap-roster-kpi strong { display:block; color:var(--green); font-size:1.35rem; }
.nap-info { color:var(--muted); cursor:help; font-size:.72rem; font-weight:700; }
.nap-tooltip { position:relative; display:inline-flex; align-items:center; vertical-align:middle; }
.nap-tooltip-trigger { display:inline-flex; align-items:center; justify-content:center; width:1.55rem; min-width:1.55rem; height:1.55rem; margin:-.25rem 0; padding:0; border:0; border-radius:50%; color:var(--muted); background:transparent; font:inherit; font-size:.76rem; font-weight:800; line-height:1; cursor:help; }
.nap-tooltip-trigger:hover,.nap-tooltip-trigger:focus-visible { color:var(--blue); background:rgba(66,165,255,.12); outline:2px solid rgba(66,165,255,.55); outline-offset:1px; }
.nap-tooltip-content { position:absolute; z-index:10020; left:0; bottom:calc(100% + .5rem); display:block; width:max-content; min-width:min(220px,calc(100vw - 2rem)); max-width:min(350px,calc(100vw - 2rem)); padding:.72rem .8rem; border:1px solid var(--border); border-radius:8px; color:var(--text); background:#10243a; box-shadow:0 12px 32px rgba(0,0,0,.42); font:500 .74rem/1.55 "Segoe UI",Arial,sans-serif; font-stretch:normal; letter-spacing:normal !important; word-spacing:normal !important; text-align:left; text-transform:none; white-space:normal; overflow-wrap:break-word; word-break:normal; hyphens:auto; opacity:0; visibility:hidden; pointer-events:none; transform:translateY(4px); transition:opacity .12s ease,transform .12s ease,visibility .12s ease; }
.nap-tooltip-right .nap-tooltip-content { right:0; left:auto; }
.nap-tooltip:hover .nap-tooltip-content,.nap-tooltip:focus-within .nap-tooltip-content { opacity:1; visibility:visible; transform:translateY(0); }
.nap-heading-with-help,.nap-card:has(.nap-tooltip) { position:relative; overflow:visible; }
.nap-card:has(.nap-tooltip):hover,.nap-card:has(.nap-tooltip):focus-within { z-index:10010; }
.nap-unit-strip { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem; margin:.25rem 0 1rem; }
.nap-unit-rank { display:flex; align-items:baseline; gap:.35rem; padding:.65rem .8rem; border:1px solid var(--border); border-radius:8px; background:rgba(11,27,46,.72); }
.nap-unit-rank span { flex:1; color:var(--muted); font-size:.72rem; }
.nap-unit-rank strong { color:var(--green); font-size:1.05rem; }
.nap-unit-rank small { color:var(--muted); font-size:.62rem; }
.nap-formation { position:relative; isolation:isolate; display:grid; grid-template-columns:repeat(7,minmax(92px,1fr)); grid-template-rows:repeat(3,minmax(112px,auto)); gap:1.15rem .65rem; align-items:center; padding:2rem 1.2rem; overflow-x:auto; border:1px solid rgba(78,204,134,.25); border-radius:12px; background:linear-gradient(180deg,rgba(10,47,49,.56),rgba(6,27,37,.88)); box-shadow:inset 0 0 55px rgba(0,0,0,.22); }
.nap-formation.offense { grid-template-areas:"wr1 lt lg center rg rt te" ". . . qb . . ." ". slot . rb . wr2 ."; }
.nap-formation.defense-34 { grid-template-areas:"cb1 . dl1 dl2 dl3 . cb2" ". edge1 lb1 . lb2 edge2 ." ". . s1 . s2 . ."; }
.nap-formation.defense-43 { grid-template-areas:"cb1 . dl1 dl2 dl3 dl4 cb2" ". . lb1 lb2 lb3 . ." ". . s1 . s2 . ."; }
.nap-field-lines { position:absolute; z-index:-1; inset:0; opacity:.5; background:repeating-linear-gradient(90deg,transparent 0,transparent calc(10% - 1px),rgba(213,241,231,.12) 10%); }
.nap-field-lines:after { content:""; position:absolute; left:50%; top:0; bottom:0; border-left:1px dashed rgba(213,241,231,.15); }
.nap-formation-player { min-width:0; min-height:92px; display:flex; align-items:center; justify-content:center; gap:.35rem; padding:.55rem .4rem; text-align:center; background:rgba(7,20,35,.94); border:1px solid rgba(255,255,255,.13); border-radius:9px; box-shadow:0 10px 24px rgba(0,0,0,.28); }
.nap-formation-player img { width:42px; height:42px; flex:0 0 42px; object-fit:cover; object-position:top center; border-radius:50%; background:#15283d; }
.nap-formation-player.nap-player-featured { border-color:rgba(54,227,154,.55); box-shadow:0 0 0 1px rgba(54,227,154,.14),0 12px 28px rgba(0,0,0,.3); }
.nap-player-copy { min-width:0; }
.nap-player-position { display:block; color:var(--green); font-size:.64rem; font-weight:850; letter-spacing:.08em; }
.nap-player-copy strong { display:block; overflow:hidden; color:var(--text); font-size:.74rem; line-height:1.15; text-overflow:ellipsis; }
.nap-player-copy small { color:var(--muted); font-size:.64rem; }
.nap-player-injury { display:inline-block; margin-top:.4rem; padding:.15rem .35rem; border-radius:4px; color:#ffe589; background:rgba(139,107,8,.42); font-size:.65rem; font-weight:750; }
.nap-depth-groups { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem; }
.nap-depth-group { padding:.75rem; border:1px solid var(--border); border-radius:9px; background:rgba(11,27,46,.78); }
.nap-depth-group h4 { color:var(--green); margin:0 0 .5rem; }
.nap-depth-group h4 a { display:none !important; }
.nap-depth-player { display:grid; grid-template-columns:22px 1fr auto; gap:.4rem; align-items:center; padding:.4rem 0; border-top:1px solid rgba(255,255,255,.06); }
.nap-depth-player > span { color:var(--muted); font-size:.68rem; text-align:center; }
.nap-depth-player strong { color:var(--text); font-size:.76rem; }
.nap-depth-player small { color:var(--muted); font-size:.65rem; }
.nap-schedule-table { margin:.65rem 0 1rem; overflow:visible; border:1px solid var(--border); border-radius:10px; background:rgba(11,27,46,.72); }
.nap-schedule-header,.nap-schedule-row { display:grid; grid-template-columns:.55fr 1fr 2.4fr .65fr 1fr 1.25fr; gap:.65rem; align-items:center; padding:.65rem .8rem; }
.nap-schedule-header { color:var(--muted); background:rgba(255,255,255,.025); font-size:.67rem; font-weight:800; text-transform:uppercase; letter-spacing:.05em; }
.nap-schedule-row { min-height:3.2rem; border-top:1px solid var(--border); color:var(--text); font-size:.78rem; }
.nap-schedule-row:hover { background:rgba(66,165,255,.045); }
.nap-schedule-opponent { display:flex; align-items:center; gap:.55rem; min-width:0; }
.nap-schedule-opponent strong { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.nap-schedule-date,.nap-schedule-venue,.nap-schedule-elo { color:var(--muted); }
.nap-schedule-status { justify-self:start; padding:.22rem .45rem; border-radius:999px; background:rgba(255,255,255,.05); font-size:.69rem; font-weight:800; }
.nap-schedule-completed .nap-schedule-status { color:var(--blue); }
.nap-schedule-current { border-left:3px solid var(--green); background:rgba(54,227,154,.07); }
.nap-schedule-current .nap-schedule-status { color:var(--green); background:rgba(54,227,154,.12); }
.nap-schedule-bye { background:rgba(255,194,71,.055); }
.nap-schedule-bye .nap-schedule-status,.nap-schedule-bye-label { color:#ffc247; }
div[data-testid="stRadio"] label { padding:.42rem .55rem; border-radius:7px; }
div[data-testid="stRadio"] label:hover { background:rgba(54,227,154,.07); }
div[data-testid="stRadio"] label:has(input:checked) { background:rgba(54,227,154,.11); border:1px solid rgba(54,227,154,.22); }
@media (max-width: 760px) {
  [data-testid="stSidebar"][aria-expanded="true"] { min-width:280px; max-width:280px; }
  [data-testid="stHeader"] {
    display:block !important; visibility:visible !important;
    min-height:3.5rem !important; pointer-events:auto !important;
  }
  [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {
    display:flex !important; visibility:visible !important; opacity:1 !important;
    pointer-events:auto !important;
    position:fixed !important; top:.55rem !important; left:.55rem !important;
    z-index:1000000 !important; width:2.75rem !important; height:2.75rem !important;
    align-items:center !important; justify-content:center !important;
    border:1px solid var(--border) !important; border-radius:9px !important;
    background:rgba(7,21,37,.96) !important; box-shadow:0 8px 24px rgba(0,0,0,.35) !important;
  }
  [data-testid="stSidebarCollapseButton"] {
    display:flex !important; visibility:visible !important; opacity:1 !important;
    pointer-events:auto !important; z-index:1000000 !important;
  }
  [data-testid="collapsedControl"] button, [data-testid="stSidebarCollapsedControl"] button,
  [data-testid="stSidebarCollapseButton"] button {
    display:flex !important; visibility:visible !important; opacity:1 !important;
    pointer-events:auto !important;
    min-width:2.75rem !important; min-height:2.75rem !important; color:var(--text) !important;
  }
  .block-container { padding:.8rem .8rem 6rem; }
  .nap-page-header { display:block; }
  .nap-refresh { padding:.15rem 0 .7rem; white-space:normal; }
  .nap-title { font-size:1.65rem; }
  .nap-game-team { font-size:1rem; gap:.4rem; }
  .nap-prediction-grid { grid-template-columns:1fr; }
  .nap-market-row { grid-template-columns:1fr 1fr; }
  .nap-market-edge { font-size:1.8rem; }
  .nap-sim-leaders { grid-template-columns:1fr 1fr; }
  .nap-roster-kpi { display:none; }
  .nap-unit-strip { grid-template-columns:1fr; }
  .nap-formation { grid-template-columns:repeat(7,minmax(82px,1fr)); padding:1rem .7rem; gap:.7rem .4rem; }
  .nap-formation-player { display:block; }
  .nap-formation-player img { margin:0 auto .25rem; }
  .nap-depth-groups { grid-template-columns:1fr; }
  .nap-schedule-table { border:0; background:transparent; }
  .nap-schedule-header { display:none; }
  .nap-schedule-row { grid-template-columns:2.3rem minmax(0,1fr) auto; grid-template-areas:"week opponent status" "week date venue"; gap:.25rem .5rem; margin-bottom:.5rem; padding:.65rem; border:1px solid var(--border); border-radius:8px; background:rgba(11,27,46,.72); }
  .nap-schedule-week { grid-area:week; align-self:center; color:var(--muted); }
  .nap-schedule-date { grid-area:date; }
  .nap-schedule-opponent { grid-area:opponent; }
  .nap-schedule-venue { grid-area:venue; justify-self:end; }
  .nap-schedule-status { grid-area:status; justify-self:end; white-space:nowrap; }
  .nap-schedule-elo { display:none; }
  .nap-schedule-current { border-left:3px solid var(--green); }
  .nap-tooltip-trigger { width:2.25rem; min-width:2.25rem; height:2.25rem; margin:-.6rem -.25rem; cursor:pointer; }
  .nap-tooltip-content { width:min(320px,calc(100vw - 2.5rem)); min-width:0; max-width:calc(100vw - 2.5rem); padding:.78rem .85rem; font-size:.8rem; line-height:1.6; }
}
</style>
"""
