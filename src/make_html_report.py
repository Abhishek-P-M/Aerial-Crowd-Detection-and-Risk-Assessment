<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>🛰️ Aerial Congestion & Risk — React Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <!-- Tailwind (CDN) -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- React + ReactDOM (CDN) -->
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <!-- PapaParse for CSV -->
  <script src="https://unpkg.com/papaparse@5.4.1/papaparse.min.js"></script>
  <!-- Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <!-- Lucide icons -->
  <script src="https://unpkg.com/lucide@0.469.0/dist/umd/lucide.min.js"></script>
  <style>
    /* nice scrollbars */
    ::-webkit-scrollbar{height:10px;width:10px}::-webkit-scrollbar-thumb{background:#c9d2e6;border-radius:8px}
    .card{ @apply bg-white dark:bg-slate-800 shadow-sm rounded-2xl border border-slate-200 dark:border-slate-700; }
    .muted{ @apply text-slate-500 dark:text-slate-400; }
    .kpi{ @apply card p-5 flex flex-col gap-1; }
    .pill{ @apply inline-block text-xs px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300; }
    .btn{ @apply px-3 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition; }
    .inp{ @apply w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100; }
    .select{ @apply inp pr-9; }
  </style>
</head>
<body class="bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100">
  <div id="root" class="p-4 sm:p-6 max-w-7xl mx-auto"></div>

  <script>
  (() => {
    const { useEffect, useMemo, useRef, useState } = React;

    // -------- Helpers --------
    const fmtPct = x => `${(x*100).toFixed(1)}%`;
    const toNum = v => (v === "" || v == null) ? 0 : Number(v);

    function useChart(canvasRef, cfgDeps, buildConfig) {
      useEffect(() => {
        if (!canvasRef.current) return;
        const ctx = canvasRef.current.getContext('2d');
        const chart = new Chart(ctx, buildConfig());
        return () => chart.destroy();
      // eslint-disable-next-line react-hooks/exhaustive-deps
      }, cfgDeps);
    }

    function useQueryDefaults() {
      const params = new URLSearchParams(location.search);
      return {
        metricsPath: params.get("metrics") || "metrics.csv",
        overlaysDir: params.get("overlays") || "overlays",
        heatmapsDir: params.get("heatmaps") || "heatmaps",
      };
    }

    function DarkModeToggle() {
      const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
      return (
        <button className="btn !bg-slate-700 hover:!bg-slate-800"
          onClick={() => {
            document.documentElement.classList.toggle("dark");
            setDark(d => !d);
          }}>
          {dark ? "Light" : "Dark"} Mode
        </button>
      );
    }

    function KPI({ label, value, sub }) {
      return (
        <div className="kpi">
          <div className="muted text-xs uppercase tracking-wide">{label}</div>
          <div className="text-2xl font-semibold">{value}</div>
          {sub && <div className="muted text-xs">{sub}</div>}
        </div>
      );
    }

    function ChartCard({ title, data, label }) {
      const ref = useRef(null);
      useChart(ref, [data], () => ({
        type: 'bar',
        data: {
          labels: data.map((_, i) => i+1),
          datasets: [{ label, data }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false }, title: { display: true, text: title } },
          scales: { x: { ticks: { display: false }}}
        }
      }));
      return (
        <div className="card p-4">
          <canvas ref={ref} height="140"></canvas>
        </div>
      );
    }

    function ImageCard({ image, overlaysDir, heatmapsDir, pri, ci, occ, dets }) {
      const base = image;
      const overlayPath = `${overlaysDir}/${base}`;
      const hm1 = `${heatmapsDir}/${base}`;
      const hm2 = `${heatmapsDir}/${base.replace(/\.[^.]+$/, '')}_heatmap.png`;
      return (
        <div className="card p-4">
          <div className="flex items-center justify-between gap-2 mb-3">
            <div className="font-medium">{base}</div>
            <div className="flex gap-2 flex-wrap">
              <span className="pill">PRI {pri.toFixed(2)}</span>
              <span className="pill">CI {ci.toFixed(2)}</span>
              <span className="pill">Occ {fmtPct(occ)}</span>
              <span className="pill">Det {dets}</span>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <img src={overlayPath} className="w-full rounded-xl border border-slate-200 dark:border-slate-700" loading="lazy"/>
            <img src={hm1}" onerror={`this.onerror=null;this.src='${hm2}'`} className="w-full rounded-xl border border-slate-200 dark:border-slate-700" loading="lazy"/>
          </div>
        </div>
      );
    }

    function App() {
      const defaults = useQueryDefaults();
      const [metricsPath, setMetricsPath] = useState(defaults.metricsPath);
      const [overlaysDir, setOverlaysDir] = useState(defaults.overlaysDir);
      const [heatmapsDir, setHeatmapsDir] = useState(defaults.heatmapsDir);
      const [sortBy, setSortBy] = useState("proximity_risk_index");
      const [minPRI, setMinPRI] = useState(0);
      const [minCI, setMinCI] = useState(0);
      const [topN, setTopN] = useState(24);
      const [rows, setRows] = useState([]);
      const [loading, setLoading] = useState(false);
      const [error, setError] = useState("");

      const numericColumns = ["proximity_risk_index","congestion_index","occupancy_frac","total_detections"];

      useEffect(() => {
        setLoading(true); setError("");
        Papa.parse(metricsPath, {
          download: true, header: true, dynamicTyping: true, skipEmptyLines: true,
          complete: (res) => {
            let data = res.data || [];
            // Ensure numeric types
            data = data.map(r => {
              const obj = {...r};
              numericColumns.forEach(k => obj[k] = toNum(obj[k]));
              obj.image = String(obj.image || "");
              return obj;
            });
            setRows(data);
            setLoading(false);
          },
          error: (err) => {
            setError("Failed to load metrics.csv: " + err.message);
            setRows([]); setLoading(false);
          }
        });
      }, [metricsPath]);

      const filtered = useMemo(() => {
        let out = rows.filter(r => r.proximity_risk_index >= minPRI && r.congestion_index >= minCI);
        out.sort((a,b) => (b[sortBy] - a[sortBy]) || (b.congestion_index - a.congestion_index));
        return out.slice(0, topN);
      }, [rows, sortBy, minPRI, minCI, topN]);

      const kpi = useMemo(() => {
        const n = rows.length;
        const mean = (k) => n ? rows.reduce((s,r)=>s+toNum(r[k]),0)/n : 0;
        return {
          images: n,
          meanPRI: mean("proximity_risk_index"),
          meanCI:  mean("congestion_index"),
          meanOcc: mean("occupancy_frac"),
        };
      }, [rows]);

      return (
        <div className="space-y-6">
          <header className="flex items-center justify-between gap-3">
            <div>
              <h1 className="text-2xl sm:text-3xl font-semibold">🛰️ Aerial Congestion & Risk — React Dashboard</h1>
              <p className="muted mt-1">Point this at your <code>metrics.csv</code>, <code>overlays/</code>, and <code>heatmaps/</code>. Use the filters to explore.</p>
            </div>
            <DarkModeToggle/>
          </header>

          <!-- Controls -->
          <div className="card p-4 grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-3">
            <label className="text-sm">metrics.csv path
              <input className="inp mt-1" value={metricsPath} onChange={e=>setMetricsPath(e.target.value)} />
            </label>
            <label className="text-sm">overlays dir
              <input className="inp mt-1" value={overlaysDir} onChange={e=>setOverlaysDir(e.target.value)} />
            </label>
            <label className="text-sm">heatmaps dir
              <input className="inp mt-1" value={heatmapsDir} onChange={e=>setHeatmapsDir(e.target.value)} />
            </label>
            <div className="grid grid-cols-2 gap-3">
              <label className="text-sm">min PRI
                <input type="number" step="0.1" className="inp mt-1" value={minPRI} onChange={e=>setMinPRI(Number(e.target.value))}/>
              </label>
              <label className="text-sm">min CI
                <input type="number" step="0.1" className="inp mt-1" value={minCI} onChange={e=>setMinCI(Number(e.target.value))}/>
              </label>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <label className="text-sm">sort by
                <select className="select mt-1" value={sortBy} onChange={e=>setSortBy(e.target.value)}>
                  <option value="proximity_risk_index">proximity_risk_index</option>
                  <option value="congestion_index">congestion_index</option>
                  <option value="occupancy_frac">occupancy_frac</option>
                  <option value="total_detections">total_detections</option>
                </select>
              </label>
              <label className="text-sm">top N
                <input type="number" min="1" max="2000" className="inp mt-1" value={topN} onChange={e=>setTopN(Number(e.target.value))}/>
              </label>
            </div>
          </div>

          <!-- KPIs -->
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <KPI label="Images" value={kpi.images}/>
            <KPI label="Mean PRI" value={kpi.meanPRI.toFixed(2)}/>
            <KPI label="Mean CI" value={kpi.meanCI.toFixed(2)}/>
            <KPI label="Mean Occupancy" value={fmtPct(kpi.meanOcc)}/>
          </div>

          <!-- Charts -->
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <ChartCard title="PRI distribution" data={rows.map(r=>toNum(r.proximity_risk_index))} label="PRI"/>
            <ChartCard title="CI distribution"  data={rows.map(r=>toNum(r.congestion_index))}     label="CI"/>
            <ChartCard title="Occupancy"        data={rows.map(r=>toNum(r.occupancy_frac))}      label="Occ"/>
          </div>

          <!-- List -->
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">Top frames</h2>
            {loading && <div className="muted text-sm">Loading…</div>}
            {error && <div className="text-red-500 text-sm">{error}</div>}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {filtered.map((r,i) => (
              <ImageCard key={r.image||i}
                image={r.image}
                overlaysDir={overlaysDir}
                heatmapsDir={heatmapsDir}
                pri={toNum(r.proximity_risk_index)}
                ci={toNum(r.congestion_index)}
                occ={toNum(r.occupancy_frac)}
                dets={toNum(r.total_detections)}
              />
            ))}
            {filtered.length === 0 && !loading && (
              <div className="muted">No rows after filters.</div>
            )}
          </div>

          <footer className="muted text-xs text-center py-6">
            Tip: add <code>?metrics=outputs/analytics/metrics.csv&overlays=outputs/analytics/overlays&heatmaps=outputs/analytics/heatmaps</code> to the URL.
          </footer>
        </div>
      );
    }

    ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
    window.lucide && window.lucide.createIcons();
  })();
  </script>
</body>
</html>
