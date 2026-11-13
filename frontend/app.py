"""Crop Advisor dashboard (Streamlit).

    cd frontend && streamlit run app.py

Talks to the FastAPI backend (CA_API_URL, default http://localhost:8000).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from api_client import API_KEY, API_URL, EMBEDDED, APIError, make_client

st.set_page_config(page_title="Crop Advisor", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
.crop-name {font-size: 2.4rem; font-weight: 700; line-height: 1.1; text-transform: capitalize; margin: 0;}
.badge {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin-right: 6px;}
.badge.ok {background:#dff0e4; color:#1d6b3a;} .badge.warn {background:#fdf0d0; color:#8a5a00;}
.badge.bad {background:#f8dcd8; color:#9b2c1f;} .badge.mute {background:#e8e6dc; color:#555;}
.small {font-size: .82rem; color: #6b7468;}
</style>
""",
    unsafe_allow_html=True,
)

CROP_EMOJI = {"rice": "🌾", "maize": "🌽", "banana": "🍌", "apple": "🍎", "mango": "🥭", "orange": "🍊", "coconut": "🥥",
              "watermelon": "🍉", "grapes": "🍇", "coffee": "☕", "cotton": "☁️", "papaya": "🍈", "pomegranate": "🍎"}
FEATURE_ORDER = ["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]
STATUS_ICON = {"ok": "🟢 in range", "low": "🔻 too low", "high": "🔺 too high", "missing": "⚪ no data"}
SOURCE_LABEL = {"sensor": "field sensor", "weather": "live weather", "soil-map": "soil map (SoilGrids)", "manual": "manual", None: "—"}
PALETTE = {"N": "#2f7d4f", "P": "#c9822b", "K": "#6a5acd", "ph": "#b5483a", "moisture": "#2b7fb8", "soil_temp": "#d1603d"}


@st.cache_resource
def get_client():
    return make_client()


api = get_client()


# ----------------------------------------------------------------------------- helpers
def minutes_ago(ts: str | None) -> float | None:
    if not ts:
        return None
    dt = datetime.fromisoformat(ts)
    dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60


def ago_text(ts: str | None) -> str:
    m = minutes_ago(ts)
    if m is None:
        return "never"
    if m < 1:
        return "just now"
    if m < 90:
        return f"{m:.0f} min ago"
    return f"{m / 60:.1f} h ago"


def badge(text: str, kind: str) -> str:
    return f'<span class="badge {kind}">{text}</span>'


@st.cache_data(ttl=3600, show_spinner=False)
def geocode(query: str) -> list[dict]:
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": query, "count": 6}, timeout=8)
        return r.json().get("results", [])
    except Exception:
        return []


def line_chart(df: pd.DataFrame, series: dict[str, tuple[str, str]], height=250, ytitle=None, yrange=None) -> go.Figure:
    fig = go.Figure()
    for col, (label, color) in series.items():
        if col in df and df[col].notna().any():
            fig.add_trace(go.Scatter(x=df["ts"], y=df[col], name=label, mode="lines", line=dict(color=color, width=2), connectgaps=True))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.15, x=0),
                      yaxis_title=ytitle, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.2)", range=yrange)
    return fig


# ----------------------------------------------------------------------------- result widgets (shared)
def render_recommendation(rec: dict, area_ha: float | None) -> None:
    top = rec["recommendations"][0]
    conf = rec["confidence"]
    ckind = {"high": "ok", "medium": "warn", "low": "bad"}[conf]
    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.caption("RECOMMENDED CROP")
            st.markdown(f'<p class="crop-name">{CROP_EMOJI.get(top["crop"], "🌱")} {top["crop"]}</p>', unsafe_allow_html=True)
            st.markdown(
                badge(f"{top['probability']:.0%} match", "ok") + badge(f"{conf} confidence", ckind)
                + (badge(f"{len(rec['missing'])} input(s) missing", "mute") if rec["missing"] else ""),
                unsafe_allow_html=True,
            )
            y = top.get("yield")
            if y:
                c1, c2 = st.columns(2)
                c1.metric("Expected yield", f"{y['p50']:,.0f} kg/ha", help="Median estimate from the regression model")
                c2.metric("Likely range (80%)", f"{y['p10']:,.0f} – {y['p90']:,.0f}", help="Approximate 80% prediction interval, kg/ha")
                if area_ha:
                    st.caption(f"≈ **{y['p50'] * area_ha / 1000:,.1f} tonnes** over this plot's {area_ha:g} ha")
            for n in rec["notes"]:
                st.caption("⚠️ " + n)
    with right:
        with st.container(border=True):
            st.caption("OTHER OPTIONS")
            for alt in rec["recommendations"][1:]:
                st.markdown(f"**{CROP_EMOJI.get(alt['crop'], '🌱')} {alt['crop'].title()}** · {alt['probability']:.0%}")
                st.progress(min(max(alt["probability"], 0.0), 1.0))
                if alt.get("yield"):
                    st.caption(f"≈ {alt['yield']['p50']:,.0f} kg/ha  ({alt['yield']['p10']:,.0f}–{alt['yield']['p90']:,.0f})")


def render_explanation(rec: dict) -> None:
    ex = rec["explanation"]
    st.write(ex["summary"])
    rows = []
    for f in sorted(ex["features"], key=lambda f: FEATURE_ORDER.index(f["feature"])):
        lo, hi = f.get("ideal_low"), f.get("ideal_high")
        rows.append({
            "Input": f["label"],
            "Value": f["value"],
            f"Typical for {ex['crop']}": f"{lo:g} – {hi:g}" if lo is not None else "—",
            "Status": STATUS_ICON[f["status"]],
            "Source": SOURCE_LABEL.get(rec.get("sources", {}).get(f["feature"]), "—"),
            "Influence": f["influence"],
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={"Influence": st.column_config.ProgressColumn(
            "Influence on match", help="How much the match probability drops if this input were unknown",
            min_value=0, max_value=0.5, format="%.2f")},
    )
    if ex["advice"]:
        st.markdown("**What you can do**")
        for a in ex["advice"]:
            st.markdown(f"- {a}")
    st.caption("General guidance only: confirm fertiliser doses with a local agriculture extension officer.")


# ----------------------------------------------------------------------------- live dashboard
def live_view(plot_id: str, top_k: int) -> None:
    try:
        rec = api.recommendation(plot_id, top_k=top_k)
        rows = api.readings(plot_id, limit=300)
    except APIError as exc:
        st.error(str(exc))
        return
    st.session_state["last_rec"] = rec
    plot, sensor, wx = rec["plot"], rec["sensor"], rec.get("weather")

    # status strip
    s_badge = badge("sensors online", "ok") if sensor["online"] else badge("sensors offline", "bad" if sensor["last_seen"] else "mute")
    w_badge = (badge("weather MOCK", "warn") if wx and wx.get("source") == "mock"
               else badge("weather live", "ok") if wx and not wx["stale"]
               else badge("weather cached", "warn") if wx else badge("weather unavailable", "bad"))
    src = rec["model"]["data_source"]
    d_badge = badge("demo data model", "warn") if src["crop"] == "synthetic" else badge("trained on real data", "ok")
    st.markdown(
        f"{s_badge}{w_badge}{d_badge}<span class='small'> last sensor reading: {ago_text(sensor['last_seen'])}"
        f" · updated {datetime.now().strftime('%H:%M:%S')}</span>",
        unsafe_allow_html=True,
    )

    for a in rec["alerts"]:
        {"critical": st.error, "warning": st.warning, "info": st.info}[a["level"]](a["message"], icon={"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}[a["level"]])

    render_recommendation(rec, plot.get("area_ha"))

    with st.expander("Why this crop?", expanded=False):
        render_explanation(rec)

    # sensor tiles
    st.subheader("Field sensors")
    df = pd.DataFrame(rows)
    vals = sensor["values"]
    tiles = [("N", "n", "Nitrogen", ""), ("P", "p", "Phosphorus", ""), ("K", "k", "Potassium", ""),
             ("pH", "ph", "Soil pH", ""), ("Moisture", "soil_moisture", "Soil moisture", "%"), ("Soil temp", "soil_temp", "Soil temp", " °C")]
    cols = st.columns(len(tiles))
    for col, (title, key, _, unit) in zip(cols, tiles):
        v = vals.get(key)
        delta = None
        if not df.empty and key in df and df[key].notna().sum() >= 2 and v is not None:
            delta = round(float(df[key].dropna().iloc[-1] - df[key].dropna().iloc[-2]), 2)
        col.metric(title, "—" if v is None else f"{v:g}{unit}", delta=delta)
    if df.empty:
        st.info("No sensor readings yet. Start the simulator or send a test reading from the **Connect sensors** tab.")
    else:
        df["ts"] = pd.to_datetime(df["ts"])
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(line_chart(df, {"n": ("N", PALETTE["N"]), "p": ("P", PALETTE["P"]), "k": ("K", PALETTE["K"])}), width="stretch", key="ch_npk")
        c2.plotly_chart(line_chart(df, {"ph": ("pH", PALETTE["ph"])}, yrange=[4, 9]), width="stretch", key="ch_ph")
        c3.plotly_chart(line_chart(df, {"soil_moisture": ("Moisture %", PALETTE["moisture"]), "soil_temp": ("Soil °C", PALETTE["soil_temp"])}), width="stretch", key="ch_moist")

    # weather
    st.subheader("Live weather")
    if not wx:
        st.warning("Weather data is currently unavailable.")
    else:
        cur, inp = wx["current"], rec["inputs"]
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Temperature now", "—" if cur.get("temperature") is None else f"{cur['temperature']:.1f} °C")
        w2.metric("Humidity now", "—" if cur.get("humidity") is None else f"{cur['humidity']:.0f} %")
        w3.metric("Rain, last 30 days", "—" if inp["rainfall"] is None else f"{inp['rainfall']:.0f} mm", help="Used as the model's rainfall input")
        rain3 = sum((d["rain_mm"] or 0) for d in wx["forecast"][:3])
        w4.metric("Rain forecast, 3 days", f"{rain3:.0f} mm")
        if wx["forecast"]:
            fig = go.Figure(go.Bar(x=[d["date"] for d in wx["forecast"]], y=[d["rain_mm"] or 0 for d in wx["forecast"]], marker_color=PALETTE["moisture"], name="Rain mm"))
            fig.update_layout(height=190, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="mm / day", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch", key="ch_fc")


# ----------------------------------------------------------------------------- sidebar
def sidebar_add_plot() -> None:
    with st.expander("➕ Add a plot"):
        q = st.text_input("Search a place", key="geo_q", placeholder="e.g. Ludhiana")
        if q:
            results = geocode(q)
            if results:
                labels = [f"{r['name']}, {r.get('admin1', '')}, {r.get('country', '')}" for r in results]
                pick = st.selectbox("Matches", range(len(results)), format_func=lambda i: labels[i], key="geo_pick")

                def apply():
                    r = results[st.session_state["geo_pick"]]
                    st.session_state.update(np_lat=float(r["latitude"]), np_lon=float(r["longitude"]), np_name=r["name"])

                st.button("Use this location", on_click=apply)
            else:
                st.caption("No matches (or geocoding is offline) — enter coordinates manually.")
        with st.form("new_plot", clear_on_submit=False):
            name = st.text_input("Plot name", key="np_name", value=st.session_state.get("np_name", ""))
            lat = st.number_input("Latitude", -90.0, 90.0, st.session_state.get("np_lat", 30.9), format="%.5f", key="np_lat")
            lon = st.number_input("Longitude", -180.0, 180.0, st.session_state.get("np_lon", 75.85), format="%.5f", key="np_lon")
            area = st.number_input("Area (hectares)", 0.01, 100000.0, 1.0)
            if st.form_submit_button("Create plot", width="stretch"):
                try:
                    p = api.create_plot(name.strip() or "My plot", lat, lon, area)
                    st.session_state["plot_id"] = p["id"]
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))


# ----------------------------------------------------------------------------- tabs
def tab_whatif(plot: dict, classes: list[str]) -> None:
    st.write("Try different soil/weather values on top of the live data, for example *\"what if I add nitrogen?\"* or *\"what if it turns drier?\"*. Nothing is saved.")
    live = st.session_state.get("last_rec", {}).get("inputs", {})
    defaults = {"N": 60.0, "P": 45.0, "K": 40.0, "ph": 6.5, "temperature": 25.0, "humidity": 70.0, "rainfall": 100.0}
    ranges = {"N": (0.0, 200.0, 1.0), "P": (0.0, 200.0, 1.0), "K": (0.0, 250.0, 1.0), "ph": (3.5, 9.5, 0.1),
              "temperature": (5.0, 45.0, 0.5), "humidity": (10.0, 100.0, 1.0), "rainfall": (0.0, 350.0, 5.0)}
    labels = {"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium", "ph": "Soil pH", "temperature": "Temperature °C", "humidity": "Humidity %", "rainfall": "Rainfall 30 d, mm"}
    chosen: dict[str, float] = {}
    cols = st.columns(4)
    for i, f in enumerate(FEATURE_ORDER):
        with cols[i % 4]:
            on = st.checkbox(f"Override {labels[f]}", key=f"ov_{f}")
            lo, hi, step = ranges[f]
            start = float(min(max(live.get(f) if live.get(f) is not None else defaults[f], lo), hi))
            v = st.slider(labels[f], lo, hi, start, step, key=f"sl_{f}", disabled=not on, label_visibility="collapsed")
            if on:
                chosen[f] = v
    crop = st.selectbox("Estimate yield for a specific crop (optional)", ["(best match)"] + classes)
    if st.button("Run what-if", type="primary"):
        try:
            res = api.whatif(plot["id"], chosen, None if crop.startswith("(") else crop)
            st.session_state["whatif"] = res
        except APIError as exc:
            st.error(str(exc))
    res = st.session_state.get("whatif")
    if res:
        render_recommendation(res, plot.get("area_ha"))
        with st.expander("Why?", expanded=True):
            render_explanation(res)


def tab_history(plot: dict) -> None:
    try:
        h = api.history(plot["id"], 200)
    except APIError as exc:
        st.error(str(exc))
        return
    if not h:
        st.info("No predictions logged yet.")
        return
    df = pd.DataFrame(h)
    df["ts"] = pd.to_datetime(df["ts"])
    fig = go.Figure(go.Scatter(x=df["ts"], y=df["yield_p50"], mode="lines+markers", line=dict(color=PALETTE["N"]), text=df["top_crop"], hovertemplate="%{text}: %{y:,.0f} kg/ha<extra></extra>"))
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Predicted yield, kg/ha", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", key="ch_hist")
    st.caption("A new entry is stored when the recommended crop changes or every 5 minutes.")
    st.dataframe(df.sort_values("ts", ascending=False).rename(columns={"ts": "Time", "top_crop": "Crop", "probability": "Match", "yield_p50": "Yield kg/ha", "confidence": "Confidence"}).drop(columns=["id"]),
                 hide_index=True, width="stretch")


def tab_model() -> None:
    try:
        info = api.model_info()
    except APIError as exc:
        st.error(str(exc))
        return
    m, src = info["metrics"], info["data_source"]
    if src["crop"] == "synthetic" or src["yield"] == "synthetic":
        st.warning(
            f"**Demo models.** Crop data: `{src['crop']}`, yield data: `{src['yield']}`. Synthetic data is generated from published "
            "crop-requirement profiles, so the accuracy below is optimistic and the yield numbers are illustrative. "
            "Add the real CSVs to `data/` and call `POST /api/model/retrain` (see README)."
        )
    c, y = m["classifier"], m["yield"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Crop accuracy (hold-out)", f"{c['complete_data']['accuracy']:.1%}")
    k2.metric("Crop macro-F1", f"{c['complete_data']['macro_f1']:.3f}")
    k3.metric("Yield R²", f"{y['holdout']['r2']:.3f}")
    k4.metric("Yield MAPE", f"{y['holdout']['mape']:.1%}")

    a, b = st.columns(2)
    with a:
        st.markdown("**Robustness: accuracy as inputs go missing**")
        ks = sorted(c["missing_k_features"]["robust_model"], key=int)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=[c["missing_k_features"]["robust_model"][k] for k in ks], name="Trained with missing-data augmentation", line=dict(color=PALETTE["N"], width=3)))
        fig.add_trace(go.Scatter(x=ks, y=[c["missing_k_features"]["plain_model"][k] for k in ks], name="Plain model", line=dict(color="#999", dash="dash")))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Number of missing inputs (of 7)", yaxis_title="Accuracy", yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.18), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", key="m_rob")
    with b:
        st.markdown("**Soil vs weather: which matters for choosing a crop?**")
        g = c["feature_group_ablation"]
        fig = go.Figure(go.Bar(x=list(g), y=list(g.values()), marker_color=[PALETTE["N"], PALETTE["P"], PALETTE["moisture"]], text=[f"{v:.0%}" for v in g.values()], textposition="outside"))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis_tickformat=".0%", yaxis_range=[0, 1.1], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", key="m_grp")

    a, b = st.columns(2)
    with a:
        st.markdown("**Feature importance (crop model)**")
        imp = pd.Series(info["feature_importance"]).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=PALETTE["N"]))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", key="m_imp")
    with b:
        st.markdown("**Yield: regional generalisation (leave-one-region-out)**")
        lo = y.get("leave_one_region_out", {})
        if lo:
            st.dataframe(pd.DataFrame(lo).T.rename(columns={"r2": "R²", "mape": "MAPE"}).style.format({"R²": "{:.3f}", "MAPE": "{:.1%}"}), width="stretch")
        sv = y["soil_vs_weather"]
        st.caption(f"Yield R² with everything: {sv['none']:.2f} · without weather: {sv['weather_missing']:.2f} · without soil: {sv['soil_missing']:.2f}")


def _connect_instructions(plot: dict) -> None:
    st.write("Any device that can send an HTTP request (ESP32, Raspberry Pi, LoRa gateway, a phone app) or publish to MQTT can feed this plot. Send only what your hardware measures; missing fields are fine.")
    st.markdown("**HTTP**")
    st.code(
        f"""curl -X POST {API_URL}/api/sensors/readings \\
  -H "X-API-Key: <your key>" -H "Content-Type: application/json" \\
  -d '{{"plot_id": "{plot['id']}", "device_id": "node-1", "n": 80, "p": 48, "k": 40, "ph": 6.4, "soil_moisture": 31.5, "soil_temp": 24.1}}'""",
        language="bash",
    )
    st.markdown("**MQTT** (set `CA_MQTT_HOST` on the backend)")
    st.code(f'topic:   cropadvisor/plots/{plot["id"]}/readings\npayload: {{"n": 80, "p": 48, "k": 40, "ph": 6.4, "soil_moisture": 31.5}}', language="text")
    st.markdown("**No hardware yet?** Run the simulator, or send a manual test reading:")
    st.code(f"python simulator/sensor_simulator.py --plot {plot['id']} --interval 5", language="bash")


def tab_connect(plot: dict) -> None:
    if EMBEDDED:
        st.info(
            "This hosted demo runs the dashboard and the model in one process. The demo plot is fed by a built-in "
            "simulated sensor, and the sensor HTTP/MQTT API is not exposed here. To connect real hardware, run the full "
            "version with the separate API (see the README). You can still send a manual test reading below."
        )
    else:
        _connect_instructions(plot)
    with st.form("test_reading"):
        c = st.columns(6)
        n, p, k = c[0].number_input("N", 0.0, 500.0, 80.0), c[1].number_input("P", 0.0, 500.0, 48.0), c[2].number_input("K", 0.0, 500.0, 40.0)
        ph, mo, stt = c[3].number_input("pH", 0.0, 14.0, 6.4, step=0.1), c[4].number_input("Moisture %", 0.0, 100.0, 35.0), c[5].number_input("Soil °C", -10.0, 60.0, 24.0)
        if st.form_submit_button("Send test reading", type="primary"):
            try:
                api.send_reading(plot["id"], n=n, p=p, k=k, ph=ph, soil_moisture=mo, soil_temp=stt)
                st.success("Stored. Switch to the Live tab.")
            except APIError as exc:
                st.error(str(exc))
    if not EMBEDDED:
        st.caption(f"Backend: `{API_URL}` · API docs: `{API_URL}/docs`")


# ----------------------------------------------------------------------------- main
def main() -> None:
    st.sidebar.title("🌾 Crop Advisor")
    try:
        health = api.health()
        plots = api.plots()
    except APIError as exc:
        st.error(str(exc))
        st.code("cd backend && uvicorn cropadvisor.main:app --port 8000", language="bash")
        st.stop()
    if not health["models_ready"]:
        st.sidebar.warning("Models are training for the first time (about a minute)…")

    with st.sidebar:
        sidebar_add_plot()
    if not plots:
        st.title("Welcome 👋")
        st.write("Create your first plot in the sidebar (search a place or enter coordinates). Sensors and live weather attach to it.")
        st.stop()

    ids = [p["id"] for p in plots]
    if st.session_state.get("plot_id") not in ids:
        st.session_state["plot_id"] = ids[0]
    st.sidebar.selectbox("Plot", ids, key="plot_id", format_func=lambda i: next(p["name"] for p in plots if p["id"] == i))
    plot = next(p for p in plots if p["id"] == st.session_state["plot_id"])
    st.sidebar.caption(f"📍 {plot['lat']:.3f}, {plot['lon']:.3f} · {plot['area_ha']:g} ha")
    refresh = st.sidebar.select_slider("Auto-refresh", ["Off", "5 s", "10 s", "30 s", "60 s"], value="10 s")
    top_k = st.sidebar.slider("Crops to show", 2, 5, 3)
    with st.sidebar.expander("Danger zone"):
        if st.checkbox("I want to delete this plot and its data"):
            if st.button("Delete plot", type="secondary"):
                api.delete_plot(plot["id"])
                st.session_state.pop("plot_id", None)
                st.rerun()

    st.title(plot["name"])
    t_live, t_what, t_hist, t_model, t_conn = st.tabs(["Live", "What-if", "History", "Model insights", "Connect sensors"])
    with t_live:
        every = None if refresh == "Off" else int(refresh.split()[0])
        st.fragment(live_view, run_every=every)(plot["id"], top_k)
    with t_what:
        try:
            classes = api.model_info()["classes"]
        except APIError:
            classes = []
        tab_whatif(plot, classes)
    with t_hist:
        tab_history(plot)
    with t_model:
        tab_model()
    with t_conn:
        tab_connect(plot)


main()
