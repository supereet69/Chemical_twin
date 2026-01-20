import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from graphviz import Digraph
from sklearn.metrics import r2_score, mean_squared_error


@st.cache_data
def compute_model_metrics(_model, FEATURES):
    """
    Offline validation metrics computed from experimental dataset.
    Used ONLY for reporting model credibility (not runtime prediction).
    """
    try:
        from preprocessing import load_and_preprocess, build_features
    except Exception:
        return None

    df = load_and_preprocess("data/data.csv", dt=10)
    df_feat, FEATURES_TRAIN, TARGETS = build_features(df)

    X = df_feat[FEATURES]
    y_true = df_feat[TARGETS].values
    y_pred = _model.predict(X)

    metrics = {}
    for i, tgt in enumerate(TARGETS):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        metrics[tgt] = {"rmse": rmse, "r2": r2}

    return metrics


# Optional: Arduino serial
try:
    import serial
    from serial.serialutil import SerialException
except Exception:
    serial = None
    SerialException = Exception


# =============================
# CONFIG
# =============================
MODEL_PATH = "models/chemical_rate_model.pkl"

DT = 10                      # seconds (simulation step)
ALPHA_INTEGRATION = 0.35     # damping factor to avoid runaway

# Realtime tuning
RT_READ_BATCH_DEFAULT = 25   # how many serial lines to attempt per rerun
RT_SLEEP = 0.02              # streamlit refresh sleep
SER_TIMEOUT = 0.05           # serial timeout (fast)

# Keep last N points (avoid heavy plots)
MAX_POINTS = 600

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]

model_metrics = compute_model_metrics(model, FEATURES)



# =============================
# UTILS
# =============================
def stabilized_flag(df, window=6, thr=10):
    if len(df) < window + 1:
        return False
    recent = df["tds"].diff().tail(window).abs().mean()
    return recent < thr

# ---------- Add these into the UTILS section (above the UI code) ----------
def compute_rate(df, col, time_col="time_s", window=3):
    """Return dataframe with time_s and smoothed rate d{col}/dt as 'd{col}_dt'."""
    if df is None or col not in df.columns or time_col not in df.columns:
        return None
    if len(df) < window + 1:
        return None

    dC = df[col].diff()
    dt = df[time_col].diff().replace(0, np.nan)
    rate = (dC / dt).rolling(window, min_periods=1).mean()
    out = df[[time_col]].copy()
    out_col = f"d{col}_dt"
    out[out_col] = rate
    return out.dropna().reset_index(drop=True)


def compute_recovery_index(df):
    """Normalized TDS-based recovery index in [0,1]. Returns df with 'recovery_index'."""
    if df is None or "tds" not in df.columns:
        return None
    if len(df) < 2:
        return None

    tds0 = df["tds"].iloc[0]
    tds_max = df["tds"].max()
    if np.isclose(tds_max, tds0) or tds_max <= tds0:
        return None

    rec = (df["tds"] - tds0) / (tds_max - tds0)
    out = df[["time_s"]].copy()
    out["recovery_index"] = rec.clip(0, 1).reset_index(drop=True)
    return out


def estimate_k_app(df):
    """
    Estimate an apparent pseudo-first-order rate constant from TDS vs time.
    Returns k_app (s^-1) or None if not enough or unsuitable data.
    Uses linear fit to ln((C_inf - C)/(C_inf - C0)).
    """
    if df is None or "tds" not in df.columns or "time_s" not in df.columns:
        return None
    if len(df) < 6:
        return None

    t = np.asarray(df["time_s"], dtype=float)
    C = np.asarray(df["tds"], dtype=float)
    C_inf = np.max(C)
    eps = 1e-8

    # make safe y = ln( (C_inf - C) / (C_inf - C0) )
    denom = (C_inf - C[0])
    if denom <= eps:
        return None

    y_raw = (C_inf - C) / denom
    # keep only positive finite entries
    mask = np.isfinite(y_raw) & (y_raw > 0)
    if mask.sum() < 4:
        return None

    t_fit = t[mask]
    y_fit = np.log(y_raw[mask])

    # linear fit y = -k * t + b  => slope = -k
    try:
        slope, intercept = np.polyfit(t_fit, y_fit, 1)
        k_app = -slope
        if k_app <= 0 or not np.isfinite(k_app):
            return None
        return float(k_app)
    except Exception:
        return None
# --------------------------------------------------------------------------



def estimate_time_to_plateau(df, thr=10):
    if len(df) < 8:
        return None
    recent = df["tds"].diff().tail(6).abs().mean()
    if recent < thr:
        return 0
    return int((recent / thr) * 60)


def realtime_eta_to_stable(df, window=8, thr=8.0):
    """
    ETA to stabilization in realtime based on avg |dTDS/dt|.
    Returns:
      None  -> not enough data
      0     -> already stable
      int   -> seconds
    """
    if df is None or len(df) < window + 2:
        return None

    dtds = df["tds"].diff()
    dt = df["time_s"].diff()

    rate = (dtds / dt.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rate) < window:
        return None

    recent_rate = rate.tail(window).abs().mean()

    if recent_rate < thr:
        return 0

    scale = 120.0
    eta = int((recent_rate / thr) * scale)
    return max(5, min(eta, 3600))


def temp_dynamics(temp_curr, h2o2_ml, dt):
    ambient = 28.0
    h = float(h2o2_ml)

    if h <= 0:
        return float(temp_curr)

    target = ambient + (10.0 + 7.0 * h)
    k = 0.004
    next_temp = temp_curr + (target - temp_curr) * (1 - np.exp(-k * dt))
    next_temp += 0.01 * h * dt
    return float(max(temp_curr, next_temp))


def turbidity_sensor_update(turb_curr, turb_rate_ml, dt, noise_scale=0.8):
    turb_curr = float(turb_curr)
    growth = max(0.0, float(turb_rate_ml))

    turb_next = turb_curr + ALPHA_INTEGRATION * growth * dt
    turb_next += 0.08 * dt
    turb_next += np.random.normal(0.0, noise_scale)
    turb_next = max(0.0, turb_next)

    beta = 0.85
    return float(beta * turb_curr + (1 - beta) * turb_next)


def apply_anomaly_to_rates(tds_rate, turb_rate, mode="Sensor Spike", severity=1.0):
    s = float(severity)
    if mode == "Sensor Spike":
        return tds_rate * (1.0 + 0.6 * s), turb_rate * (1.0 + 0.4 * s)
    if mode == "Acid Underfeed":
        return tds_rate * (1.0 - 0.35 * s), turb_rate * (1.0 - 0.25 * s)
    if mode == "Over-oxidation (H2O2)":
        return tds_rate * (1.0 + 0.25 * s), turb_rate * (1.0 + 0.20 * s)
    return tds_rate, turb_rate


def predict_rates(model, FEATURES, row_dict):
    X = np.array([[row_dict.get(c, 0.0) for c in FEATURES]], dtype=float)
    pred = model.predict(X)[0]
    return float(pred[0]), float(pred[1])

def predict_anomaly(row_dict):
    # plug-in later
    return False, 0.0

def short_forecast(model, FEATURES, last_state, steps=6):
    temp = last_state.copy()
    out = []
    for _ in range(steps):
        r_tds, r_turb = predict_rates(model, FEATURES, temp)
        temp["time_s"] += DT
        temp["tds"] += ALPHA_INTEGRATION * r_tds * DT
        temp["temp_C"] = temp_dynamics(temp["temp_C"], temp["h2o2_ml"], DT)
        temp["turbidity"] = turbidity_sensor_update(temp["turbidity"], r_turb, DT, 0.0)
        out.append(temp.copy())
    return pd.DataFrame(out)


def make_flowsheet(state="IDLE", anomaly=False):
    g = Digraph("DT")
    g.attr(rankdir="TB", bgcolor="transparent", nodesep="0.55", ranksep="0.65")

    C_IDLE = "#374151"
    C_RUN = "#2563EB"
    C_DONE = "#16A34A"
    C_ANOM = "#DC2626"
    C_BOX = "#0B1220"

    if state == "RUNNING":
        accent = C_RUN
    elif state == "DONE":
        accent = C_DONE
    elif state == "ANOMALY":
        accent = C_ANOM
    else:
        accent = C_IDLE

    pipeline = C_ANOM if anomaly else accent

    def node(name, label):
        g.node(
            name, label,
            shape="box",
            style="rounded,filled",
            color=pipeline,
            fillcolor=C_BOX,
            fontcolor="white",
            penwidth="2"
        )

    node("U", "User Inputs / Setpoints")
    node("R", "Batch Reactor (Leaching)")
    node("S", "Sensors\nTDS • Turbidity • Temp")
    node("ML", "ML Predictor\n(dTDS/dt, dTurb/dt)")
    node("DT", "Digital Twin\nIntegration + Forecast")
    node("OUT", "KPIs + Trends")

    g.edge("U", "R", color=pipeline)
    g.edge("R", "S", color=pipeline)
    g.edge("S", "ML", color=pipeline)
    g.edge("ML", "DT", color=pipeline)
    g.edge("DT", "OUT", color=pipeline)

    if anomaly:
        g.edge("S", "DT", label="⚠ anomaly", color=C_ANOM, fontcolor=C_ANOM)

    return g


def plotly_line(df_base, ycol, title, ylab, df_anom=None, df_forecast=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_base["time_s"],
        y=df_base[ycol],
        mode="lines+markers",
        name="Sensor",
        marker=dict(size=6),
        line=dict(width=2),
    ))

    # highlight latest point
    if len(df_base) > 0:
        fig.add_trace(go.Scatter(
            x=[df_base["time_s"].iloc[-1]],
            y=[df_base[ycol].iloc[-1]],
            mode="markers",
            marker=dict(size=12),
            showlegend=False
        ))

    if df_anom is not None and len(df_anom) > 0:
        fig.add_trace(go.Scatter(
            x=df_anom["time_s"], y=df_anom[ycol],
            mode="lines", name="Anomaly"
        ))

    if df_forecast is not None and len(df_forecast) > 0:
        fig.add_trace(go.Scatter(
            x=df_forecast["time_s"], y=df_forecast[ycol],
            mode="lines", name="Forecast",
            line=dict(dash="dash")
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=ylab,
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # auto-scroll
    if len(df_base) > 20:
        x_min = float(df_base["time_s"].iloc[-20])
        x_max = float(df_base["time_s"].iloc[-1])
        fig.update_xaxes(range=[x_min, x_max])

    return fig


def keep_last_n(df, n=MAX_POINTS):
    if len(df) <= n:
        return df
    return df.iloc[-n:].reset_index(drop=True)


def safe_close_serial():
    try:
        ser = st.session_state.get("ser", None)
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
    finally:
        st.session_state.ser = None
        st.session_state.serial_connected = False


def try_connect_serial(port, baud):
    """
    Connects cleanly. Returns True/False.
    """
    if serial is None:
        return False

    safe_close_serial()
    try:
        ser = serial.Serial(port, int(baud), timeout=SER_TIMEOUT)
        time.sleep(1.0)  # allow Arduino reset
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        st.session_state.ser = ser
        st.session_state.serial_connected = True
        return True
    except Exception as e:
        safe_close_serial()
        st.session_state.events.append(f"Arduino connect failed: {e}")
        return False


# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="Leaching Digital Twin", layout="wide")

# --- PAGE HEADER ---
header_col1, header_col2 = st.columns([2.5, 1.5])
with header_col1:
    st.title("Battery Black Mass Leaching — Digital Twin")
    st.caption("Process monitoring and virtual commissioning interface for leaching operations.")

with header_col2:
    with st.container(border=True):
        st.markdown("**Session Status**")
        # Simple high-level state indicator later populated from session state
        # (just an informative label; logic unchanged below)
        if "running" in st.session_state and st.session_state.running:
            st.markdown("• RUNNING")
        elif "finalized" in st.session_state and st.session_state.finalized:
            st.markdown("• COMPLETED")
        else:
            st.markdown("• IDLE")


# ---------- Session ----------
if "running" not in st.session_state:
    st.session_state.running = False
if "finalized" not in st.session_state:
    st.session_state.finalized = False
if "hist_base" not in st.session_state:
    st.session_state.hist_base = pd.DataFrame()
if "hist_anom" not in st.session_state:
    st.session_state.hist_anom = pd.DataFrame()
if "events" not in st.session_state:
    st.session_state.events = []
if "serial_connected" not in st.session_state:
    st.session_state.serial_connected = False
if "ser" not in st.session_state:
    st.session_state.ser = None
if "rt_anom" not in st.session_state:
    st.session_state.rt_anom = False
if "rt_anom_score" not in st.session_state:
    st.session_state.rt_anom_score = 0.0

# =============================
# TOP CONTROL STRIP
# =============================
top_container = st.container(border=True)
with top_container:
    topL, topR = st.columns([1.0, 3.0])

    with topL:
        st.subheader("Operating Mode")
        mode = st.radio(
            "Select mode",
            ["Virtual Commissioning", "Online Monitoring (Arduino)"],
            label_visibility="collapsed"
        )

    with topR:
        if mode == "Virtual Commissioning":
            st.subheader("Process Setpoints")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            acid_M = c1.selectbox("Acid (M)", [0.5, 1.0, 1.5, 2.0], index=1)
            h2o2_ml = c2.selectbox("H₂O₂ (mL)", [0.0, 0.5, 1.0, 2.0], index=1)
            sim_time = c3.number_input("Reaction Time (s)", value=600, step=30, min_value=60)
            tds0 = c4.number_input("Init TDS", value=1800.0, step=50.0)
            turb0 = c5.number_input("Init Turb", value=700.0, step=10.0)
            temp0 = c6.number_input("Init Temp", value=30.0, step=1.0)
        else:
            st.subheader("Process Setpoints")
            acid_M = 1.0
            h2o2_ml = 1.0
            sim_time = 600
            tds0 = 0.0
            turb0 = 0.0
            temp0 = 0.0
            st.caption("In Online Monitoring mode, values are fixed and measurements come from field instrumentation via Arduino.")

st.divider()

with st.expander("Digital Twin Governing Equations"):
    st.markdown(r"""
        ### Governing Reaction

        \[
        \mathrm{MO_2 + 4H^+ + e^- \rightarrow M^{2+} + 2H_2O}
        \]

        ### Kinetic Assumptions
        - Surface-controlled oxidative leaching
        - Pseudo-first-order behavior in early batch stage
        - Oxidant-assisted electron transfer
        - Apparent kinetics inferred from TDS evolution

        This reaction framework is consistent with transition metal oxide dissolution
        under acidic–oxidative conditions (LIB black mass leaching).
        """)



# =============================
# ARDUINO SETTINGS (CARD)
# =============================
if mode == "Online Monitoring (Arduino)":
    with st.container(border=True):
        st.subheader("Field Connectivity — Arduino")
        s1, s2, s3, s4, s5 = st.columns([1.2, 1.0, 1.2, 1.2, 1.6])
        with s1:
            com_port = st.text_input("COM Port", value="COM6")
        with s2:
            baud = st.selectbox("Baud", [9600, 57600, 115200], index=0)
        with s3:
            rt_batch = st.slider("Read Batch", 5, 60, RT_READ_BATCH_DEFAULT, 1)
        with s4:
            st.caption("Expected header:")
            st.code("Time_s,Temp_C,TDS_raw,TDS_comp,Turbidity_raw", language="text")
        with s5:
            connect_btn = st.button("Connect", use_container_width=True)
            disconnect_btn = st.button("Disconnect", use_container_width=True)

        if disconnect_btn:
            safe_close_serial()
            st.session_state.events.append("Arduino disconnected")

        if connect_btn:
            if serial is None:
                st.error("pyserial not installed. Run: pip install pyserial")
            else:
                ok = try_connect_serial(com_port, baud)
                if ok:
                    st.session_state.events.append(f"Arduino connected: {com_port} @ {baud}")
                else:
                    st.error("Arduino connection failed. Check COM port / cable / close Serial Monitor.")

# =============================
# ANOMALY SCENARIO (CARD)
# =============================
if mode == "Virtual Commissioning":
    with st.container(border=True):
        st.subheader("Scenario / Anomaly Configuration")
        a1, a2, a3, a4 = st.columns([1, 1.4, 1.2, 1.4])

        anomaly_on = a1.toggle("Enable Anomaly", value=False)
        anomaly_type = a2.selectbox(
            "Type",
            ["Sensor Spike", "Acid Underfeed", "Over-oxidation (H2O2)"],
            disabled=(not anomaly_on)
        )
        anomaly_severity = a3.slider(
            "Severity",
            0.0, 1.0, 0.5, 0.05,
            disabled=(not anomaly_on)
        )
        anomaly_time = a4.slider(
            "Start Time (s)",
            0, int(sim_time), int(sim_time * 0.35), 10,
            disabled=(not anomaly_on)
        )
else:
    anomaly_on = False
    anomaly_type = "Sensor Spike"
    anomaly_severity = 0.0
    anomaly_time = 0

# =============================
# RUN CONTROL (CARD)
# =============================
with st.container(border=True):
    st.subheader("Simulation / Monitoring Control")
    b1, b2, b3 = st.columns(3)
    run_btn = b1.button("Run", use_container_width=True)
    stop_btn = b2.button("Stop", use_container_width=True)
    reset_btn = b3.button("Reset", use_container_width=True)

    if reset_btn:
        st.session_state.running = False
        st.session_state.finalized = False
        st.session_state.hist_base = pd.DataFrame()
        st.session_state.hist_anom = pd.DataFrame()
        st.session_state.events = []
        st.session_state.rt_anom = False
        st.session_state.rt_anom_score = 0.0
        safe_close_serial()

    if run_btn:
        st.session_state.running = True
        st.session_state.finalized = False
        st.session_state.events.append(f"RUN → {mode}")

    if stop_btn:
        st.session_state.running = False
        st.session_state.finalized = True
        st.session_state.events.append("STOP pressed")
        if mode == "Online Monitoring (Arduino)":
            safe_close_serial()

st.divider()

# =============================
# MIDDLE: FLOWSHEET + GRAPHS
# =============================
mid_container = st.container(border=False)
with mid_container:
    left_mid, right_mid = st.columns([1.05, 1.95], gap="large")

    # ---- FLOWSHEET CARD ----
    with left_mid:
        with st.container(border=True):
            st.subheader("Process Flowsheet")
            if st.session_state.running and mode == "Virtual Commissioning" and anomaly_on:
                flow_state = "ANOMALY"
                anom_flag = True
            elif st.session_state.running and mode == "Online Monitoring (Arduino)" and st.session_state.rt_anom:
                flow_state = "ANOMALY"
                anom_flag = True
            elif st.session_state.running:
                flow_state = "RUNNING"
                anom_flag = False
            elif st.session_state.finalized:
                flow_state = "DONE"
                anom_flag = False
            else:
                flow_state = "IDLE"
                anom_flag = False

            st.graphviz_chart(make_flowsheet(flow_state, anomaly=anom_flag))

    # ---- TRENDS CARD ----
    with right_mid:
        with st.container(border=True):
            st.subheader("Trend Plots")
            g1 = st.empty()
            g2 = st.empty()
            g3 = st.empty()
            g4 = st.empty()
            g5 = st.empty() 


            if len(st.session_state.hist_base) == 0:
                dummy = pd.DataFrame({
                    "time_s": [0, 1],
                    "tds": [0, 0],
                    "turbidity": [0, 0],
                    "temp_C": [0, 0],
                })
                g1.plotly_chart(
                    plotly_line(dummy, "tds", "TDS vs Time", "ppm"),
                    use_container_width=True, config={"displayModeBar": False}
                )
                g2.plotly_chart(
                    plotly_line(dummy, "turbidity", "Turbidity vs Time", "ADC/NTU"),
                    use_container_width=True, config={"displayModeBar": False}
                )
                g3.plotly_chart(
                    plotly_line(dummy, "temp_C", "Temperature vs Time", "°C"),
                    use_container_width=True, config={"displayModeBar": False}
                )
            else:
                base_plot = st.session_state.hist_base.copy()
                anom_plot = (
                    st.session_state.hist_anom.copy()
                    if (mode == "Virtual Commissioning" and anomaly_on and len(st.session_state.hist_anom) > 0)
                    else None
                )

                df_fore = None
                if mode == "Virtual Commissioning" and len(base_plot) > 0:
                    last_state = base_plot.iloc[-1].to_dict()
                    df_fore = short_forecast(model, FEATURES, last_state, steps=6)

                g1.plotly_chart(
                    plotly_line(base_plot, "tds", "TDS vs Time", "ppm", anom_plot, df_fore),
                    use_container_width=True, config={"displayModeBar": False}
                )
                g2.plotly_chart(
                    plotly_line(base_plot, "turbidity", "Turbidity vs Time", "ADC/NTU", anom_plot, df_fore),
                    use_container_width=True, config={"displayModeBar": False}
                )
                g3.plotly_chart(
                    plotly_line(base_plot, "temp_C", "Temperature vs Time", "°C", anom_plot, df_fore),
                    use_container_width=True, config={"displayModeBar": False}
                )
                rate_df = compute_rate(base_plot, "tds")

                if rate_df is not None:
                    g4.plotly_chart(
                        plotly_line(
                            rate_df,
                            "dtds_dt",  
                            "TDS Dissolution Rate vs Time",
                            "d(TDS)/dt (ppm/s)"
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False}
                    )
                rec_df = compute_recovery_index(base_plot)

                if rec_df is not None:
                    g5.plotly_chart(
                        plotly_line(
                            rec_df,
                            "recovery_index",
                            "Relative Metal Recovery Index",
                            "Fractional Recovery (–)"
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False}
                    )



# =============================
# BOTTOM: EVENTS + RESULTS
# =============================
bottom_container = st.container(border=False)
with bottom_container:
    botL, botR = st.columns([1.2, 1.8], gap="large")

    # ---- EVENT LOG CARD ----
    with botL:
        with st.container(border=True):
            st.subheader("Event Log")
            box = st.container(border=False)
            with box:
                if len(st.session_state.events) == 0:
                    st.info("No events recorded.")
                else:
                    for e in st.session_state.events[-14:]:
                        st.write("•", e)

    # ---- RESULTS / KPIs CARD ----
with botR:
    with st.container(border=True):
        st.subheader("Key Performance Indicators")
        if len(st.session_state.hist_base) > 0:
            last = st.session_state.hist_base.iloc[-1]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Time (s)", f"{last['time_s']:.1f}")
            r2.metric("TDS (ppm)", f"{last['tds']:.1f}")
            r3.metric("Turbidity", f"{last['turbidity']:.1f}")
            r4.metric("Temperature (°C)", f"{last['temp_C']:.2f}")

            stable = stabilized_flag(st.session_state.hist_base)
            st.metric("Process State", "STABLE" if stable else "TRANSIENT")

            eta = estimate_time_to_plateau(st.session_state.hist_base)
            if eta is not None:
                st.metric("ETA to Plateau (rough)", f"{eta} s")
            k_app = estimate_k_app(st.session_state.hist_base)
            if k_app is not None:
                st.metric("Apparent Dissolution Rate Constant (kₐₚₚ)", f"{k_app:.3e} s⁻¹")
            else:
                st.caption("kₐₚₚ not identifiable (non-exponential TDS evolution)")


            if mode == "Online Monitoring (Arduino)":
                st.metric("Realtime Anomaly", "YES" if st.session_state.rt_anom else "NO")
                if st.session_state.rt_anom:
                    st.caption(f"Score: {st.session_state.rt_anom_score:.3f}")

                eta_rt = realtime_eta_to_stable(
                    st.session_state.hist_base, window=8, thr=8.0
                )
                if eta_rt is None:
                    st.metric("ETA to Stabilize", "Collecting…")
                else:
                    st.metric("ETA to Stabilize", f"{eta_rt} s")
                k_app = estimate_k_app(st.session_state.hist_base)
                if k_app is not None:
                    st.caption("kₐₚₚ not identifiable (non-exponential TDS evolution)")



            # ================= MODEL VALIDATION =================
            if model_metrics is not None:
                st.divider()
                st.subheader("Model Validation (Offline)")
                st.caption(
                    "Metrics computed using held-out experimental batch data. "
                    "Runtime predictions are not evaluated against live sensor noise."
                )


                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE – TDS Rate", f"{model_metrics['tds_rate']['rmse']:.3f}")
                m2.metric("RMSE – Turbidity Rate", f"{model_metrics['turbidity_rate']['rmse']:.3f}")
                m3.metric("RMSE – Recovery Rate", f"{model_metrics['recovery_rate']['rmse']:.3f}")

                m4, m5, m6 = st.columns(3)
                m4.metric("R² – TDS Rate", f"{model_metrics['tds_rate']['r2']:.3f}")
                m5.metric("R² – Turbidity Rate", f"{model_metrics['turbidity_rate']['r2']:.3f}")
                m6.metric("R² – Recovery Rate", f"{model_metrics['recovery_rate']['r2']:.3f}")

        else:
            st.info("Run a simulation or start monitoring to see KPIs.")


# ==================================================
# SIMULATION LOOP (Virtual Commissioning)
# ==================================================
if st.session_state.running and mode == "Virtual Commissioning":

    if len(st.session_state.hist_base) == 0:
        st.session_state.hist_base = pd.DataFrame([{
            "time_s": 0.0,
            "acid_M": float(acid_M),
            "h2o2_ml": float(h2o2_ml),
            "tds": float(tds0),
            "turbidity": float(turb0),
            "temp_C": float(temp0),
        }])

    # ✅ ENSURE INITIAL STATE EXISTS
    last = st.session_state.hist_base.iloc[-1].to_dict()


    if anomaly_on and len(st.session_state.hist_anom) == 0:
        st.session_state.hist_anom = st.session_state.hist_base.tail(1).copy()

    while st.session_state.running:

        base = st.session_state.hist_base.copy()
        last = base.iloc[-1].to_dict()

        r_tds, r_turb = predict_rates(model, FEATURES, last)
        r_tds = np.clip(r_tds, -50.0, 50.0)
        r_turb = np.clip(r_turb, -30.0, 30.0)


        nxt = last.copy()
        nxt["time_s"] += DT
        nxt["temp_C"] = temp_dynamics(last["temp_C"], h2o2_ml, DT)
        nxt["tds"] = max(0.0, float(nxt["tds"] + ALPHA_INTEGRATION * r_tds * DT))
        nxt["turbidity"] = turbidity_sensor_update(last["turbidity"], r_turb, DT, noise_scale=0.6)

        st.session_state.hist_base = pd.concat(
            [st.session_state.hist_base, pd.DataFrame([nxt])],
            ignore_index=True
        )
        st.session_state.hist_base = keep_last_n(st.session_state.hist_base)

        if anomaly_on:
            anom = st.session_state.hist_anom.copy()
            la = anom.iloc[-1].to_dict()

            r_tds_a, r_turb_a = predict_rates(model, FEATURES, la)
            r_tds_a = np.clip(r_tds_a, -50.0, 50.0)
            r_turb_a = np.clip(r_turb_a, -30.0, 30.0)

            if la["time_s"] >= anomaly_time:
                r_tds_a, r_turb_a = apply_anomaly_to_rates(r_tds_a, r_turb_a, anomaly_type, anomaly_severity)

            na = la.copy()
            na["time_s"] += DT
            na["temp_C"] = temp_dynamics(la["temp_C"], h2o2_ml, DT)
            na["tds"] = max(0.0, float(na["tds"] + ALPHA_INTEGRATION * r_tds_a * DT))
            na["turbidity"] = turbidity_sensor_update(la["turbidity"], r_turb_a, DT, noise_scale=1.2)

            st.session_state.hist_anom = pd.concat(
                [st.session_state.hist_anom, pd.DataFrame([na])],
                ignore_index=True
            )
            st.session_state.hist_anom = keep_last_n(st.session_state.hist_anom)

        if nxt["time_s"] >= sim_time:
            st.session_state.running = False
            st.session_state.finalized = True
            st.session_state.events.append("Simulation completed")
            break

        time.sleep(0.12)

    st.rerun()

# ==================================================
# REAL-TIME ARDUINO MODE
# ==================================================
if st.session_state.running and mode == "Online Monitoring (Arduino)":

    if serial is None:
        st.session_state.running = False
        st.session_state.finalized = True
        st.session_state.events.append("pyserial not installed → install pyserial")
        st.error("pyserial not installed. Run: pip install pyserial")
        st.rerun()

    if not st.session_state.serial_connected or st.session_state.ser is None:
        st.warning("Arduino not connected. Click CONNECT first.")
        time.sleep(0.2)
        st.rerun()

    ser = st.session_state.ser
    batch = int(rt_batch) if "rt_batch" in locals() else RT_READ_BATCH_DEFAULT

    new_rows = []
    for _ in range(batch):
        if not st.session_state.running:
            break

        try:
            line = ser.readline().decode(errors="ignore").strip()
        except SerialException as e:
            # FIX: port got denied mid-run
            st.session_state.events.append(f"Serial read failed: {e}")
            safe_close_serial()
            st.session_state.running = False
            st.session_state.finalized = True
            break
        except Exception:
            continue

        if not line:
            continue

        # skip header
        if "Time_s" in line and "Temp" in line:
            continue

        # skip garbage lines (common at start)
        head = line[:10]
        if any(ch not in "0123456789-., " for ch in head):
            continue

        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        if len(parts) < 5:
            continue

        try:
            t_s = float(parts[0])
            temp_meas = float(parts[1])
            tds_raw = float(parts[2])
            tds_comp = float(parts[3])
            turb_raw = float(parts[4])
        except Exception:
            continue

        # skip DS18B20 error reading
        if temp_meas <= -100:
            continue

        row = {
            "time_s": t_s,
            "acid_M": float(acid_M),
            "h2o2_ml": float(h2o2_ml),
            "tds": float(tds_comp),
            "turbidity": float(turb_raw),
            "temp_C": float(temp_meas),
            "tds_raw": float(tds_raw),
        }

        is_anom, score = predict_anomaly(row)
        st.session_state.rt_anom = bool(is_anom)
        st.session_state.rt_anom_score = float(score)

        if is_anom:
            st.session_state.events.append(f"⚠ anomaly detected (score={score:.2f})")

        new_rows.append(row)

    if len(new_rows) > 0:
        st.session_state.hist_base = pd.concat(
            [st.session_state.hist_base, pd.DataFrame(new_rows)],
            ignore_index=True
        )
        st.session_state.hist_base = keep_last_n(st.session_state.hist_base)

    time.sleep(RT_SLEEP)
    st.rerun()
