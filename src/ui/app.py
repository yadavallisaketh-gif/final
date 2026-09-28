"""SIH26168 IDR Engine - Passenger Car MVP dashboard.

    streamlit run src/ui/app.py

Replays a 60 s GNSS blackout from a held-out IO-VNBD test drive through the
real navigation engine (src/engine.py, variant D: MotionNet + EKF + NHC + map)
and animates the result. Nothing is mocked: every estimated point comes from
src.ui.sim.replay_window, which reproduces src.evaluate window for window.
"""
from __future__ import annotations

import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)            # configs, data and model paths are relative to the repo root

import numpy as np                      # noqa: E402
import plotly.graph_objects as go       # noqa: E402
import streamlit as st                  # noqa: E402

from src.engine import MODE_DR          # noqa: E402
from src.ui import sim                  # noqa: E402

GREEN, BLUE, ORANGE, GREY = "#1a9e3f", "#1f5fd6", "#e07b00", "#9aa0a6"
BEFORE_S, AFTER_S, FRAME_S = 20.0, 15.0, 0.5          # display range and frame spacing (2 Hz)

st.set_page_config(page_title="IDR Engine - Passenger Car MVP", layout="wide")


# ---------------------------------------------------------------- cached pipeline
@st.cache_resource(show_spinner="Loading configuration and MotionNet ...")
def _pipeline():
    cfg = sim.load_cfg()
    return cfg, sim.load_model(cfg)


@st.cache_resource(show_spinner="Loading drive and training-drive road network ...")
def _drive(drive_id: str):
    cfg, _ = _pipeline()
    drive = sim.load_test_drive(cfg, drive_id)
    return drive, sim.road_network(cfg, drive), sim.blackout_windows(cfg, drive)


@st.cache_data(show_spinner="Running the EKF engine through the blackout ...")
def _replay(drive_id: str, t_start: float):
    cfg, model = _pipeline()
    drive, network, windows = _drive(drive_id)
    w = next(x for x in windows if math.isclose(x.t_start, t_start))
    r = sim.replay_window(cfg, drive, w, model, network)
    return r, sim.display_slice(r, BEFORE_S, AFTER_S, step=int(round(FRAME_S * 10)))


# ---------------------------------------------------------------- drawing
def _zoom(f, px: int = 650) -> float:
    span = max(np.ptp(np.r_[f.truth_x, f.x]), np.ptp(np.r_[f.truth_y, f.y]), 50.0) * 1.35
    lat = float(np.mean(f.truth_lat))
    return float(np.clip(math.log2(156543.0 * math.cos(math.radians(lat)) * px / span), 10, 18))


def map_figure(f, k: int, street: bool):
    now = f.iloc[: k + 1]
    rel = f.rel_t.to_numpy()
    lost = f.index[rel >= 0][0] if (rel >= 0).any() else None
    back = f.index[~f.denied & (rel > 0)]
    back = back[0] if len(back) else None
    fig = go.Figure()
    if street:
        S = go.Scattermap
        xy = lambda d, truth: dict(lat=d.truth_lat if truth else d.lat, lon=d.truth_lon if truth else d.lon)
    else:
        S = go.Scatter
        xy = lambda d, truth: dict(x=d.truth_x if truth else d.x, y=d.truth_y if truth else d.y)
    fig.add_trace(S(**xy(f, True), mode="lines", line=dict(color=GREY, width=2), opacity=0.35,
                    name="Route (ground truth, full replay)", hoverinfo="skip"))
    fig.add_trace(S(**xy(now, True), mode="lines", line=dict(color=GREEN, width=5), name="Ground truth"))
    fig.add_trace(S(**xy(now, False), mode="lines", line=dict(color=BLUE, width=3), name="IDR estimate (EKF)"))
    for idx, label, color in ((lost, "GNSS lost", "black"), (back, "GNSS back", GREY)):
        if idx is not None and idx <= k:
            fig.add_trace(S(**xy(f.iloc[[idx]], True), mode="markers+text", text=[label], textposition="top right",
                            marker=dict(size=11, color=color), name=label, showlegend=False))
    cur = f.iloc[[k]]
    fig.add_trace(S(**xy(cur, True), mode="markers", marker=dict(size=13, color=GREEN), showlegend=False,
                    hoverinfo="skip"))
    fig.add_trace(S(**xy(cur, False), mode="markers", marker=dict(size=13, color=BLUE), showlegend=False,
                    hoverinfo="skip"))
    layout = dict(height=620, margin=dict(l=0, r=0, t=0, b=0), uirevision="keep",
                  legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                              bgcolor="rgba(255,255,255,0.8)"))
    if street:
        layout["map"] = dict(style="open-street-map", zoom=_zoom(f),
                             center=dict(lat=float(np.mean(f.truth_lat)), lon=float(np.mean(f.truth_lon))))
    else:
        pad = 30
        layout |= dict(xaxis=dict(title="East (m)", range=[min(f.x.min(), f.truth_x.min()) - pad,
                                                             max(f.x.max(), f.truth_x.max()) + pad]),
                       yaxis=dict(title="North (m)", scaleanchor="x", scaleratio=1), plot_bgcolor="white")
    fig.update_layout(**layout)
    return fig


def speed_figure(f, k: int, duration: float):
    now = f.iloc[: k + 1]
    kmh = 3.6
    top = max(np.nanmax(f.truth_speed), np.nanmax(f.v_f), np.nanmax(f.motionnet_speed.fillna(0))) * kmh * 1.15 + 1
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=duration, fillcolor="#d93025", opacity=0.08, line_width=0,
                  annotation_text="GNSS denied", annotation_position="top left")
    fig.add_trace(go.Scatter(x=now.rel_t, y=now.truth_speed * kmh, name="Ground truth", line=dict(color=GREEN, width=3)))
    fig.add_trace(go.Scatter(x=now.rel_t, y=now.v_f * kmh, name="IDR v_f (EKF)", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=now.rel_t, y=now.motionnet_speed * kmh, name="MotionNet output", mode="markers",
                             marker=dict(color=ORANGE, size=4)))
    fig.add_vline(x=float(f.rel_t.iloc[k]), line_color="black", line_width=1)
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(title="km/h", range=[0, top]),
                      xaxis=dict(title="s from GNSS loss", range=[f.rel_t.min(), f.rel_t.max()]),
                      legend=dict(orientation="h", y=-0.35), plot_bgcolor="white", uirevision="keep")
    return fig


def telemetry(f, k: int, r):
    row = f.iloc[k]
    w = r.window
    denied_engine = row["mode"] == MODE_DR
    if denied_engine:
        badge = ("#d93025", "GNSS DENIED: PURE DR", "phone IMU + MotionNet + NHC + road proxy only")
    elif row.denied:
        badge = ("#e07b00", "GNSS LOST: CONFIRMING", f"no fix for < {cfg['filter']['gnss_timeout_s']} s yet")
    else:
        badge = ("#1a9e3f", "GNSS ACTIVE", "GNSS + INS fusion")
    st.markdown(
        f"<div style='background:{badge[0]};color:white;padding:14px 18px;border-radius:8px;"
        f"font-size:26px;font-weight:700;letter-spacing:0.5px'>{badge[1]}"
        f"<div style='font-size:13px;font-weight:400;opacity:0.9'>{badge[2]}</div></div>",
        unsafe_allow_html=True)
    rel = row.rel_t
    phase = (f"{rel:+.1f} s to GNSS loss" if rel < 0 else f"{rel:.1f} s into the {w.duration:.0f} s blackout"
             if rel <= w.duration else f"{rel - w.duration:.1f} s after GNSS returned")
    st.caption(f"Replay time: {phase}")

    st.markdown("**Live speed**")
    c1, c2, c3 = st.columns(3)
    c1.metric("IDR v_f, km/h", f"{row.v_f * 3.6:.1f}", help="Forward velocity state of the EKF.")
    mn = row.motionnet_speed
    c2.metric("MotionNet, km/h", f"{mn * 3.6:.1f}" if np.isfinite(mn) else "idle",
              help="MotionNet runs only while GNSS is denied; its output is the EKF's speed measurement.")
    c3.metric("Truth, km/h", f"{row.truth_speed * 3.6:.1f}")
    st.plotly_chart(speed_figure(f, k, w.duration), width="stretch", key=f"speed_{k}")

    st.markdown("**Position**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Error now, m", f"{row.pos_error:.1f}")
    c2.metric("No-GNSS dist, m", f"{row.blackout_dist:.0f}")
    if rel > 0 and row.blackout_dist > 1:
        err = row.pos_error if rel <= w.duration else r.metrics["endpoint_error_m"]
        c3.metric("Drift so far" if rel <= w.duration else "Window drift",
                  f"{100 * err / row.blackout_dist:.1f} %")
    else:
        c3.metric("Drift so far", "-")


# ---------------------------------------------------------------- page
st.title("SIH26168: IDR Engine - Passenger Car MVP")
st.caption("Replay of a recorded IO-VNBD test drive that was held out from training. When GNSS is cut, every "
           "blue point is computed by the EKF engine from the phone IMU alone. Green is the car's reference "
           "receiver, shown only for comparison and never given to the engine.")

cfg, model = _pipeline()
with st.sidebar:
    st.header("Replay")
    drive_id = st.selectbox("Test drive", sim.test_drives(cfg))
    drive, network, windows = _drive(drive_id)
    ver = sim.verified_results()
    ver = ver[ver.drive == drive_id]

    def _drift(w):
        v = ver[np.isclose(ver.t_start, w.t_start)].drift_percent
        return float(v.iloc[0]) if len(v) else float("nan")

    by_drift = sorted(windows, key=_drift)
    median_w = by_drift[len(by_drift) // 2]
    labels = {w.t_start: f"t = {w.t_start:.0f} s   (verified drift {_drift(w):.1f}%)"
                         + ("  - median window" if w is median_w else "") for w in windows}
    t_start = st.selectbox(f"60 s blackout ({len(windows)} on this drive)", [w.t_start for w in windows],
                           index=windows.index(median_w), format_func=labels.get)
    street = st.radio("Map", ["Local metres (offline)", "Street map"], index=0,
                      help="Street map tiles load from openstreetmap.org in your browser (needs internet). "
                           "The track data is identical in both views.") == "Street map"
    speed = st.select_slider("Replay speed", [1, 2, 5, 10], value=2, format_func=lambda v: f"{v}x")

    st.header("Kinematics (locked)")
    fc = cfg["filter"]
    st.markdown(f"- Vehicle: **passenger car**\n- Lever arm r_x: **{fc['lever_arm_x']:.1f} m** (phone ahead of "
                f"rear axle)\n- NHC sigma: **{fc['nhc_sigma']} m/s**\n- Pipeline: MotionNet + EKF + NHC + "
                f"road proxy")
    st.caption("Before the blackout the engine uses the car's GNSS receiver (sih_mvp profile). "
               "During the blackout no GNSS of any kind reaches the engine; this is asserted on every replay.")

r, f = _replay(drive_id, t_start)
n = len(f)

sel = (drive_id, t_start)
if st.session_state.get("sel") != sel:
    st.session_state["sel"] = sel
    st.session_state["frame"] = 0
if "pending_frame" in st.session_state:
    st.session_state["frame"] = st.session_state.pop("pending_frame")
st.session_state["frame"] = min(st.session_state.get("frame", 0), n - 1)

left, right = st.columns([3, 2], gap="large")
with left:
    b1, b2, _ = st.columns([1, 1, 4])
    play = b1.button("▶ Play", type="primary", width="stretch")
    if b2.button("⟲ Reset", width="stretch"):
        st.session_state["frame"] = 0
    st.slider("Replay frame", 0, n - 1, key="frame", label_visibility="collapsed")
    map_ph = st.empty()
with right:
    st.markdown(
        "<div style='border:2px solid #1f5fd6;border-radius:8px;padding:10px 16px;margin-bottom:12px'>"
        "<div style='font-size:22px;font-weight:700'>Verified 60s Median Drift: 9.9%</div>"
        "<div style='font-size:13px;opacity:0.8'>Median over the 12 held-out 60 s blackouts on test drives S1 + M. "
        "Single windows vary; 30 s (17.0%) and 120 s (21.3%) medians do not meet the 10% target.</div></div>",
        unsafe_allow_html=True)
    tel_ph = st.empty()


def draw(k: int):
    with map_ph.container():
        st.plotly_chart(map_figure(f, k, street), width="stretch", key=f"map_{k}")
    with tel_ph.container():
        telemetry(f, k, r)


if play:
    start = st.session_state["frame"] if st.session_state["frame"] < n - 1 else 0
    for k in range(start, n):
        t0 = time.perf_counter()
        draw(k)
        time.sleep(max(FRAME_S / speed - (time.perf_counter() - t0), 0.0))
    st.session_state["pending_frame"] = n - 1
    st.rerun()
else:
    draw(st.session_state["frame"])

m = r.metrics
st.caption(f"This window ({drive_id}, blackout {r.window.t_start:.0f}-{r.window.t_end:.0f} s): drift "
           f"{m['drift_percent']:.1f}% ({m['endpoint_error_m']:.1f} m endpoint error over {m['distance_m']:.0f} m), "
           f"mean position error {m['ate_m']:.1f} m, speed RMSE {m['speed_rmse_mps']:.2f} m/s, "
           f"GNSS updates inside the blackout: {r.gnss_updates_in_blackout}. The blue track is the scored filter "
           f"estimate; the jump when GNSS returns is the filter re-anchoring.")
