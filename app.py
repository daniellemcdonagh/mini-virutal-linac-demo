import math

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle


st.set_page_config(
    page_title="Educational Virtual Linac VLE",
    layout="wide",
)


CASE = {
    "title": "Pelvis setup verification",
    "setup_note": (
        "Align to tattoos, verify lasers at treatment isocentre, apply the couch shift "
        "from image guidance, then complete a final safety check before treatment."
    ),
    "expected_shift": {"VRT": 0.8, "LAT": -0.4, "LON": 1.2, "ROT": 0.0},
    "tolerance": {"linear": 0.2, "rot": 1.0},
}

ORIENTATIONS = {
    "HFS - head first supine": {"head_sign": -1, "label": "HFS"},
    "HFP - head first prone": {"head_sign": -1, "label": "HFP"},
    "FFS - feet first supine": {"head_sign": 1, "label": "FFS"},
    "FFP - feet first prone": {"head_sign": 1, "label": "FFP"},
}


def clamp(value, low, high):
    return max(low, min(high, value))


def init_couch_state():
    defaults = {
        "couch_vrt": 0.0,
        "couch_lat": 0.0,
        "couch_lon": 0.0,
        "couch_rot": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def nudge_couch(axis, delta):
    limits = {
        "couch_vrt": (-3.0, 3.0),
        "couch_lat": (-3.0, 3.0),
        "couch_lon": (-3.0, 3.0),
        "couch_rot": (-10.0, 10.0),
    }
    low, high = limits[axis]
    st.session_state[axis] = round(clamp(st.session_state[axis] + delta, low, high), 2)


def reset_couch():
    st.session_state.couch_vrt = 0.0
    st.session_state.couch_lat = 0.0
    st.session_state.couch_lon = 0.0
    st.session_state.couch_rot = 0.0


def apply_expected_shift():
    st.session_state.couch_vrt = CASE["expected_shift"]["VRT"]
    st.session_state.couch_lat = CASE["expected_shift"]["LAT"]
    st.session_state.couch_lon = CASE["expected_shift"]["LON"]
    st.session_state.couch_rot = CASE["expected_shift"]["ROT"]


def evaluate_shift(vrt, lat, lon, rot):
    expected = CASE["expected_shift"]
    linear_errors = {
        "VRT": abs(vrt - expected["VRT"]),
        "LAT": abs(lat - expected["LAT"]),
        "LON": abs(lon - expected["LON"]),
    }
    rot_error = abs(rot - expected["ROT"])

    linear_tol = CASE["tolerance"]["linear"]
    rot_tol = CASE["tolerance"]["rot"]

    all_correct = all(error <= linear_tol for error in linear_errors.values()) and rot_error <= rot_tol
    any_unsafe = (
        any(error > linear_tol * 3 for error in linear_errors.values())
        or rot_error > rot_tol * 3
        or any(abs(value) > 3.0 for value in [vrt, lat, lon])
        or abs(rot) > 5.0
    )

    if all_correct:
        state = "Correct"
        decision = "Proceed"
        message = "The couch shift matches the case expectation within tolerance."
    elif any_unsafe:
        state = "Unsafe"
        decision = "Stop/Escalate"
        message = "The entered shift is outside a safe educational threshold. Stop and escalate."
    else:
        state = "Partially correct"
        decision = "Recheck"
        message = "Some shift components are close, but the setup needs a deliberate recheck."

    return {
        "state": state,
        "decision": decision,
        "message": message,
        "linear_errors": linear_errors,
        "rot_error": rot_error,
    }


def decision_style(decision):
    if decision == "Proceed":
        return "success"
    if decision == "Recheck":
        return "warning"
    return "error"


def draw_room(
    gantry,
    collimator,
    couch_vrt,
    couch_lat,
    couch_lon,
    couch_rot,
    jaw_x1,
    jaw_x2,
    jaw_y1,
    jaw_y2,
    field_light,
    lasers,
    odi,
    clearance,
    orientation,
):
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    ax.set_facecolor("#fbfbf8")
    ax.set_aspect("equal")
    ax.set_xlim(-8.2, 8.2)
    ax.set_ylim(-4.8, 5.0)
    ax.axis("off")

    def shade(color, amount):
        import matplotlib.colors as mcolors

        rgb = np.array(mcolors.to_rgb(color))
        return tuple(clamp(channel + amount, 0, 1) for channel in rgb)

    def add_box(center, width, height, depth, color, edge="#51565c", zorder=3):
        x, y = center
        dx = depth * 0.45
        dy = depth * 0.28
        front = np.array(
            [
                [x - width / 2, y - height / 2],
                [x + width / 2, y - height / 2],
                [x + width / 2, y + height / 2],
                [x - width / 2, y + height / 2],
            ]
        )
        top = np.array(
            [
                [x - width / 2, y + height / 2],
                [x + width / 2, y + height / 2],
                [x + width / 2 + dx, y + height / 2 + dy],
                [x - width / 2 + dx, y + height / 2 + dy],
            ]
        )
        side = np.array(
            [
                [x + width / 2, y - height / 2],
                [x + width / 2 + dx, y - height / 2 + dy],
                [x + width / 2 + dx, y + height / 2 + dy],
                [x + width / 2, y + height / 2],
            ]
        )
        ax.add_patch(Polygon(top, closed=True, facecolor=shade(color, 0.08), edgecolor=edge, lw=0.8, zorder=zorder + 1))
        ax.add_patch(Polygon(side, closed=True, facecolor=shade(color, -0.08), edgecolor=edge, lw=0.8, zorder=zorder))
        ax.add_patch(Polygon(front, closed=True, facecolor=color, edgecolor=edge, lw=1.0, zorder=zorder + 2))

    # Browser-like training viewport.
    ax.add_patch(Rectangle((-7.85, -4.35), 15.7, 8.75, facecolor="#ffffff", edgecolor="#b8bdc4", lw=1.2, zorder=0))
    ax.add_patch(Rectangle((-7.85, 4.08), 15.7, 0.32, facecolor="#edf0f4", edgecolor="#b8bdc4", lw=0.8, zorder=1))
    for i, color in enumerate(["#ff5f57", "#ffbd2e", "#28c840"]):
        ax.add_patch(Circle((-7.58 + i * 0.22, 4.24), 0.055, facecolor=color, edgecolor="none", zorder=2))

    # Perspective floor grid resembling the reference simulator.
    horizon = 0.85
    floor_y = -4.2
    vanishing = np.array([-0.65, horizon])
    ax.add_patch(Polygon([(-6.5, floor_y), (6.95, floor_y), (2.55, horizon), (-2.65, horizon)], closed=True, facecolor="#fbfbfb", edgecolor="none", zorder=0))
    for x in np.linspace(-6.3, 6.7, 15):
        ax.plot([x, vanishing[0]], [floor_y, vanishing[1]], color="#c9cdd2", lw=0.8, zorder=1)
    for t in np.linspace(0.0, 1.0, 12):
        y = floor_y * (1 - t) + horizon * t
        left = -6.5 * (1 - t) + -2.65 * t
        right = 6.95 * (1 - t) + 2.55 * t
        ax.plot([left, right], [y, y], color="#c9cdd2", lw=0.8, zorder=1)

    # Side UI panels drawn inside the viewport to echo the Virtual Linac training environment.
    ax.add_patch(Rectangle((-7.72, -4.35), 1.95, 8.42, facecolor="#fbfbfb", edgecolor="#aeb4bd", lw=1.0, zorder=8))
    ax.add_patch(Rectangle((5.55, -4.35), 2.16, 8.42, facecolor="#fbfbfb", edgecolor="#aeb4bd", lw=1.0, zorder=8))
    ax.text(-6.74, 3.86, "View settings", ha="center", va="center", fontsize=8, weight="bold", color="#2e343b", zorder=10)
    for i, label in enumerate(["Clearance", "Laser", "ODI", "Couch top", "Gantry", "Field"]):
        y = 3.55 - i * 0.31
        enabled = {
            "Clearance": clearance,
            "Laser": lasers,
            "ODI": odi,
            "Couch top": True,
            "Gantry": True,
            "Field": field_light,
        }[label]
        ax.add_patch(Rectangle((-7.55, y - 0.075), 0.13, 0.13, facecolor="#1971c2" if enabled else "#ffffff", edgecolor="#8b949e", lw=0.8, zorder=10))
        ax.text(-7.34, y, f"{label} ON/OFF", va="center", fontsize=6.8, color="#2e343b", zorder=10)
    for i, label in enumerate(["Collimator", "Gantry", "Couch VRT", "Couch LAT", "Couch LON", "Couch ROT", "Jaws Y1", "Jaws X1"]):
        y = 1.15 - i * 0.55
        ax.text(-6.8, y + 0.18, label, ha="center", fontsize=6.8, weight="bold", color="#3f4650", zorder=10)
        ax.add_patch(Rectangle((-7.45, y - 0.05), 1.2, 0.16, facecolor="#ffffff", edgecolor="#d0d5da", lw=0.8, zorder=10))
        ax.add_patch(Rectangle((-6.9, y - 0.09), 0.12, 0.24, facecolor="#e9ecef", edgecolor="#c8cdd2", lw=0.6, zorder=11))
    ax.text(6.63, 3.82, ORIENTATIONS[orientation]["label"], ha="center", fontsize=8.5, weight="bold", color="#2e343b", zorder=10)
    for i, label in enumerate(["LON", "LAT", "VRT"]):
        y = 3.25 - i * 0.48
        ax.text(6.02, y, label, ha="left", fontsize=7, weight="bold", color="#3f4650", zorder=10)
        ax.add_patch(Rectangle((6.34, y - 0.08), 0.92, 0.15, facecolor="#ffffff", edgecolor="#d0d5da", lw=0.8, zorder=10))
        ax.add_patch(Rectangle((6.74, y - 0.13), 0.12, 0.25, facecolor="#e9ecef", edgecolor="#c8cdd2", lw=0.6, zorder=11))
    ax.text(6.63, 1.58, "External Bounds", ha="center", fontsize=6.8, weight="bold", color="#3f4650", zorder=10)
    for i in range(4):
        ax.add_patch(Rectangle((5.83, 1.28 - i * 0.18), 1.55, 0.14, facecolor="#ffffff", edgecolor="#d0d5da", lw=0.5, zorder=10))
    ax.add_patch(Rectangle((5.83, -3.9), 1.55, 1.55, facecolor="#101214", edgecolor="#3b4148", lw=0.8, zorder=10))
    for i in range(6):
        ax.add_patch(Rectangle((5.91, -2.55 - i * 0.22), 0.22, 0.13, facecolor="#f8f9fa", edgecolor="#8b949e", lw=0.45, zorder=11))
        ax.add_patch(Rectangle((7.17, -2.54 - i * 0.22), 0.12, 0.12, facecolor="#1971c2", edgecolor="#8b949e", lw=0.45, zorder=11))

    # Treatment isocentre is fixed in the patient setup area; the gantry head rotates around it.
    iso = np.array([0.75, -0.9])
    ax.add_patch(Circle(iso, 0.08, facecolor="#ff922b", edgecolor="#9c4a00", lw=0.8, zorder=7))
    ax.text(0.14, 0.05, "isocentre", fontsize=8, color="#4b5563", zorder=7)

    if clearance:
        ax.add_patch(
            Ellipse(
                iso,
                3.9,
                1.18,
                fill=False,
                lw=2.0,
                linestyle="--",
                edgecolor="#8a63d2",
                alpha=0.75,
                zorder=5,
            )
        )

    if lasers:
        ax.plot([iso[0] - 2.7, iso[0] + 3.0], [iso[1] - 0.08, iso[1] + 0.32], color="#37d35d", lw=1.5, alpha=0.9, zorder=6)
        ax.plot([iso[0] - 2.65, iso[0] + 2.55], [iso[1] + 0.33, iso[1] - 0.1], color="#37d35d", lw=1.3, alpha=0.75, zorder=6)
        ax.plot([iso[0], iso[0]], [-3.95, 1.95], color="#37d35d", lw=1.2, alpha=0.7, zorder=6)
        ax.plot([iso[0] - 0.1, iso[0] - 0.1], [iso[1] + 0.58, iso[1] + 3.2], color="#ff922b", lw=1.0, linestyle=":", alpha=0.9, zorder=7)

    if odi:
        ax.add_patch(Rectangle((iso[0] + 0.7, iso[1] + 0.75), 0.84, 0.72, facecolor="#111315", edgecolor="#6c757d", lw=1.2, zorder=4))
        ax.add_patch(Rectangle((iso[0] + 0.84, iso[1] + 0.88), 0.56, 0.46, facecolor="#08090a", edgecolor="#22282e", lw=0.8, zorder=5))
        ax.text(iso[0] + 1.12, iso[1] + 1.52, "ODI", ha="center", va="bottom", color="#5dd17a", fontsize=7, weight="bold", zorder=7)

    # Block-style linear accelerator: one continuous rectangular gantry and head.
    source = iso + np.array([0.0, 1.72])
    gantry_body = np.array(
        [
            [-4.6, -3.55],
            [-3.15, -3.55],
            [-3.15, 2.72],
            [source[0] + 0.9, 2.72],
            [source[0] + 0.9, source[1] - 0.62],
            [source[0] + 0.62, source[1] - 0.62],
            [source[0] + 0.62, source[1] - 1.0],
            [source[0] - 0.62, source[1] - 1.0],
            [source[0] - 0.62, source[1] - 0.62],
            [-2.0, source[1] - 0.62],
            [-2.0, -3.55],
        ]
    )
    ax.add_patch(Polygon(gantry_body, closed=True, facecolor="#6f7478", edgecolor="#343a40", lw=1.4, zorder=3))
    ax.add_patch(Polygon(gantry_body + np.array([0.28, 0.22]), closed=True, facecolor="#858a8f", edgecolor="none", alpha=0.28, zorder=2))
    ax.add_patch(Rectangle((-4.38, -3.25), 1.0, 0.46, facecolor="#555b60", edgecolor="#343a40", lw=0.8, zorder=4))
    ax.add_patch(Rectangle((-4.32, -2.55), 0.86, 4.55, facecolor="#7d8286", edgecolor="none", alpha=0.36, zorder=4))
    ax.add_patch(Rectangle((-3.02, 2.2), 3.96, 0.32, facecolor="#9ca1a5", edgecolor="none", alpha=0.45, zorder=4))
    ax.add_patch(Rectangle((source[0] - 0.5, source[1] - 0.9), 1.0, 0.28, facecolor="#3f454b", edgecolor="#2b3035", lw=0.8, zorder=6))
    ax.add_patch(Rectangle((source[0] - 0.34, source[1] - 1.03), 0.68, 0.13, facecolor="#202429", edgecolor="#15181b", lw=0.8, zorder=7))
    ax.text(source[0] + 0.72, source[1] - 0.86, "treatment head", fontsize=7.5, color="#343a40", zorder=9)

    if field_light:
        field_width = clamp((abs(jaw_x1) + abs(jaw_x2)) / 4.0, 0.5, 2.8)
        field_length = clamp((abs(jaw_y1) + abs(jaw_y2)) / 5.0, 0.5, 2.8)
        beam_dir = (iso - source) / np.linalg.norm(iso - source)
        perp = np.array([-beam_dir[1], beam_dir[0]])
        near = source + beam_dir * 0.45
        far = iso + beam_dir * 0.95
        corners = np.array(
            [
                near + perp * (field_width * 0.34),
                near - perp * (field_width * 0.34),
                far - perp * (field_length * 0.55),
                far + perp * (field_length * 0.55),
            ]
        )
        ax.add_patch(Polygon(corners, closed=True, facecolor="#ffd43b", edgecolor="#f08c00", alpha=0.32, zorder=2))

        col_rad = math.radians(collimator)
        field_x = (abs(jaw_x1) + abs(jaw_x2)) / 8.0
        field_y = (abs(jaw_y1) + abs(jaw_y2)) / 8.0
        rect = np.array(
            [
                [-field_x, -field_y],
                [field_x, -field_y],
                [field_x, field_y],
                [-field_x, field_y],
            ]
        )
        rot = np.array([[math.cos(col_rad), -math.sin(col_rad)], [math.sin(col_rad), math.cos(col_rad)]])
        projected = rect @ rot.T
        projected = projected + iso
        ax.add_patch(Polygon(projected, closed=True, facecolor="#ffe066", edgecolor="#e67700", alpha=0.45, zorder=5))

    # Couch translation and rotation are deliberately scaled for educational visibility.
    couch_center = iso + np.array([0.32 + couch_lat * 0.52, -0.24 + couch_lon * 0.28 + couch_vrt * 0.2])
    couch_angle = math.radians(couch_rot)
    couch_len = 4.35
    couch_wid = 0.86
    base = np.array(
        [
            [-couch_len / 2, -couch_wid / 2],
            [couch_len / 2, -couch_wid / 2],
            [couch_len / 2, couch_wid / 2],
            [-couch_len / 2, couch_wid / 2],
        ]
    )
    rot = np.array([[math.cos(couch_angle), -math.sin(couch_angle)], [math.sin(couch_angle), math.cos(couch_angle)]])
    couch_poly = base @ rot.T + couch_center
    top_offset = np.array([0.26, 0.18])
    ax.add_patch(Polygon(couch_poly + top_offset, closed=True, facecolor="#2c3035", edgecolor="#111315", lw=1.2, zorder=4))
    ax.add_patch(Polygon(couch_poly, closed=True, facecolor="#050607", edgecolor="#111315", lw=1.2, zorder=5))
    add_box((couch_center[0] + 0.28, couch_center[1] - 1.5), 2.15, 0.68, 0.75, "#4e5459", zorder=2)
    for i in range(5):
        ax.plot([couch_center[0] - 0.7, couch_center[0] + 1.25], [couch_center[1] - 1.57 - i * 0.12, couch_center[1] - 1.57 - i * 0.12], color="#343a40", lw=0.8, zorder=5)

    orient = ORIENTATIONS[orientation]
    patient_center = iso + np.array([couch_lat * 0.28, couch_lon * 0.12 + couch_vrt * 0.08])
    head_x = patient_center[0] + orient["head_sign"] * math.cos(couch_angle) * 1.0
    head_y = patient_center[1] + orient["head_sign"] * math.sin(couch_angle) * 1.0
    body_center = patient_center - np.array([math.cos(couch_angle), math.sin(couch_angle)]) * orient["head_sign"] * 0.15
    ax.add_patch(
        Ellipse(
            body_center,
            width=1.85,
            height=0.5,
            angle=couch_rot,
            facecolor="#b89a85",
            edgecolor="#755f52",
            lw=1.1,
            zorder=7,
        )
    )
    for offset in np.linspace(-0.65, 0.65, 6):
        rib = body_center + np.array([math.cos(couch_angle), math.sin(couch_angle)]) * offset
        ax.add_patch(Ellipse(rib, width=0.22, height=0.56, angle=couch_rot, facecolor="#9d826f", edgecolor="none", alpha=0.35, zorder=8))
    ax.add_patch(Circle((head_x, head_y), 0.24, facecolor="#b89a85", edgecolor="#755f52", lw=1.1, zorder=8))
    ax.plot([source[0], iso[0]], [source[1], iso[1]], color="#ff922b", lw=1.1, linestyle="--", alpha=0.85, zorder=8)
    ax.text(couch_center[0], couch_center[1] - 0.78, orient["label"], ha="center", va="top", fontsize=8, color="#404854", zorder=8)

    # Jaw readout as a compact aperture icon.
    jaw_left = -abs(jaw_x1) / 10.0
    jaw_right = abs(jaw_x2) / 10.0
    jaw_bottom = -abs(jaw_y1) / 10.0
    jaw_top = abs(jaw_y2) / 10.0
    ax.add_patch(Rectangle((source[0] - 0.34, source[1] - 0.95), 0.66, 0.58, facecolor="#15181b", edgecolor="#5d646b", lw=1.0, zorder=4))
    ax.add_patch(Rectangle((source[0] - 0.19 + jaw_left * 0.18, source[1] - 0.79 + jaw_bottom * 0.12), (jaw_right - jaw_left) * 0.18, (jaw_top - jaw_bottom) * 0.12, facecolor="#ffe066", edgecolor="#e67700", alpha=0.75, zorder=6))

    ax.text(-3.1, -4.05, f"Gantry {gantry:.0f} deg   Collimator {collimator:.0f} deg   Couch VRT {couch_vrt:+.1f} cm   LAT {couch_lat:+.1f} cm   LON {couch_lon:+.1f} cm", fontsize=9, color="#404854", zorder=12)
    return fig


def evidence_panel(evaluation, lasers, field_light, odi, clearance, orientation):
    expected = CASE["expected_shift"]
    errors = evaluation["linear_errors"]
    rot_error = evaluation["rot_error"]
    linear_tol = CASE["tolerance"]["linear"]
    rot_tol = CASE["tolerance"]["rot"]

    safety_score = 0
    safety_notes = []
    if lasers:
        safety_score += 1
        safety_notes.append("Lasers enabled")
    else:
        safety_notes.append("Lasers off")
    if clearance:
        safety_score += 1
        safety_notes.append("Clearance boundary checked")
    else:
        safety_notes.append("No clearance boundary")
    if evaluation["decision"] != "Stop/Escalate":
        safety_score += 1
        safety_notes.append("Shift not in unsafe range")
    else:
        safety_notes.append("Unsafe shift range")

    technical_score = sum(error <= linear_tol for error in errors.values()) + int(rot_error <= rot_tol)
    communication_score = 2 if field_light and odi else 1 if field_light or odi else 0
    reasoning_score = 2 if evaluation["state"] == "Correct" else 1 if evaluation["state"] == "Partially correct" else 0

    return [
        {
            "domain": "Safety",
            "score": f"{safety_score}/3",
            "evidence": "; ".join(safety_notes),
        },
        {
            "domain": "Communication",
            "score": f"{communication_score}/2",
            "evidence": "Visible setup aids support a verbal check-back." if communication_score else "No setup aid selected for check-back.",
        },
        {
            "domain": "Technical execution",
            "score": f"{technical_score}/4",
            "evidence": (
                f"Expected VRT {expected['VRT']:+.1f}, LAT {expected['LAT']:+.1f}, "
                f"LON {expected['LON']:+.1f}, ROT {expected['ROT']:+.1f}; "
                f"errors are VRT {errors['VRT']:.1f}, LAT {errors['LAT']:.1f}, "
                f"LON {errors['LON']:.1f}, ROT {rot_error:.1f}."
            ),
        },
        {
            "domain": "Clinical reasoning",
            "score": f"{reasoning_score}/2",
            "evidence": f"Current decision is {evaluation['decision']} for a {evaluation['state'].lower()} setup.",
        },
        {
            "domain": "Orientation awareness",
            "score": "1/1",
            "evidence": f"Patient orientation selected: {orientation}.",
        },
    ]


st.title("Educational Virtual Linac VLE")
st.caption("Simplified radiation therapy learning environment. Educational only - not for clinical use.")

init_couch_state()

with st.sidebar:
    st.header("Linac Controls")
    gantry = st.slider("Gantry angle (deg)", 0, 359, 0, 1)
    collimator = st.slider("Collimator angle (deg)", 0, 359, 0, 1)

    st.subheader("Table Motion")
    st.caption("Use the buttons to nudge the couch, or fine tune with the sliders below.")
    lon_back, vrt_up, lon_forward = st.columns(3)
    with lon_back:
        st.button("LON -", use_container_width=True, on_click=nudge_couch, args=("couch_lon", -0.1))
    with vrt_up:
        st.button("VRT +", use_container_width=True, on_click=nudge_couch, args=("couch_vrt", 0.1))
    with lon_forward:
        st.button("LON +", use_container_width=True, on_click=nudge_couch, args=("couch_lon", 0.1))

    lat_left, vrt_down, lat_right = st.columns(3)
    with lat_left:
        st.button("LAT -", use_container_width=True, on_click=nudge_couch, args=("couch_lat", -0.1))
    with vrt_down:
        st.button("VRT -", use_container_width=True, on_click=nudge_couch, args=("couch_vrt", -0.1))
    with lat_right:
        st.button("LAT +", use_container_width=True, on_click=nudge_couch, args=("couch_lat", 0.1))

    rot_left, reset_btn, rot_right = st.columns(3)
    with rot_left:
        st.button("ROT -", use_container_width=True, on_click=nudge_couch, args=("couch_rot", -0.5))
    with reset_btn:
        st.button("Reset", use_container_width=True, on_click=reset_couch)
    with rot_right:
        st.button("ROT +", use_container_width=True, on_click=nudge_couch, args=("couch_rot", 0.5))
    st.button("Apply expected case shift", use_container_width=True, on_click=apply_expected_shift)

    st.subheader("Couch Shift")
    couch_vrt = st.slider("Couch VRT (cm)", -3.0, 3.0, step=0.1, key="couch_vrt")
    couch_lat = st.slider("Couch LAT (cm)", -3.0, 3.0, step=0.1, key="couch_lat")
    couch_lon = st.slider("Couch LON (cm)", -3.0, 3.0, step=0.1, key="couch_lon")
    couch_rot = st.slider("Couch ROT (deg)", -10.0, 10.0, step=0.5, key="couch_rot")

    st.subheader("Jaw Settings")
    jaw_x1 = st.slider("Jaw X1 (cm)", -20.0, 0.0, -5.0, 0.5)
    jaw_x2 = st.slider("Jaw X2 (cm)", 0.0, 20.0, 5.0, 0.5)
    jaw_y1 = st.slider("Jaw Y1 (cm)", -20.0, 0.0, -5.0, 0.5)
    jaw_y2 = st.slider("Jaw Y2 (cm)", 0.0, 20.0, 5.0, 0.5)

    st.subheader("Room Aids")
    field_light = st.checkbox("Field light", value=True)
    lasers = st.checkbox("Lasers", value=True)
    odi = st.checkbox("ODI", value=True)
    clearance = st.checkbox("Clearance cylinder", value=True)
    orientation = st.selectbox("Patient orientation", list(ORIENTATIONS.keys()), index=0)

st.info(
    f"**Clinical case: {CASE['title']}**\n\n"
    f"**Setup note:** {CASE['setup_note']}\n\n"
    f"**Expected shift:** VRT {CASE['expected_shift']['VRT']:+.1f} cm, "
    f"LAT {CASE['expected_shift']['LAT']:+.1f} cm, "
    f"LON {CASE['expected_shift']['LON']:+.1f} cm, "
    f"ROT {CASE['expected_shift']['ROT']:+.1f} deg. "
    f"**Tolerance:** +/-{CASE['tolerance']['linear']:.1f} cm and +/-{CASE['tolerance']['rot']:.1f} deg."
)

evaluation = evaluate_shift(couch_vrt, couch_lat, couch_lon, couch_rot)

room_col, status_col = st.columns([2.2, 1.0], gap="large")

with room_col:
    st.subheader("Treatment Room Visualization")
    fig = draw_room(
        gantry,
        collimator,
        couch_vrt,
        couch_lat,
        couch_lon,
        couch_rot,
        jaw_x1,
        jaw_x2,
        jaw_y1,
        jaw_y2,
        field_light,
        lasers,
        odi,
        clearance,
        orientation,
    )
    st.pyplot(fig, clear_figure=True)

with status_col:
    st.subheader("Shift Check")
    if evaluation["state"] == "Correct":
        st.success(evaluation["message"])
    elif evaluation["state"] == "Partially correct":
        st.warning(evaluation["message"])
    else:
        st.error(evaluation["message"])

    st.metric("State", evaluation["state"])
    st.write(
        {
            "VRT error (cm)": round(evaluation["linear_errors"]["VRT"], 2),
            "LAT error (cm)": round(evaluation["linear_errors"]["LAT"], 2),
            "LON error (cm)": round(evaluation["linear_errors"]["LON"], 2),
            "ROT error (deg)": round(evaluation["rot_error"], 2),
        }
    )

    st.subheader("Final Decision")
    style = decision_style(evaluation["decision"])
    decision_text = f"**{evaluation['decision']}**"
    if style == "success":
        st.success(decision_text)
    elif style == "warning":
        st.warning(decision_text)
    else:
        st.error(decision_text)

st.subheader("OSCE Evidence Panel")
evidence = evidence_panel(evaluation, lasers, field_light, odi, clearance, orientation)
cols = st.columns(len(evidence))
for col, item in zip(cols, evidence):
    with col:
        st.markdown(f"**{item['domain']}**")
        st.metric("Evidence score", item["score"])
        st.caption(item["evidence"])

with st.expander("Educational boundaries"):
    st.write(
        "This prototype simplifies geometry, scale, collision checking, and clinical workflow. "
        "It is designed for classroom discussion and OSCE-style practice only. It must not be "
        "used to plan, approve, verify, or deliver patient treatment."
    )
