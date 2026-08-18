import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Arc, Circle, Ellipse, Polygon, Rectangle


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
    "HFS - head first supine": {"head_sign": 1, "label": "HFS"},
    "HFP - head first prone": {"head_sign": 1, "label": "HFP"},
    "FFS - feet first supine": {"head_sign": -1, "label": "FFS"},
    "FFP - feet first prone": {"head_sign": -1, "label": "FFP"},
}


def clamp(value, low, high):
    return max(low, min(high, value))


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
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ax.set_facecolor("#f6f7f9")
    ax.set_aspect("equal")
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(-4.5, 5.2)
    ax.axis("off")

    # Room floor and isocentre reference.
    ax.add_patch(Rectangle((-7.0, -4.5), 14.0, 9.7, facecolor="#f6f7f9", edgecolor="none"))
    ax.plot([-6.4, 6.4], [0, 0], color="#d6dbe1", lw=1.2)
    ax.plot([0, 0], [-4.0, 4.8], color="#d6dbe1", lw=1.2)
    ax.add_patch(Circle((0, 0), 0.08, facecolor="#2f3542", edgecolor="none", zorder=6))
    ax.text(0.15, 0.12, "ISO", fontsize=9, color="#404854")

    if clearance:
        ax.add_patch(
            Circle(
                (0, 0),
                2.15,
                fill=False,
                lw=2.0,
                linestyle="--",
                edgecolor="#8a5cf6",
                alpha=0.85,
                zorder=1,
            )
        )
        ax.text(1.55, 1.62, "clearance", fontsize=9, color="#6d4bd8")

    if lasers:
        ax.plot([-6.4, 6.4], [0, 0], color="#e03131", lw=1.6, alpha=0.78, zorder=3)
        ax.plot([0, 0], [-3.9, 4.7], color="#e03131", lw=1.6, alpha=0.78, zorder=3)
        ax.plot([-2.8, 2.8], [-2.8, 2.8], color="#e03131", lw=0.9, alpha=0.38, zorder=2)

    if odi:
        ax.add_patch(Rectangle((4.85, -3.7), 1.25, 0.7, facecolor="#20242a", edgecolor="#4b5563", lw=1.2))
        ax.text(5.48, -3.42, "ODI", ha="center", va="center", color="#b7f7d7", fontsize=10, weight="bold")
        ax.text(5.48, -3.75, f"{100 + couch_vrt:05.1f} cm", ha="center", va="top", color="#404854", fontsize=8)

    # Gantry head and arm.
    gantry_rad = math.radians(90 - gantry)
    source_radius = 3.15
    source = np.array([math.cos(gantry_rad) * source_radius, math.sin(gantry_rad) * source_radius])
    arm_inner = np.array([math.cos(gantry_rad) * 0.55, math.sin(gantry_rad) * 0.55])
    ax.plot([arm_inner[0], source[0]], [arm_inner[1], source[1]], color="#45515f", lw=14, solid_capstyle="round", zorder=2)
    ax.add_patch(Circle((0, 0), 0.42, facecolor="#7c8794", edgecolor="#45515f", lw=1.5, zorder=4))
    ax.add_patch(
        Ellipse(
            source,
            width=1.25,
            height=0.82,
            angle=-gantry,
            facecolor="#c9d0d8",
            edgecolor="#4b5563",
            lw=1.6,
            zorder=4,
        )
    )
    ax.add_patch(
        Ellipse(
            source - np.array([math.cos(gantry_rad), math.sin(gantry_rad)]) * 0.35,
            width=0.62,
            height=0.34,
            angle=-gantry,
            facecolor="#5b6470",
            edgecolor="#303742",
            lw=1.0,
            zorder=5,
        )
    )
    ax.add_patch(Arc((0, 0), 6.35, 6.35, theta1=0, theta2=360, color="#b7bec8", lw=1.6, alpha=0.85, zorder=1))

    if field_light:
        field_width = clamp((abs(jaw_x1) + abs(jaw_x2)) / 4.0, 0.5, 2.8)
        field_length = clamp((abs(jaw_y1) + abs(jaw_y2)) / 5.0, 0.5, 2.8)
        beam_dir = -source / np.linalg.norm(source)
        perp = np.array([-beam_dir[1], beam_dir[0]])
        near = source + beam_dir * 0.42
        far = np.array([0.0, 0.0]) + beam_dir * 0.75
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
        ax.add_patch(Polygon(projected, closed=True, facecolor="#ffe066", edgecolor="#e67700", alpha=0.45, zorder=5))

    # Couch translation and rotation are deliberately scaled for educational visibility.
    couch_center = np.array([couch_lat * 0.45, -1.6 + couch_lon * 0.28 + couch_vrt * 0.18])
    couch_angle = math.radians(couch_rot)
    couch_len = 5.5
    couch_wid = 1.02
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
    ax.add_patch(Polygon(couch_poly, closed=True, facecolor="#dce3ea", edgecolor="#53606e", lw=1.7, zorder=3))
    ax.add_patch(Rectangle((couch_center[0] - 0.48, -3.55), 0.96, 1.7, facecolor="#b2bbc5", edgecolor="#67717d", lw=1.2, zorder=2))
    ax.add_patch(Rectangle((couch_center[0] - 0.78, -3.95), 1.56, 0.38, facecolor="#8f99a5", edgecolor="#67717d", lw=1.0, zorder=2))

    orient = ORIENTATIONS[orientation]
    head_x = couch_center[0] + orient["head_sign"] * math.cos(couch_angle) * 1.48
    head_y = couch_center[1] + orient["head_sign"] * math.sin(couch_angle) * 1.48
    body_center = couch_center - np.array([math.cos(couch_angle), math.sin(couch_angle)]) * orient["head_sign"] * 0.15
    ax.add_patch(
        Ellipse(
            body_center,
            width=2.55,
            height=0.58,
            angle=couch_rot,
            facecolor="#f1c7a8",
            edgecolor="#a36f55",
            lw=1.1,
            zorder=5,
        )
    )
    ax.add_patch(Circle((head_x, head_y), 0.32, facecolor="#f1c7a8", edgecolor="#a36f55", lw=1.1, zorder=6))
    ax.text(couch_center[0], couch_center[1] - 0.9, orient["label"], ha="center", va="top", fontsize=9, color="#404854")

    # Jaw readout as a compact aperture icon.
    jaw_left = -abs(jaw_x1) / 10.0
    jaw_right = abs(jaw_x2) / 10.0
    jaw_bottom = -abs(jaw_y1) / 10.0
    jaw_top = abs(jaw_y2) / 10.0
    ax.add_patch(Rectangle((-6.5, 3.35), 2.1, 1.35, facecolor="#ffffff", edgecolor="#c7ced6", lw=1.0, zorder=2))
    ax.add_patch(Rectangle((-5.82 + jaw_left, 3.75 + jaw_bottom), jaw_right - jaw_left, jaw_top - jaw_bottom, facecolor="#ffe066", edgecolor="#e67700", alpha=0.75, zorder=3))
    ax.text(-5.45, 4.52, "Jaw aperture", ha="center", va="center", fontsize=9, color="#404854")
    ax.text(-5.45, 3.48, f"X {jaw_x1:.1f}/{jaw_x2:.1f}  Y {jaw_y1:.1f}/{jaw_y2:.1f}", ha="center", va="center", fontsize=8, color="#606b78")

    ax.text(-6.55, -4.13, f"Gantry {gantry:.0f} deg  |  Collimator {collimator:.0f} deg  |  Couch ROT {couch_rot:.1f} deg", fontsize=9, color="#404854")
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

with st.sidebar:
    st.header("Linac Controls")
    gantry = st.slider("Gantry angle (deg)", 0, 359, 0, 1)
    collimator = st.slider("Collimator angle (deg)", 0, 359, 0, 1)

    st.subheader("Couch Shift")
    couch_vrt = st.slider("Couch VRT (cm)", -3.0, 3.0, 0.0, 0.1)
    couch_lat = st.slider("Couch LAT (cm)", -3.0, 3.0, 0.0, 0.1)
    couch_lon = st.slider("Couch LON (cm)", -3.0, 3.0, 0.0, 0.1)
    couch_rot = st.slider("Couch ROT (deg)", -10.0, 10.0, 0.0, 0.5)

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
