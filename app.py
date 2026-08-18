import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Mini Virtual Linac Demo", layout="wide")

st.title("Mini Virtual Linac Prototype")
st.caption("Educational prototype only — not for clinical treatment planning or delivery.")

st.sidebar.header("Virtual Linac Controls")

gantry = st.sidebar.slider("Gantry angle (degrees)", 0, 360, 0, step=5)
collimator = st.sidebar.slider("Collimator angle (degrees)", 0, 360, 0, step=5)

couch_vrt = st.sidebar.slider("Couch vertical shift (cm)", -5.0, 5.0, 0.0, step=0.5)
couch_lat = st.sidebar.slider("Couch lateral shift (cm)", -5.0, 5.0, 0.0, step=0.5)
couch_lon = st.sidebar.slider("Couch longitudinal shift (cm)", -5.0, 5.0, 0.0, step=0.5)

field_size = st.sidebar.slider("Field size (cm)", 2.0, 20.0, 10.0, step=1.0)
clearance_limit = st.sidebar.slider("Clearance warning threshold (cm)", 1.0, 8.0, 3.0, step=0.5)

distance_from_isocenter = np.sqrt(couch_lat**2 + couch_lon**2)
clearance_risk = distance_from_isocenter > clearance_limit

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(7, 7))

    room = plt.Circle((0, 0), 10, fill=False, linewidth=2)
    ax.add_patch(room)

    ax.scatter(0, 0, s=80)
    ax.text(0.3, 0.3, "Isocenter", fontsize=10)

    couch_x = couch_lat
    couch_y = couch_lon

    couch = plt.Rectangle((couch_x - 1.5, couch_y - 4), 3, 8, fill=False, linewidth=3)
    ax.add_patch(couch)
    ax.text(couch_x - 1.2, couch_y, "Couch", fontsize=11)

    patient = plt.Circle((couch_x, couch_y), 1.0, fill=False, linewidth=2)
    ax.add_patch(patient)
    ax.text(couch_x - 0.7, couch_y - 1.5, "Patient", fontsize=10)

    theta = np.deg2rad(gantry)
    arm_length = 8
    gx = arm_length * np.sin(theta)
    gy = arm_length * np.cos(theta)

    ax.plot([0, gx], [0, gy], linewidth=4)
    ax.scatter(gx, gy, s=180)
    ax.text(gx, gy, f"Gantry {gantry}°", fontsize=10)

    half = field_size / 10
    field = plt.Rectangle((-half, -half), 2 * half, 2 * half, fill=False, linestyle="--", linewidth=2)
    ax.add_patch(field)
    ax.text(-half, -half - 0.6, "Field light / treatment field", fontsize=10)

    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_aspect("equal")
    ax.set_title("Top-Down Educational Treatment-Room View")
    ax.set_xlabel("Lateral direction")
    ax.set_ylabel("Longitudinal direction")
    ax.grid(True)

    st.pyplot(fig)

with col2:
    st.subheader("OSCE Evidence Output")

    st.markdown(
        """
        Student action being demonstrated:

        - Adjust gantry angle
        - Apply couch LAT/LON/VRT shift
        - Check field size and setup relationship
        - Identify clearance or collision concern
        - Verbalize whether to proceed or stop
        """
    )

    st.write("Current simulated values:")

    st.code(
        f"""
Gantry: {gantry} degrees
Collimator: {collimator} degrees
Couch VRT: {couch_vrt} cm
Couch LAT: {couch_lat} cm
Couch LON: {couch_lon} cm
Field size: {field_size} cm
        """
    )

    if clearance_risk:
        st.error("Safety stop point: clearance/collision review required before proceeding.")
        decision = "STOP / ESCALATE"
    else:
        st.success("No clearance warning in this simplified model.")
        decision = "Proceed with supervised verification"

    st.subheader("Rater Interpretation")

    st.markdown(
        f"""
        Decision: {decision}

        Rubric domains demonstrated:

        - Safety: recognizes clearance/collision stop point
        - Technical execution: applies couch shift using correct axis
        - Communication: explains machine/couch movement
        - Clinical reasoning: decides whether to proceed or escalate
        """
    )

st.divider()

st.subheader("Prompt Engineering Teaching Point")

st.write(
    """
    This prototype began as a clinical education specification:
    create a simulated linac view with couch shifts, gantry movement,
    field visualization, clearance logic, and OSCE evidence output.

    An LLM can help draft the Python code, but RTTs, CMDs, physicists,
    and faculty must review the clinical accuracy and safety logic.
    """
)
