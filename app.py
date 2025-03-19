import streamlit as st
import wandb
import os
from components import (
    render_sidebar,
    render_auth_page,
    render_projects_page,
    render_runs_page,
    render_run_details_page,
    render_sweeps_page,
    render_sweep_details_page
)
from components.quantum_assistant import quantum_assistant
from utils import initialize_session_state

# Page title and configuration
st.set_page_config(
    page_title="W&B Experiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
initialize_session_state()

# Main application
def main():
    # App header
    st.title("Quantum AI Experiment Dashboard")
    
    # Render sidebar for navigation
    render_sidebar()
    
    # Display the appropriate page based on navigation state
    if not st.session_state.authenticated:
        render_auth_page()
    else:
        if st.session_state.current_page == "quantum_assistant":
            quantum_assistant()
        elif st.session_state.current_page == "projects":
            render_projects_page()
        elif st.session_state.current_page == "runs":
            render_runs_page()
        elif st.session_state.current_page == "run_details":
            render_run_details_page()
        elif st.session_state.current_page == "sweeps":
            render_sweeps_page()
        elif st.session_state.current_page == "sweep_details":
            render_sweep_details_page()

if __name__ == "__main__":
    main()
