import streamlit as st
import wandb
import os
import sys

# Import utility functions directly 
from utils import initialize_session_state

# Import components directly
from components import (
    quantum_assistant,
    artifact_manager,
    authenticate_wandb, 
    data_export,
    project_explorer,
    run_details,
    sweep_analyzer
)

# Import components functions directly from components.py
sys.path.append(".")
import components as comp

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
    comp.render_sidebar()
    
    # Display the appropriate page based on navigation state
    if not st.session_state.authenticated:
        comp.render_auth_page()
    else:
        if st.session_state.current_page == "quantum_assistant":
            quantum_assistant()
        elif st.session_state.current_page == "projects":
            comp.render_projects_page()
        elif st.session_state.current_page == "runs":
            comp.render_runs_page()
        elif st.session_state.current_page == "run_details":
            comp.render_run_details_page()
        elif st.session_state.current_page == "sweeps":
            comp.render_sweeps_page()
        elif st.session_state.current_page == "sweep_details":
            comp.render_sweep_details_page()

if __name__ == "__main__":
    main()
