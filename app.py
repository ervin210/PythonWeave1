import streamlit as st
import wandb
import os
import sys

# Import all utility functions directly from the utils.py file
from utils import *

# For clarity's sake, let's also list the main functions we're using:
# - initialize_session_state
# - authenticate_wandb
# - logout_wandb
# - get_projects
# - get_runs
# - get_run_details 
# - get_sweeps
# - get_sweep_details

# Import the quantum_assistant function
from components.quantum_assistant import quantum_assistant

# Page title and configuration
st.set_page_config(
    page_title="W&B Experiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
initialize_session_state()

# Navigation sidebar
def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("Navigation")
    
    if st.session_state.authenticated:
        # Quantum AI Assistant button
        if st.sidebar.button("🧠 Quantum AI Assistant", use_container_width=True):
            st.session_state.current_page = "quantum_assistant"
            
        # Project management buttons
        if st.sidebar.button("📋 Projects", use_container_width=True):
            st.session_state.current_page = "projects"
            st.session_state.selected_project = None
            st.session_state.selected_run = None
            st.session_state.selected_sweep = None
        
        if st.session_state.selected_project:
            project_id = st.session_state.selected_project["id"]
            if st.sidebar.button(f"🏃 Runs in {project_id}", use_container_width=True):
                st.session_state.current_page = "runs"
                st.session_state.selected_run = None
                st.session_state.selected_sweep = None
            
            if st.sidebar.button(f"🧹 Sweeps in {project_id}", use_container_width=True):
                st.session_state.current_page = "sweeps"
                st.session_state.selected_run = None
                st.session_state.selected_sweep = None
        
        st.sidebar.divider()
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout_wandb()
            st.rerun()
    
    st.sidebar.divider()
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """
        This dashboard helps you explore your Weights & Biases experiments. 
        Browse projects, runs, and sweeps, visualize metrics, and download artifacts.
        """
    )

# Authentication page
def render_auth_page():
    """Render the authentication page."""
    st.header("Connect to Weights & Biases")
    
    with st.form("auth_form"):
        api_key = st.text_input("W&B API Key", type="password", help="Enter your W&B API key to authenticate.")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            with st.spinner("Authenticating..."):
                if authenticate_wandb(api_key):
                    # Fetch projects after successful authentication
                    with st.spinner("Fetching projects..."):
                        get_projects()
                    st.rerun()
    
    st.markdown(
        """
        ### How to get your API key
        
        1. Go to [https://wandb.ai/settings](https://wandb.ai/settings)
        2. Find the API Keys section
        3. Copy your existing key or create a new one
        4. Paste it in the field above
        
        Your API key is stored only in your session and is used to access your W&B account.
        """
    )

# Projects page
def render_projects_page():
    """Render the projects page."""
    st.header("Your W&B Projects")
    
    # Just placeholder for now
    st.info("Projects would be displayed here.")

# Runs page
def render_runs_page():
    """Render the runs page for a selected project."""
    st.header("Your Project Runs")
    
    # Just placeholder for now
    st.info("Runs would be displayed here.")

# Run details page
def render_run_details_page():
    """Render detailed information for a selected run."""
    st.header("Run Details")
    
    # Just placeholder for now
    st.info("Run details would be displayed here.")

# Sweeps page
def render_sweeps_page():
    """Render the sweeps page for a selected project."""
    st.header("Your Project Sweeps")
    
    # Just placeholder for now
    st.info("Sweeps would be displayed here.")

# Sweep details page
def render_sweep_details_page():
    """Render detailed information for a selected sweep."""
    st.header("Sweep Details")
    
    # Just placeholder for now
    st.info("Sweep details would be displayed here.")

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
