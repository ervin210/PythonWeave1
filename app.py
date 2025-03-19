import streamlit as st
import wandb
import os
from components.authentication import authenticate_wandb
from components.project_explorer import project_explorer
from components.run_details import run_details
from components.sweep_analyzer import sweep_analyzer
from components.artifact_manager import artifact_manager
from components.data_export import data_export

# Set page config
st.set_page_config(
    page_title="W&B Experiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Weights & Biases Experiment Dashboard")
st.markdown("""
This dashboard allows you to explore, visualize, and manage your Weights & Biases machine learning experiments.
Connect to your W&B account to get started.
""")

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'api' not in st.session_state:
    st.session_state.api = None
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None
if 'selected_run' not in st.session_state:
    st.session_state.selected_run = None
if 'selected_sweep' not in st.session_state:
    st.session_state.selected_sweep = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Projects"

# Authentication
if not st.session_state.authenticated:
    authenticate_wandb()
else:
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.api = None
        st.session_state.selected_project = None
        st.session_state.selected_run = None
        st.session_state.selected_sweep = None
        st.rerun()
    
    # Tab selection
    st.session_state.active_tab = st.sidebar.radio(
        "Select Section",
        ["Projects", "Run Details", "Sweep Analysis", "Artifacts", "Export Data"]
    )
    
    # Display content based on active tab
    if st.session_state.active_tab == "Projects":
        project_explorer()
        
    elif st.session_state.active_tab == "Run Details":
        run_details()
        
    elif st.session_state.active_tab == "Sweep Analysis":
        sweep_analyzer()
        
    elif st.session_state.active_tab == "Artifacts":
        artifact_manager()
        
    elif st.session_state.active_tab == "Export Data":
        data_export()
