import streamlit as st
import wandb
import pandas as pd
import os
import sys
import io
import tempfile
from datetime import datetime
import time

# Define all utility functions directly in this file for now
# This avoids circular import issues while we restructure the code

def initialize_session_state():
    """Initialize all session state variables needed for the application."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "projects"
    
    if "selected_project" not in st.session_state:
        st.session_state.selected_project = None
    
    if "selected_run" not in st.session_state:
        st.session_state.selected_run = None
    
    if "selected_sweep" not in st.session_state:
        st.session_state.selected_sweep = None
    
    if "run_data" not in st.session_state:
        st.session_state.run_data = None
    
    if "sweep_data" not in st.session_state:
        st.session_state.sweep_data = None
    
    if "projects" not in st.session_state:
        st.session_state.projects = []

def authenticate_wandb(api_key):
    """Authenticate with Weights & Biases API."""
    try:
        wandb.login(key=api_key)
        st.session_state.api_key = api_key
        st.session_state.authenticated = True
        st.success("Authentication successful!")
        return True
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return False

def logout_wandb():
    """Log out from Weights & Biases."""
    st.session_state.authenticated = False
    st.session_state.api_key = ""
    st.session_state.current_page = "projects"
    st.session_state.selected_project = None
    st.session_state.selected_run = None
    st.session_state.selected_sweep = None
    st.session_state.run_data = None
    st.session_state.sweep_data = None
    st.session_state.projects = []
    wandb.logout()

def get_projects():
    """Get list of available projects."""
    try:
        api = wandb.Api()
        projects = []
        for project in api.projects():
            projects.append({
                "name": project.name,
                "entity": project.entity,
                "id": f"{project.entity}/{project.name}",
                "description": getattr(project, "description", ""),
                "created_at": getattr(project, "created_at", ""),
                "last_updated": getattr(project, "updated_at", "")
            })
        st.session_state.projects = projects
        return projects
    except Exception as e:
        st.error(f"Error fetching projects: {str(e)}")
        return []

def get_runs(project_id):
    """Get list of runs for a project."""
    try:
        api = wandb.Api()
        runs = []
        for run in api.runs(project_id):
            # Get the basic run information
            run_info = {
                "id": run.id,
                "name": run.name if run.name else run.id,
                "state": run.state,
                "created_at": run.created_at,
                "config": {},
                "summary": {}
            }
            
            # Extract config parameters
            for key, value in run.config.items():
                if not key.startswith('_'):
                    run_info["config"][key] = value
            
            # Extract summary metrics
            for key, value in run.summary.items():
                if not key.startswith('_'):
                    run_info["summary"][key] = value
            
            runs.append(run_info)
        
        return runs
    except Exception as e:
        st.error(f"Error fetching runs: {str(e)}")
        return []

def get_run_details(project_id, run_id):
    """Get detailed information for a specific run."""
    try:
        api = wandb.Api()
        run = api.run(f"{project_id}/{run_id}")
        
        # Get run history
        history_df = run.history()
        
        # Get run summary
        summary = {}
        for key, value in run.summary.items():
            if not key.startswith('_'):
                summary[key] = value
        
        # Get run config
        config = {}
        for key, value in run.config.items():
            if not key.startswith('_'):
                config[key] = value
        
        # Get run files/artifacts
        files = []
        for file in run.files():
            files.append({
                "name": file.name,
                "size": file.size,
                "updated_at": file.updatedAt
            })
        
        # Compile all run data
        run_data = {
            "id": run.id,
            "name": run.name if run.name else run.id,
            "state": run.state,
            "created_at": run.created_at,
            "entity": run.entity,
            "project": run.project,
            "summary": summary,
            "config": config,
            "history": history_df,
            "files": files,
            "url": run.url
        }
        
        return run_data
    except Exception as e:
        st.error(f"Error fetching run details: {str(e)}")
        return None

def download_run_artifact(project_id, run_id, file_name):
    """Download a specific artifact from a run."""
    try:
        api = wandb.Api()
        run = api.run(f"{project_id}/{run_id}")
        
        # Create a temporary file to store the download
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            run.file(file_name).download(root=os.path.dirname(tmp_file.name), replace=True)
            downloaded_path = os.path.join(os.path.dirname(tmp_file.name), file_name)
        
        # Read the file and return its contents
        with open(downloaded_path, 'rb') as f:
            data = f.read()
        
        # Clean up the temp file
        os.unlink(downloaded_path)
        
        return data
    except Exception as e:
        st.error(f"Error downloading artifact: {str(e)}")
        return None

def get_sweeps(project_id):
    """Get list of sweeps for a project."""
    try:
        api = wandb.Api()
        sweeps = []
        for sweep in api.sweeps(project_id):
            sweep_info = {
                "id": sweep.id,
                "name": sweep.name if hasattr(sweep, "name") and sweep.name else sweep.id,
                "state": sweep.state if hasattr(sweep, "state") else "unknown",
                "created_at": sweep.created_at if hasattr(sweep, "created_at") else "",
                "config": sweep.config if hasattr(sweep, "config") else {}
            }
            sweeps.append(sweep_info)
        
        return sweeps
    except Exception as e:
        st.error(f"Error fetching sweeps: {str(e)}")
        return []

def get_sweep_details(project_id, sweep_id):
    """Get detailed information for a specific sweep."""
    try:
        api = wandb.Api()
        sweep = api.sweep(f"{project_id}/{sweep_id}")
        
        # Get runs associated with this sweep
        runs = []
        for run in sweep.runs:
            run_info = {
                "id": run.id,
                "name": run.name if run.name else run.id,
                "state": run.state,
                "created_at": run.created_at,
                "config": {},
                "summary": {}
            }
            
            # Extract relevant config parameters
            for key, value in run.config.items():
                if not key.startswith('_'):
                    run_info["config"][key] = value
            
            # Extract relevant summary metrics
            for key, value in run.summary.items():
                if not key.startswith('_'):
                    run_info["summary"][key] = value
            
            runs.append(run_info)
        
        # Compile all sweep data
        sweep_data = {
            "id": sweep.id,
            "name": sweep.name if hasattr(sweep, "name") and sweep.name else sweep.id,
            "state": sweep.state if hasattr(sweep, "state") else "unknown",
            "created_at": sweep.created_at if hasattr(sweep, "created_at") else "",
            "config": sweep.config if hasattr(sweep, "config") else {},
            "runs": runs,
            "best_run": find_best_run(runs, sweep.config) if hasattr(sweep, "config") and runs else None
        }
        
        return sweep_data
    except Exception as e:
        st.error(f"Error fetching sweep details: {str(e)}")
        return None

def find_best_run(runs, sweep_config):
    """Find the best run in a sweep based on the metric defined in the sweep config."""
    if not runs:
        return None
    
    # Try to find the optimization metric from the sweep config
    optimization_metric = None
    optimization_goal = "minimize"  # default
    
    if "metric" in sweep_config:
        if isinstance(sweep_config["metric"], dict):
            if "name" in sweep_config["metric"]:
                optimization_metric = sweep_config["metric"]["name"]
            if "goal" in sweep_config["metric"]:
                optimization_goal = sweep_config["metric"]["goal"]
        elif isinstance(sweep_config["metric"], str):
            optimization_metric = sweep_config["metric"]
    
    # If no metric found, try to use a common one like accuracy or loss
    if not optimization_metric:
        for common_metric in ["accuracy", "val_accuracy", "loss", "val_loss"]:
            if any(common_metric in run["summary"] for run in runs):
                optimization_metric = common_metric
                optimization_goal = "maximize" if "accuracy" in common_metric else "minimize"
                break
    
    # If we still don't have a metric, we can't determine best run
    if not optimization_metric:
        return None
    
    # Find best run based on the metric
    best_run = None
    best_value = float('-inf') if optimization_goal == "maximize" else float('inf')
    
    for run in runs:
        if optimization_metric in run["summary"]:
            value = run["summary"][optimization_metric]
            
            # Skip if not a number
            if not isinstance(value, (int, float)):
                continue
                
            if optimization_goal == "maximize" and value > best_value:
                best_value = value
                best_run = run
            elif optimization_goal == "minimize" and value < best_value:
                best_value = value
                best_run = run
    
    return best_run

def export_to_csv(data, filename):
    """Convert data to CSV and provide a download link."""
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
    else:
        # Convert dict or list to DataFrame first
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
    
    # Create a download link
    b64 = io.StringIO()
    b64.write(csv)
    b64.seek(0)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_filename = f"{filename}_{current_time}.csv"
    
    return b64.getvalue(), download_filename

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
    
    # Refresh projects button
    if st.button("Refresh Projects"):
        with st.spinner("Fetching projects..."):
            get_projects()
    
    # Check if we have any projects
    if not st.session_state.projects:
        with st.spinner("Fetching projects..."):
            projects = get_projects()
        
        if not projects:
            st.warning("No projects found. Create a project in Weights & Biases first.")
            return
    
    # Display projects in a table
    projects_df = pd.DataFrame(st.session_state.projects)
    
    # Convert timestamps to more readable format
    if 'created_at' in projects_df.columns:
        projects_df['created_at'] = pd.to_datetime(projects_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
    if 'last_updated' in projects_df.columns:
        projects_df['last_updated'] = pd.to_datetime(projects_df['last_updated']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Display table with projects
    st.dataframe(
        projects_df[['name', 'entity', 'description', 'created_at', 'last_updated']],
        use_container_width=True,
        hide_index=True
    )
    
    # Selection for exploring a project
    st.subheader("Select a Project to Explore")
    
    # Get project names for the selectbox
    project_options = [f"{p['entity']}/{p['name']}" for p in st.session_state.projects]
    
    if project_options:
        selected_project_id = st.selectbox("Choose a project:", project_options)
        
        if st.button("Explore Project"):
            # Find the selected project in the list
            for project in st.session_state.projects:
                if f"{project['entity']}/{project['name']}" == selected_project_id:
                    st.session_state.selected_project = project
                    st.session_state.current_page = "runs"
                    st.rerun()

# Runs page
def render_runs_page():
    """Render the runs page for a selected project."""
    if not st.session_state.selected_project:
        st.warning("No project selected. Please select a project first.")
        if st.button("Back to Projects"):
            st.session_state.current_page = "projects"
            st.rerun()
        return
    
    project_id = st.session_state.selected_project["id"]
    st.header(f"Runs for {project_id}")
    
    # Button to refresh runs
    if st.button("Refresh Runs"):
        st.session_state.runs = get_runs(project_id)
    
    # Get runs for the selected project
    with st.spinner("Loading runs..."):
        runs = get_runs(project_id)
    
    if not runs:
        st.warning("No runs found for this project.")
        return
    
    # Prepare data for table display
    runs_data = []
    for run in runs:
        run_data = {
            "ID": run["id"],
            "Name": run["name"],
            "State": run["state"],
            "Created": pd.to_datetime(run["created_at"]).strftime('%Y-%m-%d %H:%M:%S') if "created_at" in run else "",
        }
        
        # Add some key metrics if available
        if "summary" in run:
            for key, value in run["summary"].items():
                if isinstance(value, (int, float)):
                    run_data[key] = value
        
        runs_data.append(run_data)
    
    # Display table with runs
    runs_df = pd.DataFrame(runs_data)
    st.dataframe(runs_df, use_container_width=True, hide_index=True)
    
    # Select a run for detailed view
    st.subheader("Select a Run for Detailed Analysis")
    
    # Get run names/IDs for the selectbox
    run_options = [f"{run['name']} ({run['id']})" for run in runs]
    
    if run_options:
        selected_run_option = st.selectbox("Choose a run:", run_options)
        selected_run_id = selected_run_option.split("(")[-1].split(")")[0]
        
        if st.button("View Run Details"):
            # Find the selected run in the list
            for run in runs:
                if run["id"] == selected_run_id:
                    st.session_state.selected_run = run
                    st.session_state.current_page = "run_details"
                    
                    # Get detailed run data
                    with st.spinner("Loading run details..."):
                        st.session_state.run_data = get_run_details(project_id, run["id"])
                    
                    st.rerun()

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
