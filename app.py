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
    if not st.session_state.selected_project or not st.session_state.selected_run:
        st.warning("No run selected. Please select a project and run first.")
        if st.button("Back to Projects"):
            st.session_state.current_page = "projects"
            st.rerun()
        return
    
    project_id = st.session_state.selected_project["id"]
    run_id = st.session_state.selected_run["id"]
    run_name = st.session_state.selected_run["name"]
    
    st.header(f"Run Details: {run_name}")
    
    # Check if we have run data
    if not st.session_state.run_data:
        with st.spinner("Loading run details..."):
            st.session_state.run_data = get_run_details(project_id, run_id)
            
    run_data = st.session_state.run_data
    if not run_data:
        st.error("Failed to load run details. Please try again.")
        if st.button("Back to Runs"):
            st.session_state.current_page = "runs"
            st.rerun()
        return
    
    # Create tabs for different aspects of the run
    overview_tab, config_tab, metrics_tab, files_tab = st.tabs([
        "Overview", "Configuration", "Metrics & Visualizations", "Files & Artifacts"
    ])
    
    with overview_tab:
        st.subheader("Run Information")
        
        # Display basic run info in columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Run ID:** {run_data['id']}")
            st.markdown(f"**Name:** {run_data['name']}")
            st.markdown(f"**State:** {run_data['state']}")
            st.markdown(f"**Project:** {run_data['project']}")
        with col2:
            created_at = pd.to_datetime(run_data['created_at']).strftime('%Y-%m-%d %H:%M:%S') if 'created_at' in run_data else "Unknown"
            st.markdown(f"**Created:** {created_at}")
            st.markdown(f"**Entity:** {run_data['entity']}")
            wandb_url = run_data.get('url', '#')
            st.markdown(f"**W&B URL:** [Open in W&B]({wandb_url})")
        
        # Summary metrics section
        st.subheader("Summary Metrics")
        if run_data['summary']:
            metrics = {}
            for key, value in run_data['summary'].items():
                if not key.startswith('_') and isinstance(value, (int, float)):
                    metrics[key] = value
            
            if metrics:
                metrics_df = pd.DataFrame([metrics])
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("No numerical metrics found in this run.")
        else:
            st.info("No summary metrics available for this run.")
    
    with config_tab:
        st.subheader("Run Configuration")
        
        if run_data['config']:
            # Organize config parameters by category if possible
            categorized_config = {}
            
            for key, value in run_data['config'].items():
                # Skip internal keys
                if key.startswith('_'):
                    continue
                
                # Try to categorize by prefix (e.g., "optimizer.learning_rate" goes into "optimizer" category)
                if '.' in key:
                    category, param = key.split('.', 1)
                    if category not in categorized_config:
                        categorized_config[category] = {}
                    categorized_config[category][param] = value
                else:
                    # No category, put in "general"
                    if "general" not in categorized_config:
                        categorized_config["general"] = {}
                    categorized_config["general"][key] = value
            
            # Display each category in an expander
            for category, params in categorized_config.items():
                with st.expander(f"{category.capitalize()} Parameters", expanded=True):
                    # Convert the parameters to a DataFrame for nice display
                    params_df = pd.DataFrame({"Parameter": list(params.keys()), "Value": list(params.values())})
                    st.dataframe(params_df, use_container_width=True, hide_index=True)
        else:
            st.info("No configuration parameters available for this run.")
    
    with metrics_tab:
        st.subheader("Metrics Over Time")
        
        # Check if we have history data
        if 'history' in run_data and not run_data['history'].empty:
            history_df = run_data['history']
            
            # Get all metric columns (numerical columns that don't start with _)
            metric_columns = [col for col in history_df.columns if not col.startswith('_') 
                             and col != '_step' and col != 'step'
                             and pd.api.types.is_numeric_dtype(history_df[col])]
            
            if metric_columns:
                # Let user select metrics to visualize
                selected_metrics = st.multiselect(
                    "Select metrics to visualize:",
                    options=metric_columns,
                    default=metric_columns[:min(3, len(metric_columns))]
                )
                
                if selected_metrics:
                    # Add a slider for smoothing
                    smoothing = st.slider("Smoothing", min_value=0.0, max_value=0.99, value=0.0, step=0.01)
                    
                    # Determine step column
                    step_col = '_step' if '_step' in history_df.columns else 'step' if 'step' in history_df.columns else None
                    
                    if step_col:
                        # Create a multi-line chart
                        import plotly.express as px
                        
                        # Apply smoothing if needed
                        plot_df = history_df.copy()
                        if smoothing > 0:
                            for metric in selected_metrics:
                                plot_df[metric] = plot_df[metric].ewm(alpha=(1 - smoothing)).mean()
                        
                        # Create a different y-axis for each metric
                        fig = px.line(
                            plot_df, 
                            x=step_col, 
                            y=selected_metrics,
                            labels={"value": "Metric Value", step_col: "Step"},
                            title="Metrics Over Time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Option to download the metrics data
                        if st.button("Download Metrics Data as CSV"):
                            csv_data, filename = export_to_csv(
                                history_df[selected_metrics + [step_col]], 
                                f"run_{run_id}_metrics"
                            )
                            st.download_button(
                                label="Download CSV",
                                data=csv_data,
                                file_name=filename,
                                mime="text/csv"
                            )
                    else:
                        st.warning("No step column found in history data. Cannot create time series visualization.")
                else:
                    st.info("Please select at least one metric to visualize.")
            else:
                st.info("No numerical metrics found in the history data.")
        else:
            st.info("No metrics history available for this run.")
    
    with files_tab:
        st.subheader("Files & Artifacts")
        
        if 'files' in run_data and run_data['files']:
            # Display file list in a table
            files_df = pd.DataFrame(run_data['files'])
            
            # Convert size to human readable format
            def format_size(size_bytes):
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if size_bytes < 1024 or unit == 'GB':
                        return f"{size_bytes:.2f} {unit}"
                    size_bytes /= 1024
            
            if 'size' in files_df.columns:
                files_df['size'] = files_df['size'].apply(format_size)
            
            # Convert timestamp if available
            if 'updated_at' in files_df.columns:
                files_df['updated_at'] = pd.to_datetime(files_df['updated_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                files_df = files_df.rename(columns={'updated_at': 'Last Updated'})
            
            # Rename columns for better display
            files_df = files_df.rename(columns={'name': 'Filename', 'size': 'Size'})
            
            st.dataframe(files_df, use_container_width=True, hide_index=True)
            
            # Allow user to select and download a file
            if len(files_df) > 0:
                st.subheader("Download Artifact")
                file_names = files_df['Filename'].tolist()
                selected_file = st.selectbox("Select a file to download:", file_names)
                
                if st.button("Download Selected File"):
                    with st.spinner(f"Downloading {selected_file}..."):
                        file_data = download_run_artifact(project_id, run_id, selected_file)
                        
                        if file_data:
                            # Determine mime type (simple version)
                            mime = "application/octet-stream"  # default
                            if selected_file.endswith('.csv'):
                                mime = "text/csv"
                            elif selected_file.endswith('.json'):
                                mime = "application/json"
                            elif selected_file.endswith('.txt'):
                                mime = "text/plain"
                            
                            st.download_button(
                                label=f"Download {selected_file}",
                                data=file_data,
                                file_name=selected_file,
                                mime=mime
                            )
                        else:
                            st.error("Failed to download the file. Please try again.")

# Sweeps page
def render_sweeps_page():
    """Render the sweeps page for a selected project."""
    if not st.session_state.selected_project:
        st.warning("No project selected. Please select a project first.")
        if st.button("Back to Projects"):
            st.session_state.current_page = "projects"
            st.rerun()
        return
    
    project_id = st.session_state.selected_project["id"]
    st.header(f"Sweeps for {project_id}")
    
    # Button to refresh sweeps
    if st.button("Refresh Sweeps"):
        st.session_state.sweeps = get_sweeps(project_id)
    
    # Get sweeps for the selected project
    with st.spinner("Loading sweeps..."):
        sweeps = get_sweeps(project_id)
    
    if not sweeps:
        st.warning("No sweeps found for this project.")
        return
    
    # Prepare data for table display
    sweeps_data = []
    for sweep in sweeps:
        sweep_data = {
            "ID": sweep["id"],
            "Name": sweep["name"],
            "State": sweep["state"],
            "Created": pd.to_datetime(sweep["created_at"]).strftime('%Y-%m-%d %H:%M:%S') if "created_at" in sweep and sweep["created_at"] else "",
        }
        
        # Add method and metric if available in config
        if "config" in sweep and sweep["config"]:
            if "method" in sweep["config"]:
                sweep_data["Method"] = sweep["config"]["method"]
            
            if "metric" in sweep["config"]:
                if isinstance(sweep["config"]["metric"], dict) and "name" in sweep["config"]["metric"]:
                    metric_name = sweep["config"]["metric"]["name"]
                    goal = sweep["config"]["metric"].get("goal", "")
                    sweep_data["Metric"] = f"{metric_name} ({goal})"
                elif isinstance(sweep["config"]["metric"], str):
                    sweep_data["Metric"] = sweep["config"]["metric"]
        
        sweeps_data.append(sweep_data)
    
    # Display table with sweeps
    sweeps_df = pd.DataFrame(sweeps_data)
    st.dataframe(sweeps_df, use_container_width=True, hide_index=True)
    
    # Select a sweep for detailed view
    st.subheader("Select a Sweep for Detailed Analysis")
    
    # Get sweep names/IDs for the selectbox
    sweep_options = [f"{sweep['name']} ({sweep['id']})" for sweep in sweeps]
    
    if sweep_options:
        selected_sweep_option = st.selectbox("Choose a sweep:", sweep_options)
        selected_sweep_id = selected_sweep_option.split("(")[-1].split(")")[0]
        
        if st.button("View Sweep Details"):
            # Find the selected sweep in the list
            for sweep in sweeps:
                if sweep["id"] == selected_sweep_id:
                    st.session_state.selected_sweep = sweep
                    st.session_state.current_page = "sweep_details"
                    
                    # Get detailed sweep data
                    with st.spinner("Loading sweep details..."):
                        st.session_state.sweep_data = get_sweep_details(project_id, sweep["id"])
                    
                    st.rerun()

# Sweep details page
def render_sweep_details_page():
    """Render detailed information for a selected sweep."""
    if not st.session_state.selected_project or not st.session_state.selected_sweep:
        st.warning("No sweep selected. Please select a project and sweep first.")
        if st.button("Back to Projects"):
            st.session_state.current_page = "projects"
            st.rerun()
        return
    
    project_id = st.session_state.selected_project["id"]
    sweep_id = st.session_state.selected_sweep["id"]
    sweep_name = st.session_state.selected_sweep["name"]
    
    st.header(f"Sweep Details: {sweep_name}")
    
    # Check if we have sweep data
    if not st.session_state.sweep_data:
        with st.spinner("Loading sweep details..."):
            st.session_state.sweep_data = get_sweep_details(project_id, sweep_id)
            
    sweep_data = st.session_state.sweep_data
    if not sweep_data:
        st.error("Failed to load sweep details. Please try again.")
        if st.button("Back to Sweeps"):
            st.session_state.current_page = "sweeps"
            st.rerun()
        return
    
    # Create tabs for different aspects of the sweep
    overview_tab, config_tab, runs_tab, visualization_tab = st.tabs([
        "Overview", "Configuration", "Runs", "Visualization"
    ])
    
    with overview_tab:
        st.subheader("Sweep Information")
        
        # Display basic sweep info in columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Sweep ID:** {sweep_data['id']}")
            st.markdown(f"**Name:** {sweep_data['name']}")
            st.markdown(f"**State:** {sweep_data['state']}")
            st.markdown(f"**Project:** {project_id}")
        with col2:
            created_at = pd.to_datetime(sweep_data['created_at']).strftime('%Y-%m-%d %H:%M:%S') if 'created_at' in sweep_data and sweep_data['created_at'] else "Unknown"
            st.markdown(f"**Created:** {created_at}")
            if 'config' in sweep_data and sweep_data['config']:
                if 'method' in sweep_data['config']:
                    st.markdown(f"**Method:** {sweep_data['config']['method']}")
                
                if 'metric' in sweep_data['config']:
                    if isinstance(sweep_data['config']['metric'], dict) and 'name' in sweep_data['config']['metric']:
                        metric_name = sweep_data['config']['metric']['name']
                        goal = sweep_data['config']['metric'].get('goal', '')
                        st.markdown(f"**Optimization Metric:** {metric_name} ({goal})")
                    elif isinstance(sweep_data['config']['metric'], str):
                        st.markdown(f"**Optimization Metric:** {sweep_data['config']['metric']}")
        
        # Best run section if available
        if 'best_run' in sweep_data and sweep_data['best_run']:
            st.subheader("Best Run")
            best_run = sweep_data['best_run']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Run ID:** {best_run['id']}")
                st.markdown(f"**Name:** {best_run['name']}")
                st.markdown(f"**State:** {best_run['state']}")
            with col2:
                created_at = pd.to_datetime(best_run['created_at']).strftime('%Y-%m-%d %H:%M:%S') if 'created_at' in best_run else "Unknown"
                st.markdown(f"**Created:** {created_at}")
            
            # Show the key metrics for the best run
            if 'summary' in best_run and best_run['summary']:
                st.subheader("Best Run Metrics")
                metrics = {}
                for key, value in best_run['summary'].items():
                    if not key.startswith('_') and isinstance(value, (int, float)):
                        metrics[key] = value
                
                if metrics:
                    metrics_df = pd.DataFrame([metrics])
                    st.dataframe(metrics_df, use_container_width=True)
                
                # Button to view the best run details
                if st.button("View Best Run Details"):
                    st.session_state.selected_run = best_run
                    st.session_state.current_page = "run_details"
                    with st.spinner("Loading run details..."):
                        st.session_state.run_data = get_run_details(project_id, best_run['id'])
                    st.rerun()
    
    with config_tab:
        st.subheader("Sweep Configuration")
        
        if 'config' in sweep_data and sweep_data['config']:
            # Create expandable sections for different parts of the config
            if 'parameters' in sweep_data['config']:
                with st.expander("Hyperparameter Search Space", expanded=True):
                    parameters = sweep_data['config']['parameters']
                    
                    for param_name, param_config in parameters.items():
                        st.markdown(f"**{param_name}**")
                        
                        # Display parameter configuration based on its type
                        if isinstance(param_config, dict):
                            param_type = next(iter(param_config)) if param_config else None
                            
                            if param_type == 'values':
                                st.markdown(f"Type: Discrete choice from values: {param_config['values']}")
                            elif param_type == 'min' and 'max' in param_config:
                                step = param_config.get('step', 'N/A')
                                st.markdown(f"Type: Range from {param_config['min']} to {param_config['max']} (step: {step})")
                            elif param_type == 'distribution':
                                st.markdown(f"Type: Distribution - {param_config['distribution']}")
                                if 'min' in param_config and 'max' in param_config:
                                    st.markdown(f"Range: {param_config['min']} to {param_config['max']}")
                            else:
                                st.json(param_config)
                        else:
                            st.markdown(f"Value: {param_config}")
                        
                        st.markdown("---")
            
            # Display method, metric, early termination, etc.
            general_config = {k: v for k, v in sweep_data['config'].items() if k != 'parameters'}
            if general_config:
                with st.expander("General Configuration", expanded=True):
                    st.json(general_config)
        else:
            st.info("No configuration available for this sweep.")
    
    with runs_tab:
        st.subheader("Sweep Runs")
        
        if 'runs' in sweep_data and sweep_data['runs']:
            # Prepare data for table display
            runs_data = []
            for run in sweep_data['runs']:
                run_data = {
                    "ID": run["id"],
                    "Name": run["name"],
                    "State": run["state"],
                    "Created": pd.to_datetime(run["created_at"]).strftime('%Y-%m-%d %H:%M:%S') if "created_at" in run else "",
                }
                
                # Add hyperparameters from config
                if "config" in run:
                    for key, value in run["config"].items():
                        if not key.startswith('_') and key not in run_data:
                            if isinstance(value, (int, float, str, bool)):
                                run_data[key] = value
                
                # Add metrics from summary
                if "summary" in run:
                    for key, value in run["summary"].items():
                        if not key.startswith('_') and isinstance(value, (int, float)) and key not in run_data:
                            run_data[key] = value
                
                runs_data.append(run_data)
            
            # Display table with runs
            runs_df = pd.DataFrame(runs_data)
            
            # Allow filtering on numerical columns
            if len(runs_df) > 0:
                st.markdown("### Filter Runs")
                
                # Get numerical columns for filtering
                numerical_cols = [col for col in runs_df.columns if runs_df[col].dtype in ['int64', 'float64']]
                
                if numerical_cols:
                    # Let user select a column to filter on
                    filter_col = st.selectbox("Select column to filter:", numerical_cols)
                    
                    # Get min and max values for the selected column
                    min_val = float(runs_df[filter_col].min())
                    max_val = float(runs_df[filter_col].max())
                    
                    # Create a slider for filtering
                    filter_range = st.slider(
                        f"Filter by {filter_col}",
                        min_val,
                        max_val,
                        (min_val, max_val)
                    )
                    
                    # Filter the dataframe
                    filtered_df = runs_df[(runs_df[filter_col] >= filter_range[0]) & (runs_df[filter_col] <= filter_range[1])]
                    
                    # Display the filtered dataframe
                    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(runs_df, use_container_width=True, hide_index=True)
            else:
                st.dataframe(runs_df, use_container_width=True, hide_index=True)
            
            # Allow selecting a run to view details
            st.markdown("### Select Run to View Details")
            run_options = [f"{run['name']} ({run['id']})" for run in sweep_data['runs']]
            
            if run_options:
                selected_run_option = st.selectbox("Choose a run:", run_options)
                selected_run_id = selected_run_option.split("(")[-1].split(")")[0]
                
                if st.button("View Run Details"):
                    # Find the selected run in the list
                    for run in sweep_data['runs']:
                        if run["id"] == selected_run_id:
                            st.session_state.selected_run = run
                            st.session_state.current_page = "run_details"
                            
                            # Get detailed run data
                            with st.spinner("Loading run details..."):
                                st.session_state.run_data = get_run_details(project_id, run["id"])
                            
                            st.rerun()
        else:
            st.info("No runs available for this sweep.")
    
    with visualization_tab:
        st.subheader("Parameter Importance & Performance Visualization")
        
        if 'runs' in sweep_data and sweep_data['runs'] and len(sweep_data['runs']) > 1:
            # Get optimization metric name if available
            optimization_metric = None
            if 'config' in sweep_data and sweep_data['config'] and 'metric' in sweep_data['config']:
                if isinstance(sweep_data['config']['metric'], dict) and 'name' in sweep_data['config']['metric']:
                    optimization_metric = sweep_data['config']['metric']['name']
                elif isinstance(sweep_data['config']['metric'], str):
                    optimization_metric = sweep_data['config']['metric']
            
            # Try to find a common metric across runs if none is specified
            if not optimization_metric:
                # Look for common metrics in the first run's summary
                if sweep_data['runs'] and 'summary' in sweep_data['runs'][0]:
                    possible_metrics = [key for key, value in sweep_data['runs'][0]['summary'].items() 
                                      if not key.startswith('_') and isinstance(value, (int, float))]
                    
                    # Check if any of these metrics exist in all runs
                    for metric in possible_metrics:
                        if all(metric in run.get('summary', {}) for run in sweep_data['runs']):
                            optimization_metric = metric
                            break
            
            if optimization_metric:
                # Collect hyperparameters and metrics from all runs
                run_results = []
                for run in sweep_data['runs']:
                    if 'summary' in run and optimization_metric in run['summary']:
                        result = {'run_id': run['id'], 'run_name': run['name']}
                        
                        # Add hyperparameters from config
                        if 'config' in run:
                            for key, value in run['config'].items():
                                if not key.startswith('_') and isinstance(value, (int, float, str, bool)):
                                    result[key] = value
                        
                        # Add the optimization metric
                        result[optimization_metric] = run['summary'][optimization_metric]
                        
                        run_results.append(result)
                
                if run_results:
                    # Convert to DataFrame for visualization
                    results_df = pd.DataFrame(run_results)
                    
                    # Get hyperparameter columns (exclude run_id, run_name, and metrics)
                    hyperparam_cols = [col for col in results_df.columns 
                                     if col not in ['run_id', 'run_name', optimization_metric]]
                    
                    if hyperparam_cols:
                        # Let user select hyperparameters to visualize
                        selected_hyperparams = st.multiselect(
                            "Select hyperparameters to visualize:",
                            options=hyperparam_cols,
                            default=hyperparam_cols[:min(2, len(hyperparam_cols))]
                        )
                        
                        if selected_hyperparams:
                            import plotly.express as px
                            
                            # Scatter plot for selected hyperparameters vs. optimization metric
                            if len(selected_hyperparams) == 1:
                                # 2D scatter plot
                                x_param = selected_hyperparams[0]
                                
                                fig = px.scatter(
                                    results_df,
                                    x=x_param,
                                    y=optimization_metric,
                                    hover_data=['run_name'],
                                    title=f"Impact of {x_param} on {optimization_metric}",
                                    labels={x_param: x_param, optimization_metric: optimization_metric}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            elif len(selected_hyperparams) == 2:
                                # 3D scatter plot with two hyperparameters
                                x_param = selected_hyperparams[0]
                                y_param = selected_hyperparams[1]
                                
                                fig = px.scatter(
                                    results_df,
                                    x=x_param,
                                    y=y_param,
                                    color=optimization_metric,
                                    hover_data=['run_name'],
                                    title=f"Impact of {x_param} and {y_param} on {optimization_metric}",
                                    labels={
                                        x_param: x_param, 
                                        y_param: y_param, 
                                        optimization_metric: optimization_metric
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            elif len(selected_hyperparams) > 2:
                                # Parallel coordinates plot for multiple hyperparameters
                                dimensions = selected_hyperparams + [optimization_metric]
                                
                                fig = px.parallel_coordinates(
                                    results_df,
                                    dimensions=dimensions,
                                    color=optimization_metric,
                                    title=f"Parallel Coordinates Plot of Hyperparameters and {optimization_metric}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Please select at least one hyperparameter to visualize.")
                    else:
                        st.info("No hyperparameters found in the run configurations.")
                else:
                    st.info(f"No runs with the metric '{optimization_metric}' found.")
            else:
                st.info("No common optimization metric found across runs.")
        else:
            st.info("Not enough runs to visualize. A sweep needs at least 2 runs for meaningful visualization.")

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
