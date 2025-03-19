import streamlit as st
import pandas as pd
import wandb
from datetime import datetime

def format_timestamp(timestamp):
    """Convert Unix timestamp to readable date format"""
    if timestamp:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return "N/A"

def project_explorer():
    """
    Browse and explore W&B projects, runs, and sweeps
    """
    if not st.session_state.authenticated or not st.session_state.api:
        st.warning("Please authenticate with W&B first")
        return
    
    st.header("Project Explorer")
    
    try:
        # Get all projects for the current user
        user = st.session_state.api.viewer()['entity']
        projects = st.session_state.api.projects(user)
        
        project_names = [p.name for p in projects]
        
        if not project_names:
            st.info("No projects found in your W&B account")
            return
        
        # Project selection
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_project_name = st.selectbox(
                "Select Project",
                options=project_names,
                index=0 if st.session_state.selected_project is None else project_names.index(st.session_state.selected_project)
            )
        
        with col2:
            if st.button("Load Project"):
                st.session_state.selected_project = selected_project_name
                st.session_state.selected_run = None
                st.session_state.selected_sweep = None
                st.rerun()
        
        # If a project is selected
        if st.session_state.selected_project:
            project = st.session_state.api.project(user, st.session_state.selected_project)
            
            # Project info
            st.subheader(f"Project: {project.name}")
            st.markdown(f"**Description**: {project.description if project.description else 'No description'}")
            
            # Create tabs for Runs and Sweeps
            run_tab, sweep_tab = st.tabs(["Runs", "Sweeps"])
            
            # Runs tab
            with run_tab:
                st.subheader("Runs")
                
                # Get all runs for the project
                runs = st.session_state.api.runs(f"{user}/{st.session_state.selected_project}")
                
                if not runs:
                    st.info("No runs found in this project")
                else:
                    # Create a dataframe of runs
                    runs_data = []
                    for run in runs:
                        run_data = {
                            "Run ID": run.id,
                            "Name": run.name,
                            "Status": run.state,
                            "Created": format_timestamp(run.created_at),
                            "Runtime (min)": round((run.runtime or 0) / 60, 2),
                            "Tags": ", ".join(run.tags) if run.tags else ""
                        }
                        
                        # Add summary metrics if available
                        if hasattr(run, 'summary') and run.summary:
                            for key in run.summary:
                                if key not in ['_timestamp', '_step'] and not key.startswith('_'):
                                    value = run.summary[key]
                                    # Only include primitive types or ones that can be easily converted to string
                                    if isinstance(value, (int, float, str, bool)) or value is None:
                                        run_data[key] = value
                        
                        runs_data.append(run_data)
                    
                    # Convert to dataframe
                    runs_df = pd.DataFrame(runs_data)
                    
                    # Filter options
                    st.markdown("### Filter Runs")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        status_filter = st.multiselect(
                            "Status",
                            options=sorted(runs_df["Status"].unique()),
                            default=sorted(runs_df["Status"].unique())
                        )
                    
                    with col2:
                        search_term = st.text_input("Search by Name or Tags")
                    
                    # Apply filters
                    filtered_df = runs_df[runs_df["Status"].isin(status_filter)]
                    
                    if search_term:
                        filtered_df = filtered_df[
                            filtered_df["Name"].str.contains(search_term, case=False, na=False) |
                            filtered_df["Tags"].str.contains(search_term, case=False, na=False)
                        ]
                    
                    # Show filtered runs table
                    if not filtered_df.empty:
                        st.dataframe(filtered_df, use_container_width=True)
                        
                        # Run selection
                        selected_run_id = st.selectbox(
                            "Select a run to view details",
                            options=filtered_df["Run ID"].tolist(),
                            format_func=lambda x: f"{x} - {filtered_df[filtered_df['Run ID']==x]['Name'].iloc[0]}"
                        )
                        
                        if st.button("View Run Details"):
                            st.session_state.selected_run = selected_run_id
                            st.session_state.active_tab = "Run Details"
                            st.rerun()
                    else:
                        st.info("No runs match the current filters")
            
            # Sweeps tab
            with sweep_tab:
                st.subheader("Sweeps")
                
                try:
                    # Get all sweeps for the project
                    sweeps = st.session_state.api.sweeps(f"{user}/{st.session_state.selected_project}")
                    
                    if not sweeps:
                        st.info("No sweeps found in this project")
                    else:
                        # Create a dataframe of sweeps
                        sweeps_data = []
                        for sweep in sweeps:
                            sweeps_data.append({
                                "Sweep ID": sweep.id,
                                "Name": sweep.name or sweep.id,
                                "Status": sweep.state,
                                "Created": format_timestamp(sweep.created),
                                "Method": sweep.config.get("method", "N/A"),
                                "Runs": sweep.runs_count,
                                "Best Run": sweep.best_run.name if hasattr(sweep, 'best_run') and sweep.best_run else "N/A"
                            })
                        
                        # Convert to dataframe
                        sweeps_df = pd.DataFrame(sweeps_data)
                        
                        # Show sweeps table
                        st.dataframe(sweeps_df, use_container_width=True)
                        
                        # Sweep selection
                        if not sweeps_df.empty:
                            selected_sweep_id = st.selectbox(
                                "Select a sweep to analyze",
                                options=sweeps_df["Sweep ID"].tolist(),
                                format_func=lambda x: f"{x} - {sweeps_df[sweeps_df['Sweep ID']==x]['Name'].iloc[0]}"
                            )
                            
                            if st.button("Analyze Sweep"):
                                st.session_state.selected_sweep = selected_sweep_id
                                st.session_state.active_tab = "Sweep Analysis"
                                st.rerun()
                except Exception as e:
                    st.error(f"Error loading sweeps: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading projects: {str(e)}")
