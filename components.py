import streamlit as st
import pandas as pd
import base64
import io
import sys

# Ensure utils.py is in the import path
sys.path.append(".")
from utils import (
    authenticate_wandb,
    logout_wandb,
    get_projects,
    get_runs,
    get_run_details,
    download_run_artifact,
    get_sweeps,
    get_sweep_details,
    export_to_csv
)
from visualizations import (
    plot_metric_over_time,
    plot_multiple_metrics,
    plot_parameter_importance,
    plot_run_comparisons,
    plot_sweep_results
)

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

def render_projects_page():
    """Render the projects page."""
    st.header("Your W&B Projects")
    
    # Refresh projects button
    if st.button("🔄 Refresh Projects"):
        with st.spinner("Fetching projects..."):
            get_projects()
    
    # If projects list is empty, try to fetch them
    if not st.session_state.projects:
        with st.spinner("Fetching projects..."):
            projects = get_projects()
    else:
        projects = st.session_state.projects
    
    if not projects:
        st.info("No projects found. Create a project in W&B and try again.")
        return
    
    # Create a DataFrame for better display
    df = pd.DataFrame(projects)
    
    # Add filtering options
    st.subheader("Filter Projects")
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("Search by name", "")
    with col2:
        selected_entity = st.selectbox(
            "Filter by entity",
            options=["All"] + sorted(list(set(df["entity"]))),
            index=0
        )
    
    # Apply filters
    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df["name"].str.contains(search_term, case=False)]
    if selected_entity != "All":
        filtered_df = filtered_df[filtered_df["entity"] == selected_entity]
    
    # Format the dataframe for display
    display_df = filtered_df[["name", "entity", "description"]].copy()
    display_df.columns = ["Project", "Entity", "Description"]
    
    # Display projects
    st.subheader("Available Projects")
    if display_df.empty:
        st.info("No projects match your filters.")
    else:
        # Display as a table with clickable rows
        for _, row in display_df.iterrows():
            project_id = f"{row['Entity']}/{row['Project']}"
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{row['Project']}**")
                st.write(row['Description'] if pd.notna(row['Description']) else "")
            with col2:
                st.write(f"Entity: {row['Entity']}")
            with col3:
                if st.button("Select", key=f"project_{project_id}"):
                    # Find the full project info
                    project_info = next((p for p in projects if p["id"] == project_id), None)
                    st.session_state.selected_project = project_info
                    st.session_state.current_page = "runs"
                    st.rerun()
            st.divider()

def render_runs_page():
    """Render the runs page for a selected project."""
    if not st.session_state.selected_project:
        st.error("No project selected. Please select a project first.")
        st.session_state.current_page = "projects"
        st.rerun()
        return
    
    project_id = st.session_state.selected_project["id"]
    st.header(f"Runs in {project_id}")
    
    # Refresh runs button
    if st.button("🔄 Refresh Runs"):
        with st.spinner("Fetching runs..."):
            runs = get_runs(project_id)
    else:
        # Fetch runs if not already fetched
        with st.spinner("Fetching runs..."):
            runs = get_runs(project_id)
    
    if not runs:
        st.info(f"No runs found in project {project_id}.")
        return
    
    # Create a DataFrame for better display
    runs_data = []
    for run in runs:
        run_data = {
            "id": run["id"],
            "name": run["name"],
            "state": run["state"],
            "created_at": run["created_at"]
        }
        
        # Add summary metrics
        for key, value in run["summary"].items():
            if isinstance(value, (int, float)):
                run_data[f"metric_{key}"] = value
        
        runs_data.append(run_data)
    
    df = pd.DataFrame(runs_data)
    
    # Add filtering options
    st.subheader("Filter Runs")
    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input("Search by name or ID", "")
    with col2:
        selected_state = st.selectbox(
            "Filter by state",
            options=["All"] + sorted(list(set(df["state"]))),
            index=0
        )
    with col3:
        # Get all metric columns
        metric_columns = [col for col in df.columns if col.startswith("metric_")]
        metric_options = [col.replace("metric_", "") for col in metric_columns]
        
        if metric_options:
            sort_by = st.selectbox(
                "Sort by metric",
                options=["Created (newest first)"] + metric_options,
                index=0
            )
        else:
            sort_by = "Created (newest first)"
    
    # Apply filters
    filtered_df = df.copy()
    if search_term:
        filter_condition = (
            filtered_df["name"].str.contains(search_term, case=False) | 
            filtered_df["id"].str.contains(search_term, case=False)
        )
        filtered_df = filtered_df[filter_condition]
    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["state"] == selected_state]
    
    # Apply sorting
    if sort_by == "Created (newest first)":
        filtered_df = filtered_df.sort_values("created_at", ascending=False)
    else:
        metric_column = f"metric_{sort_by}"
        if metric_column in filtered_df.columns:
            filtered_df = filtered_df.sort_values(metric_column, ascending=False)
    
    # Format the dataframe for display
    display_df = filtered_df[["name", "id", "state", "created_at"]].copy()
    display_df.columns = ["Name", "ID", "State", "Created At"]
    
    # Display runs
    st.subheader("Available Runs")
    
    if display_df.empty:
        st.info("No runs match your filters.")
    else:
        # Add export to CSV button
        if st.button("📥 Export filtered runs to CSV"):
            csv_data, filename = export_to_csv(filtered_df, f"{project_id.replace('/', '_')}_runs")
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )
        
        # Display as a table with clickable rows
        for _, row in display_df.iterrows():
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{row['Name']}**")
                st.write(f"ID: {row['ID']}")
            with col2:
                st.write(f"State: {row['State']}")
                st.write(f"Created: {row['Created At']}")
            with col3:
                if st.button("View Details", key=f"run_{row['ID']}"):
                    run_info = next((r for r in runs if r["id"] == row['ID']), None)
                    st.session_state.selected_run = run_info
                    st.session_state.current_page = "run_details"
                    st.rerun()
            
            # Add metrics if available
            metrics_row = filtered_df[filtered_df["id"] == row["ID"]].iloc[0]
            metric_columns = [col for col in metrics_row.index if col.startswith("metric_")]
            
            if metric_columns:
                metrics_dict = {col.replace("metric_", ""): metrics_row[col] for col in metric_columns}
                st.write("Metrics:", ", ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()]))
            
            st.divider()
        
        # Add comparison feature
        st.subheader("Compare Runs")
        selected_runs_for_comparison = st.multiselect(
            "Select runs to compare",
            options=[(row['ID'], row['Name']) for _, row in display_df.iterrows()],
            format_func=lambda x: x[1]
        )
        
        if selected_runs_for_comparison:
            run_ids = [run_id for run_id, _ in selected_runs_for_comparison]
            runs_to_compare = [run for run in runs if run["id"] in run_ids]
            
            if runs_to_compare:
                # Get all available metrics across the selected runs
                all_metrics = set()
                for run in runs_to_compare:
                    all_metrics.update(run["summary"].keys())
                
                # Remove non-numeric metrics
                numeric_metrics = []
                for metric in all_metrics:
                    is_numeric = all(
                        isinstance(run["summary"].get(metric), (int, float))
                        for run in runs_to_compare
                        if metric in run["summary"]
                    )
                    if is_numeric:
                        numeric_metrics.append(metric)
                
                selected_metrics = st.multiselect(
                    "Select metrics to compare",
                    options=sorted(numeric_metrics)
                )
                
                if selected_metrics:
                    comparison_fig = plot_run_comparisons(runs_to_compare, selected_metrics)
                    if comparison_fig:
                        st.plotly_chart(comparison_fig, use_container_width=True)
                    
                    # Export comparison data
                    if st.button("📥 Export comparison data to CSV"):
                        comparison_data = []
                        for run in runs_to_compare:
                            run_data = {"run_id": run["id"], "run_name": run["name"]}
                            for metric in selected_metrics:
                                if metric in run["summary"]:
                                    run_data[metric] = run["summary"][metric]
                            comparison_data.append(run_data)
                        
                        csv_data, filename = export_to_csv(comparison_data, "runs_comparison")
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv"
                        )

def render_run_details_page():
    """Render detailed information for a selected run."""
    if not st.session_state.selected_project or not st.session_state.selected_run:
        st.error("No run selected. Please select a run first.")
        st.session_state.current_page = "runs"
        st.rerun()
        return
    
    project_id = st.session_state.selected_project["id"]
    run_id = st.session_state.selected_run["id"]
    run_name = st.session_state.selected_run["name"]
    
    st.header(f"Run Details: {run_name}")
    
    # Fetch detailed run information if not already fetched
    if not st.session_state.run_data or st.session_state.run_data["id"] != run_id:
        with st.spinner("Fetching run details..."):
            run_data = get_run_details(project_id, run_id)
            if run_data:
                st.session_state.run_data = run_data
    
    run_data = st.session_state.run_data
    
    if not run_data:
        st.error(f"Failed to fetch details for run {run_id}.")
        return
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Runs"):
            st.session_state.current_page = "runs"
            st.session_state.selected_run = None
            st.rerun()
    with col2:
        st.markdown(f"<a href='{run_data['url']}' target='_blank'>Open in W&B ↗</a>", unsafe_allow_html=True)
    
    # Display run information in tabs
    tab_overview, tab_config, tab_metrics, tab_artifacts = st.tabs(["Overview", "Config", "Metrics", "Artifacts"])
    
    with tab_overview:
        st.subheader("Run Information")
        
        # Basic information
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**ID:** {run_data['id']}")
            st.write(f"**Name:** {run_data['name']}")
            st.write(f"**Project:** {run_data['project']}")
        with col2:
            st.write(f"**Entity:** {run_data['entity']}")
            st.write(f"**Created:** {run_data['created_at']}")
            st.write(f"**State:** {run_data['state']}")
        
        # Summary metrics
        st.subheader("Summary Metrics")
        if run_data["summary"]:
            metrics_df = pd.DataFrame(
                [(k, v) for k, v in run_data["summary"].items()],
                columns=["Metric", "Value"]
            )
            st.dataframe(metrics_df, use_container_width=True)
            
            # Export summary metrics to CSV
            if st.button("📥 Export summary metrics to CSV"):
                csv_data, filename = export_to_csv(metrics_df, f"{run_id}_summary_metrics")
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv"
                )
        else:
            st.info("No summary metrics available for this run.")
    
    with tab_config:
        st.subheader("Run Configuration")
        
        if run_data["config"]:
            # Convert config to DataFrame for better display
            config_df = pd.DataFrame(
                [(k, str(v)) for k, v in run_data["config"].items()],
                columns=["Parameter", "Value"]
            )
            st.dataframe(config_df, use_container_width=True)
            
            # Export config to CSV
            if st.button("📥 Export configuration to CSV"):
                csv_data, filename = export_to_csv(config_df, f"{run_id}_config")
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv"
                )
        else:
            st.info("No configuration parameters available for this run.")
    
    with tab_metrics:
        st.subheader("Metrics Over Time")
        
        if isinstance(run_data["history"], pd.DataFrame) and not run_data["history"].empty:
            # Get numeric columns from history
            numeric_columns = run_data["history"].select_dtypes(include=["number"]).columns.tolist()
            
            # Filter out internal columns
            metric_columns = [col for col in numeric_columns if not col.startswith("_")]
            
            if metric_columns:
                # Let the user select metrics to visualize
                selected_metrics = st.multiselect(
                    "Select metrics to visualize",
                    options=metric_columns,
                    default=metric_columns[:min(3, len(metric_columns))]
                )
                
                if selected_metrics:
                    st.plotly_chart(
                        plot_multiple_metrics(run_data["history"], selected_metrics),
                        use_container_width=True
                    )
                    
                    # Export history data to CSV
                    if st.button("📥 Export metrics history to CSV"):
                        export_columns = ["_step"] + selected_metrics if "_step" in run_data["history"].columns else selected_metrics
                        export_df = run_data["history"][export_columns].dropna(how="all", subset=selected_metrics)
                        
                        csv_data, filename = export_to_csv(export_df, f"{run_id}_metrics_history")
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv"
                        )
            else:
                st.info("No metric data available in this run's history.")
        else:
            st.info("No history data available for this run.")
    
    with tab_artifacts:
        st.subheader("Run Artifacts and Files")
        
        if run_data["files"]:
            # Convert files to DataFrame for better display
            files_df = pd.DataFrame(run_data["files"])
            
            # Sort by update time
            if "updated_at" in files_df.columns:
                files_df = files_df.sort_values("updated_at", ascending=False)
            
            # Format size column
            if "size" in files_df.columns:
                files_df["size_formatted"] = files_df["size"].apply(
                    lambda x: f"{x/1024/1024:.2f} MB" if x > 1024*1024 else f"{x/1024:.2f} KB"
                )
            
            # Display files table
            st.dataframe(
                files_df[["name", "size_formatted" if "size_formatted" in files_df.columns else "size", 
                          "updated_at" if "updated_at" in files_df.columns else ""]],
                use_container_width=True
            )
            
            # Create a file selector
            selected_file = st.selectbox(
                "Select a file to download",
                options=files_df["name"].tolist()
            )
            
            if st.button(f"Download {selected_file}"):
                with st.spinner(f"Downloading {selected_file}..."):
                    file_data = download_run_artifact(project_id, run_id, selected_file)
                    if file_data:
                        st.download_button(
                            label=f"Save {selected_file}",
                            data=file_data,
                            file_name=selected_file,
                            mime="application/octet-stream"
                        )
        else:
            st.info("No files or artifacts available for this run.")

def render_sweeps_page():
    """Render the sweeps page for a selected project."""
    if not st.session_state.selected_project:
        st.error("No project selected. Please select a project first.")
        st.session_state.current_page = "projects"
        st.rerun()
        return
    
    project_id = st.session_state.selected_project["id"]
    st.header(f"Sweeps in {project_id}")
    
    # Refresh sweeps button
    if st.button("🔄 Refresh Sweeps"):
        with st.spinner("Fetching sweeps..."):
            sweeps = get_sweeps(project_id)
    else:
        # Fetch sweeps if not already fetched
        with st.spinner("Fetching sweeps..."):
            sweeps = get_sweeps(project_id)
    
    if not sweeps:
        st.info(f"No sweeps found in project {project_id}.")
        return
    
    # Create a DataFrame for better display
    df = pd.DataFrame(sweeps)
    
    # Add filtering options
    st.subheader("Filter Sweeps")
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("Search by name or ID", "", key="sweep_search")
    with col2:
        if "state" in df.columns:
            selected_state = st.selectbox(
                "Filter by state",
                options=["All"] + sorted(list(set(df["state"]))),
                index=0,
                key="sweep_state_filter"
            )
        else:
            selected_state = "All"
    
    # Apply filters
    filtered_df = df.copy()
    if search_term:
        filter_condition = (
            filtered_df["name"].str.contains(search_term, case=False) | 
            filtered_df["id"].str.contains(search_term, case=False)
        )
        filtered_df = filtered_df[filter_condition]
    if selected_state != "All" and "state" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["state"] == selected_state]
    
    # Format the dataframe for display
    display_columns = ["name", "id"]
    if "state" in filtered_df.columns:
        display_columns.append("state")
    if "created_at" in filtered_df.columns:
        display_columns.append("created_at")
    
    display_df = filtered_df[display_columns].copy()
    display_df.columns = ["Name", "ID"] + (["State"] if "state" in filtered_df.columns else []) + (["Created At"] if "created_at" in filtered_df.columns else [])
    
    # Display sweeps
    st.subheader("Available Sweeps")
    
    if display_df.empty:
        st.info("No sweeps match your filters.")
    else:
        # Display as a table with clickable rows
        for _, row in display_df.iterrows():
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{row['Name']}**")
                st.write(f"ID: {row['ID']}")
            with col2:
                if "State" in row:
                    st.write(f"State: {row['State']}")
                if "Created At" in row:
                    st.write(f"Created: {row['Created At']}")
            with col3:
                if st.button("View Details", key=f"sweep_{row['ID']}"):
                    sweep_info = next((s for s in sweeps if s["id"] == row['ID']), None)
                    st.session_state.selected_sweep = sweep_info
                    st.session_state.current_page = "sweep_details"
                    st.rerun()
            st.divider()

def render_sweep_details_page():
    """Render detailed information for a selected sweep."""
    if not st.session_state.selected_project or not st.session_state.selected_sweep:
        st.error("No sweep selected. Please select a sweep first.")
        st.session_state.current_page = "sweeps"
        st.rerun()
        return
    
    project_id = st.session_state.selected_project["id"]
    sweep_id = st.session_state.selected_sweep["id"]
    sweep_name = st.session_state.selected_sweep.get("name", sweep_id)
    
    st.header(f"Sweep Details: {sweep_name}")
    
    # Fetch detailed sweep information if not already fetched
    if not st.session_state.sweep_data or st.session_state.sweep_data["id"] != sweep_id:
        with st.spinner("Fetching sweep details..."):
            sweep_data = get_sweep_details(project_id, sweep_id)
            if sweep_data:
                st.session_state.sweep_data = sweep_data
    
    sweep_data = st.session_state.sweep_data
    
    if not sweep_data:
        st.error(f"Failed to fetch details for sweep {sweep_id}.")
        return
    
    # Navigation button
    if st.button("← Back to Sweeps"):
        st.session_state.current_page = "sweeps"
        st.session_state.selected_sweep = None
        st.rerun()
    
    # Display sweep information in tabs
    tab_overview, tab_config, tab_runs, tab_visualization = st.tabs(
        ["Overview", "Config", "Runs", "Visualization"]
    )
    
    with tab_overview:
        st.subheader("Sweep Information")
        
        # Basic information
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**ID:** {sweep_data['id']}")
            st.write(f"**Name:** {sweep_data['name']}")
        with col2:
            st.write(f"**State:** {sweep_data['state']}")
            if "created_at" in sweep_data and sweep_data["created_at"]:
                st.write(f"**Created:** {sweep_data['created_at']}")
        
        # Best run information
        st.subheader("Best Run")
        
        if sweep_data["best_run"]:
            best_run = sweep_data["best_run"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**ID:** {best_run['id']}")
                st.write(f"**Name:** {best_run['name']}")
            with col2:
                # Determine optimization metric if possible
                optimization_metric = None
                optimization_goal = None
                
                if "config" in sweep_data and "metric" in sweep_data["config"]:
                    metric_config = sweep_data["config"]["metric"]
                    if isinstance(metric_config, dict):
                        if "name" in metric_config:
                            optimization_metric = metric_config["name"]
                        if "goal" in metric_config:
                            optimization_goal = metric_config["goal"]
                    elif isinstance(metric_config, str):
                        optimization_metric = metric_config
                
                if optimization_metric and optimization_metric in best_run["summary"]:
                    metric_value = best_run["summary"][optimization_metric]
                    st.write(f"**{optimization_metric}:** {metric_value}")
                    if optimization_goal:
                        st.write(f"**Optimization goal:** {optimization_goal}")
            
            # Button to view the best run details
            if st.button("View Best Run Details"):
                # Find the run in the original runs list
                run_info = next((r for r in sweep_data["runs"] if r["id"] == best_run["id"]), None)
                if run_info:
                    st.session_state.selected_run = run_info
                    st.session_state.current_page = "run_details"
                    st.rerun()
        else:
            st.info("Could not determine the best run for this sweep.")
    
    with tab_config:
        st.subheader("Sweep Configuration")
        
        if "config" in sweep_data and sweep_data["config"]:
            # Display parameters and their ranges
            if "parameters" in sweep_data["config"]:
                st.write("**Parameters:**")
                parameters = sweep_data["config"]["parameters"]
                
                for param_name, param_config in parameters.items():
                    st.write(f"**{param_name}:**")
                    param_df = pd.DataFrame(
                        [(k, str(v)) for k, v in param_config.items()],
                        columns=["Property", "Value"]
                    )
                    st.dataframe(param_df)
            
            # Display method
            if "method" in sweep_data["config"]:
                st.write(f"**Method:** {sweep_data['config']['method']}")
            
            # Display metric
            if "metric" in sweep_data["config"]:
                metric_config = sweep_data["config"]["metric"]
                if isinstance(metric_config, dict):
                    st.write(f"**Metric name:** {metric_config.get('name', 'N/A')}")
                    st.write(f"**Optimization goal:** {metric_config.get('goal', 'N/A')}")
                else:
                    st.write(f"**Metric:** {metric_config}")
            
            # Export config to CSV
            if st.button("📥 Export sweep configuration to CSV"):
                # Flatten the config dictionary
                flat_config = []
                for key, value in sweep_data["config"].items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, dict):
                                for param_key, param_value in sub_value.items():
                                    flat_config.append((f"{key}.{sub_key}.{param_key}", str(param_value)))
                            else:
                                flat_config.append((f"{key}.{sub_key}", str(sub_value)))
                    else:
                        flat_config.append((key, str(value)))
                
                config_df = pd.DataFrame(flat_config, columns=["Parameter", "Value"])
                csv_data, filename = export_to_csv(config_df, f"{sweep_id}_config")
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv"
                )
        else:
            st.info("No configuration available for this sweep.")
    
    with tab_runs:
        st.subheader("Sweep Runs")
        
        if "runs" in sweep_data and sweep_data["runs"]:
            runs = sweep_data["runs"]
            
            # Create DataFrame for runs
            runs_data = []
            metrics = set()
            
            for run in runs:
                run_data = {
                    "id": run["id"],
                    "name": run["name"],
                    "state": run.get("state", "unknown")
                }
                
                # Add parameters from config
                for key, value in run["config"].items():
                    run_data[f"param_{key}"] = value
                
                # Add metrics from summary
                for key, value in run["summary"].items():
                    if isinstance(value, (int, float)):
                        run_data[f"metric_{key}"] = value
                        metrics.add(key)
                
                runs_data.append(run_data)
            
            runs_df = pd.DataFrame(runs_data)
            
            # Add filtering options
            st.subheader("Filter Runs")
            col1, col2 = st.columns(2)
            with col1:
                search_term = st.text_input("Search by name or ID", "", key="sweep_run_search")
            with col2:
                if "state" in runs_df.columns:
                    selected_state = st.selectbox(
                        "Filter by state",
                        options=["All"] + sorted(list(set(runs_df["state"]))),
                        index=0,
                        key="sweep_run_state_filter"
                    )
                else:
                    selected_state = "All"
            
            # Apply filters
            filtered_df = runs_df.copy()
            if search_term:
                filter_condition = (
                    filtered_df["name"].str.contains(search_term, case=False) | 
                    filtered_df["id"].str.contains(search_term, case=False)
                )
                filtered_df = filtered_df[filter_condition]
            if selected_state != "All" and "state" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["state"] == selected_state]
            
            # Sort options
            metrics_list = list(metrics)
            sort_options = ["None"] + [f"metric_{m}" for m in metrics_list]
            sort_by = st.selectbox(
                "Sort by metric",
                options=sort_options,
                format_func=lambda x: x.replace("metric_", "") if x != "None" else x
            )
            
            if sort_by != "None":
                ascending = st.checkbox("Ascending order", value=False)
                filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
            
            # Display runs table
            if filtered_df.empty:
                st.info("No runs match your filters.")
            else:
                # Choose columns to display
                display_columns = ["name", "id", "state"]
                
                # Choose which parameter and metric columns to display
                param_columns = [c for c in filtered_df.columns if c.startswith("param_")]
                metric_columns = [c for c in filtered_df.columns if c.startswith("metric_")]
                
                selected_params = st.multiselect(
                    "Select parameters to display",
                    options=[c.replace("param_", "") for c in param_columns],
                    default=[c.replace("param_", "") for c in param_columns[:min(3, len(param_columns))]]
                )
                
                selected_metrics = st.multiselect(
                    "Select metrics to display",
                    options=[c.replace("metric_", "") for c in metric_columns],
                    default=[c.replace("metric_", "") for c in metric_columns[:min(3, len(metric_columns))]]
                )
                
                # Add selected columns to display list
                display_columns += [f"param_{p}" for p in selected_params]
                display_columns += [f"metric_{m}" for m in selected_metrics]
                
                # Filter the DataFrame to only include selected columns
                display_df = filtered_df[display_columns].copy()
                
                # Rename columns for display
                column_mapping = {
                    "name": "Name",
                    "id": "ID",
                    "state": "State"
                }
                
                for param in selected_params:
                    column_mapping[f"param_{param}"] = f"P: {param}"
                
                for metric in selected_metrics:
                    column_mapping[f"metric_{metric}"] = f"M: {metric}"
                
                display_df = display_df.rename(columns=column_mapping)
                
                # Display the table
                st.dataframe(display_df, use_container_width=True)
                
                # Export data button
                if st.button("📥 Export sweep runs to CSV"):
                    csv_data, filename = export_to_csv(display_df, f"{sweep_id}_runs")
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
                
                # View selected run button
                selected_run_id = st.selectbox(
                    "Select a run to view details",
                    options=filtered_df["id"].tolist(),
                    format_func=lambda x: f"{x} - {filtered_df[filtered_df['id'] == x]['name'].iloc[0]}"
                )
                
                if st.button("View Run Details"):
                    run_info = next((r for r in runs if r["id"] == selected_run_id), None)
                    if run_info:
                        st.session_state.selected_run = run_info
                        st.session_state.current_page = "run_details"
                        st.rerun()
        else:
            st.info("No runs available for this sweep.")
    
    with tab_visualization:
        st.subheader("Sweep Visualizations")
        
        if "runs" in sweep_data and sweep_data["runs"]:
            runs = sweep_data["runs"]
            
            # Find all metrics across runs
            all_metrics = set()
            for run in runs:
                if "summary" in run:
                    for key, value in run["summary"].items():
                        if isinstance(value, (int, float)):
                            all_metrics.add(key)
            
            if all_metrics:
                # Let user select a metric to visualize
                selected_metric = st.selectbox(
                    "Select a metric to visualize",
                    options=sorted(list(all_metrics))
                )
                
                # Visualize sweep results
                sweep_fig = plot_sweep_results(sweep_data, selected_metric)
                if sweep_fig:
                    st.plotly_chart(sweep_fig, use_container_width=True)
                
                # Visualize parameter importance
                st.subheader("Parameter Importance")
                param_fig = plot_parameter_importance(runs, selected_metric)
                if param_fig:
                    st.plotly_chart(param_fig, use_container_width=True)
            else:
                st.info("No metrics available to visualize.")
        else:
            st.info("No runs available to visualize.")
