import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
# Import visualization functions directly from the correct module
from utils.visualization import plot_metrics_history, create_parallel_coordinates_plot

def run_details():
    """
    View detailed information about a selected W&B run
    """
    if not st.session_state.authenticated or not st.session_state.api:
        st.warning("Please authenticate with W&B first")
        return
        
    if not st.session_state.selected_project:
        st.warning("Please select a project first")
        return
        
    if not st.session_state.selected_run:
        st.warning("Please select a run to view its details")
        return
        
    st.header("Run Details")
    
    try:
        # Get user and project
        user = st.session_state.api.viewer()['entity']
        project_name = st.session_state.selected_project
        run_id = st.session_state.selected_run
        
        # Load the run
        run = st.session_state.api.run(f"{user}/{project_name}/{run_id}")
        
        # Overview section
        st.subheader("Overview")
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown(f"**Run Name:** {run.name}")
            st.markdown(f"**Run ID:** {run.id}")
            st.markdown(f"**Status:** {run.state}")
            
        with cols[1]:
            created_time = datetime.fromtimestamp(run.created_at).strftime('%Y-%m-%d %H:%M:%S') if run.created_at else "N/A"
            runtime = f"{round(run.runtime / 60, 2)} minutes" if run.runtime else "N/A"
            
            st.markdown(f"**Created:** {created_time}")
            st.markdown(f"**Runtime:** {runtime}")
            st.markdown(f"**Heartbeat:** {datetime.fromtimestamp(run.heartbeat_at).strftime('%Y-%m-%d %H:%M:%S') if run.heartbeat_at else 'N/A'}")
            
        with cols[2]:
            st.markdown(f"**Tags:** {', '.join(run.tags) if run.tags else 'None'}")
            st.markdown(f"**User:** {run.user.name if hasattr(run, 'user') and run.user else 'N/A'}")
            
            # Generate W&B URL to the run
            wandb_url = f"https://wandb.ai/{user}/{project_name}/runs/{run_id}"
            st.markdown(f"[View in W&B]({wandb_url})")
        
        # Create tabs for different sections
        config_tab, metrics_tab, history_tab, files_tab = st.tabs([
            "Configuration", "Metrics", "History", "Files & Artifacts"
        ])
        
        # Configuration tab
        with config_tab:
            st.subheader("Run Configuration")
            
            # Get run config
            config = run.config
            
            if not config:
                st.info("No configuration data available for this run")
            else:
                # Convert config to flat structure
                flat_config = {}
                
                def flatten_dict(d, parent_key=''):
                    for k, v in d.items():
                        key = f"{parent_key}.{k}" if parent_key else k
                        if isinstance(v, dict):
                            flatten_dict(v, key)
                        else:
                            flat_config[key] = v
                
                flatten_dict(config)
                
                # Group configs by categories
                config_categories = {}
                
                for key, value in flat_config.items():
                    category = key.split('.')[0] if '.' in key else 'general'
                    if category not in config_categories:
                        config_categories[category] = {}
                    config_categories[category][key] = value
                
                # Display config by categories
                for category, params in config_categories.items():
                    st.markdown(f"#### {category.capitalize()}")
                    
                    # Convert to DataFrame for better display
                    params_df = pd.DataFrame({
                        "Parameter": params.keys(),
                        "Value": params.values()
                    })
                    
                    st.dataframe(params_df, use_container_width=True)
                
                # Option to download config as JSON
                config_json = pd.io.json.dumps(config)
                st.download_button(
                    label="Download Config as JSON",
                    data=config_json,
                    file_name=f"wandb_run_{run.id}_config.json",
                    mime="application/json"
                )
        
        # Metrics tab
        with metrics_tab:
            st.subheader("Summary Metrics")
            
            summary = run.summary
            
            if not summary or all(k.startswith('_') for k in summary.keys()):
                st.info("No summary metrics available for this run")
            else:
                # Filter out internal keys and non-numeric values
                metrics = {k: v for k, v in summary.items() 
                          if not k.startswith('_') and isinstance(v, (int, float))}
                
                if not metrics:
                    st.info("No numeric metrics available for this run")
                else:
                    # Convert to DataFrame for better display
                    metrics_df = pd.DataFrame({
                        "Metric": metrics.keys(),
                        "Value": metrics.values()
                    })
                    
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Allow user to select metrics to visualize
                    selected_metrics = st.multiselect(
                        "Select metrics to visualize",
                        options=list(metrics.keys()),
                        default=list(metrics.keys())[:min(3, len(metrics))]
                    )
                    
                    if selected_metrics:
                        # Create a bar chart for selected metrics
                        fig = px.bar(
                            x=selected_metrics,
                            y=[metrics[m] for m in selected_metrics],
                            labels={'x': 'Metric', 'y': 'Value'},
                            title="Summary Metrics"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # History tab
        with history_tab:
            st.subheader("Metrics History")
            
            # Get run history
            try:
                history = run.scan_history()
                history_df = pd.DataFrame(history)
                
                if history_df.empty:
                    st.info("No history data available for this run")
                else:
                    # Filter out internal columns and keep only step and numeric columns
                    valid_columns = ['_step']
                    numeric_columns = []
                    
                    for col in history_df.columns:
                        if col == '_step':
                            continue
                        if col.startswith('_'):
                            continue
                        if pd.api.types.is_numeric_dtype(history_df[col]):
                            valid_columns.append(col)
                            numeric_columns.append(col)
                    
                    if not numeric_columns:
                        st.info("No numeric metrics found in history data")
                    else:
                        # Metric selection
                        selected_metrics = st.multiselect(
                            "Select metrics to plot",
                            options=numeric_columns,
                            default=numeric_columns[:min(2, len(numeric_columns))]
                        )
                        
                        if selected_metrics:
                            # Create metric history plots
                            fig = plot_metrics_history(history_df, selected_metrics)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show data table with history
                            st.subheader("Metrics Table")
                            
                            # Allow filtering by steps
                            if '_step' in history_df.columns:
                                min_step = int(history_df['_step'].min())
                                max_step = int(history_df['_step'].max())
                                
                                step_range = st.slider(
                                    "Step Range",
                                    min_value=min_step,
                                    max_value=max_step,
                                    value=(min_step, max_step)
                                )
                                
                                filtered_df = history_df[
                                    (history_df['_step'] >= step_range[0]) & 
                                    (history_df['_step'] <= step_range[1])
                                ]
                            else:
                                filtered_df = history_df
                            
                            # Allow downloading history data
                            csv = filtered_df.to_csv(index=False)
                            st.download_button(
                                label="Download History CSV",
                                data=csv,
                                file_name=f"wandb_run_{run.id}_history.csv",
                                mime="text/csv"
                            )
                            
                            # Display history table
                            display_columns = ['_step'] + selected_metrics
                            st.dataframe(filtered_df[display_columns], use_container_width=True)
            except Exception as e:
                st.error(f"Error loading history data: {str(e)}")
        
        # Files tab
        with files_tab:
            st.subheader("Files & Artifacts")
            
            # Show files
            files = run.files()
            
            if not files:
                st.info("No files found for this run")
            else:
                st.markdown("#### Files")
                
                # Create DataFrame of files
                files_data = []
                for file in files:
                    files_data.append({
                        "Name": file.name,
                        "Size (KB)": round(file.size / 1024, 2) if hasattr(file, 'size') else "N/A",
                        "Updated": datetime.fromtimestamp(file.updated_at).strftime('%Y-%m-%d %H:%M:%S') if hasattr(file, 'updated_at') else "N/A"
                    })
                
                files_df = pd.DataFrame(files_data)
                st.dataframe(files_df, use_container_width=True)
                
                # Allow downloading selected file
                if not files_df.empty:
                    selected_file = st.selectbox(
                        "Select file to download",
                        options=files_df["Name"].tolist()
                    )
                    
                    if st.button("Download Selected File"):
                        with st.spinner(f"Downloading {selected_file}..."):
                            try:
                                for file in files:
                                    if file.name == selected_file:
                                        file.download(replace=True)
                                        st.success(f"File {selected_file} downloaded successfully to the current directory")
                                        break
                            except Exception as e:
                                st.error(f"Error downloading file: {str(e)}")
            
            # Show artifacts
            try:
                artifacts = run.logged_artifacts()
                
                if not artifacts:
                    st.info("No artifacts found for this run")
                else:
                    st.markdown("#### Artifacts")
                    
                    # Create DataFrame of artifacts
                    artifacts_data = []
                    for artifact in artifacts:
                        artifacts_data.append({
                            "Name": artifact.name,
                            "Type": artifact.type,
                            "Version": artifact.version,
                            "Size (MB)": round(artifact.size / (1024 * 1024), 2) if hasattr(artifact, 'size') else "N/A",
                            "Created": datetime.fromtimestamp(artifact.created_at).strftime('%Y-%m-%d %H:%M:%S') if hasattr(artifact, 'created_at') else "N/A"
                        })
                    
                    artifacts_df = pd.DataFrame(artifacts_data)
                    st.dataframe(artifacts_df, use_container_width=True)
                    
                    # Button to go to Artifact Manager for more detailed handling
                    if st.button("Go to Artifact Manager"):
                        st.session_state.active_tab = "Artifacts"
                        st.rerun()
            except Exception as e:
                st.info(f"Artifact information not available: {str(e)}")
        
    except Exception as e:
        st.error(f"Error loading run details: {str(e)}")
