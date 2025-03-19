import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import wandb
from datetime import datetime
import json
import io
from PIL import Image

def pull_from_wandb_component():
    """Pull experiments and data from W&B for analysis"""
    st.subheader("Pull from W&B")
    st.markdown("""
    Pull your quantum experiment data from W&B for analysis and visualization.
    Download metrics, artifacts, and results.
    """)
    
    try:
        api = wandb.Api()
        
        # Project selection
        with st.spinner("Loading projects..."):
            projects = []
            for project in api.projects():
                projects.append({
                    "name": project.name,
                    "entity": project.entity,
                    "id": f"{project.entity}/{project.name}",
                })
            
            if not projects:
                st.info("No projects found. Create a project first from the 'Push to W&B' tab.")
                return
            
            project_options = [f"{p['entity']}/{p['name']}" for p in projects]
            selected_project = st.selectbox("Select project:", project_options)
            
            if selected_project:
                entity, project_name = selected_project.split("/")
                
                # Get runs for this project
                runs = api.runs(path=f"{entity}/{project_name}")
                
                # Prepare data for display
                run_data = []
                for run in runs:
                    run_data.append({
                        "id": run.id,
                        "name": run.name,
                        "status": run.state,
                        "created": run.created_at,
                        "circuit_type": run.config.get("circuit_type", "N/A") if hasattr(run, "config") else "N/A",
                        "n_qubits": run.config.get("n_qubits", "N/A") if hasattr(run, "config") else "N/A",
                        "shots": run.config.get("shots", "N/A") if hasattr(run, "config") else "N/A",
                        "tags": ", ".join(run.tags) if hasattr(run, "tags") and run.tags else "N/A"
                    })
                
                if run_data:
                    # Display runs in a dataframe
                    st.subheader(f"Runs in {selected_project}")
                    df = pd.DataFrame(run_data)
                    st.dataframe(df)
                    
                    # Allow selecting a run to view details
                    run_options = [f"{run['name']} ({run['id']})" for run in run_data]
                    selected_run_option = st.selectbox("Select a run to view details:", run_options)
                    
                    if selected_run_option:
                        # Get run ID from selected option
                        selected_run_id = selected_run_option.split("(")[-1].split(")")[0]
                        
                        # Get detailed run information
                        with st.spinner("Loading run details..."):
                            selected_run = api.run(f"{entity}/{project_name}/{selected_run_id}")
                            
                            # Display run details
                            st.subheader(f"Run Details: {selected_run.name}")
                            
                            # Basic info
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Status", selected_run.state)
                            with col2:
                                st.metric("Circuit Type", selected_run.config.get("circuit_type", "N/A"))
                            with col3:
                                st.metric("Qubits", selected_run.config.get("n_qubits", "N/A"))
                            
                            # Configuration
                            st.subheader("Configuration")
                            config_df = pd.DataFrame([{"Key": k, "Value": str(v)} for k, v in selected_run.config.items()])
                            st.dataframe(config_df)
                            
                            # Artifacts and images
                            st.subheader("Artifacts and Images")
                            
                            try:
                                # Try to download files
                                files = selected_run.files()
                                media_files = [f for f in files if f.name.endswith(('.png', '.jpg', '.jpeg'))]
                                
                                if media_files:
                                    st.markdown("##### Media Files")
                                    
                                    for media_file in media_files:
                                        try:
                                            # Download the file
                                            media_content = media_file.download(replace=True)
                                            
                                            # Display the image
                                            image = Image.open(io.BytesIO(media_content))
                                            st.image(image, caption=media_file.name)
                                        except Exception as e:
                                            st.warning(f"Could not load image {media_file.name}: {str(e)}")
                                
                                # Look for wandb artifact tables
                                tables = []
                                for artifact in selected_run.logged_artifacts():
                                    if "table" in artifact.type:
                                        tables.append(artifact)
                                
                                if tables:
                                    st.markdown("##### Tables")
                                    
                                    for table_artifact in tables:
                                        st.markdown(f"**{table_artifact.name}**")
                                        try:
                                            table = artifact.get("table")
                                            df = pd.DataFrame(table.data, columns=table.columns)
                                            st.dataframe(df)
                                        except Exception as e:
                                            st.warning(f"Could not load table: {str(e)}")
                            
                            except Exception as e:
                                st.warning(f"Error accessing run artifacts: {str(e)}")
                            
                            # Link to WandB dashboard
                            st.markdown("---")
                            st.markdown(f"[View Run in W&B Dashboard](https://wandb.ai/{entity}/{project_name}/runs/{selected_run_id})")
                            
                            # Download options
                            st.subheader("Download Options")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Download Run Summary"):
                                    # Generate summary as JSON
                                    summary = {
                                        "id": selected_run.id,
                                        "name": selected_run.name,
                                        "state": selected_run.state,
                                        "created_at": selected_run.created_at,
                                        "config": selected_run.config,
                                        "summary": selected_run.summary._json_dict
                                    }
                                    
                                    # Convert to downloadable format
                                    json_str = json.dumps(summary, indent=2)
                                    json_bytes = json_str.encode()
                                    
                                    # Create download button
                                    st.download_button(
                                        label="Download JSON",
                                        data=json_bytes,
                                        file_name=f"{selected_run.id}_summary.json",
                                        mime="application/json"
                                    )
                            
                            with col2:
                                if st.button("Download Run Config"):
                                    # Convert to downloadable format
                                    json_str = json.dumps(dict(selected_run.config), indent=2)
                                    json_bytes = json_str.encode()
                                    
                                    # Create download button
                                    st.download_button(
                                        label="Download Config",
                                        data=json_bytes,
                                        file_name=f"{selected_run.id}_config.json",
                                        mime="application/json"
                                    )
                else:
                    st.info(f"No runs found in {selected_project}. Create a run from the 'Push to W&B' tab.")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure you are properly authenticated with W&B.")