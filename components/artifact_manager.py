import streamlit as st
import pandas as pd
import os
import tempfile
from datetime import datetime

def artifact_manager():
    """
    Download and manage artifacts from W&B runs
    """
    if not st.session_state.authenticated or not st.session_state.api:
        st.warning("Please authenticate with W&B first")
        return
        
    if not st.session_state.selected_project:
        st.warning("Please select a project first")
        return
        
    st.header("Artifact Manager")
    
    try:
        # Get user and project
        user = st.session_state.api.viewer()['entity']
        project_name = st.session_state.selected_project
        
        # Fetch artifacts for the project
        artifacts = st.session_state.api.artifacts(f"{user}/{project_name}")
        
        if not artifacts:
            st.info("No artifacts found in this project")
            return
            
        # Extract artifact data
        artifacts_data = []
        for artifact in artifacts:
            artifacts_data.append({
                "Name": artifact.name,
                "Type": artifact.type,
                "Version": artifact.version,
                "Alias": ", ".join(artifact.aliases) if hasattr(artifact, 'aliases') else "",
                "Size (MB)": round(artifact.size / (1024 * 1024), 2) if hasattr(artifact, 'size') else "N/A",
                "Created": datetime.fromtimestamp(artifact.created_at).strftime('%Y-%m-%d %H:%M:%S') if hasattr(artifact, 'created_at') else "N/A",
                "ID": artifact.id
            })
        
        artifacts_df = pd.DataFrame(artifacts_data)
        
        # Filter options
        st.subheader("Filter Artifacts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter by type
            types = ["All Types"] + sorted(artifacts_df["Type"].unique().tolist())
            selected_type = st.selectbox("Artifact Type", options=types)
        
        with col2:
            # Search by name
            search_term = st.text_input("Search by Name")
        
        # Apply filters
        filtered_df = artifacts_df
        
        if selected_type != "All Types":
            filtered_df = filtered_df[filtered_df["Type"] == selected_type]
            
        if search_term:
            filtered_df = filtered_df[
                filtered_df["Name"].str.contains(search_term, case=False, na=False)
            ]
        
        # Sort options
        sort_options = {
            "Created (Newest First)": ("Created", False),
            "Created (Oldest First)": ("Created", True),
            "Name (A-Z)": ("Name", True),
            "Name (Z-A)": ("Name", False),
            "Size (Largest First)": ("Size (MB)", False),
            "Size (Smallest First)": ("Size (MB)", True)
        }
        
        sort_by = st.selectbox(
            "Sort by",
            options=list(sort_options.keys()),
            index=0
        )
        
        sort_column, ascending = sort_options[sort_by]
        filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending)
        
        # Display artifacts table
        st.subheader("Artifacts")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Select artifact to view details
        if not filtered_df.empty:
            selected_artifact_id = st.selectbox(
                "Select an artifact to view details",
                options=filtered_df["ID"].tolist(),
                format_func=lambda x: f"{filtered_df[filtered_df['ID']==x]['Name'].iloc[0]} (v{filtered_df[filtered_df['ID']==x]['Version'].iloc[0]})"
            )
            
            # Get the selected artifact
            selected_artifact = None
            for artifact in artifacts:
                if artifact.id == selected_artifact_id:
                    selected_artifact = artifact
                    break
            
            if selected_artifact:
                st.subheader(f"Artifact Details: {selected_artifact.name} (v{selected_artifact.version})")
                
                # Display artifact metadata
                cols = st.columns(3)
                
                with cols[0]:
                    st.markdown(f"**Type:** {selected_artifact.type}")
                    st.markdown(f"**Version:** {selected_artifact.version}")
                    
                with cols[1]:
                    st.markdown(f"**Size:** {round(selected_artifact.size / (1024 * 1024), 2)} MB")
                    st.markdown(f"**Created:** {datetime.fromtimestamp(selected_artifact.created_at).strftime('%Y-%m-%d %H:%M:%S')}")
                    
                with cols[2]:
                    st.markdown(f"**ID:** {selected_artifact.id}")
                    
                    # Generate W&B URL to the artifact
                    wandb_url = f"https://wandb.ai/{user}/{project_name}/artifacts/{selected_artifact.type}/{selected_artifact.name}/v{selected_artifact.version}"
                    st.markdown(f"[View in W&B]({wandb_url})")
                
                # Display artifact files
                st.markdown("### Files")
                
                try:
                    # Get and display the files in the artifact
                    manifest = selected_artifact.manifest
                    
                    if not manifest or not hasattr(manifest, 'entries'):
                        st.info("No file manifest available for this artifact")
                    else:
                        # Extract file information
                        files_data = []
                        
                        for entry in manifest.entries:
                            entry_data = {
                                "Path": entry.path,
                                "Size (KB)": round(entry.size / 1024, 2) if hasattr(entry, 'size') else "N/A",
                                "Type": entry.path.split(".")[-1] if "." in entry.path else "Unknown"
                            }
                            files_data.append(entry_data)
                        
                        files_df = pd.DataFrame(files_data)
                        
                        # Display files table
                        st.dataframe(files_df, use_container_width=True)
                        
                        # Download options
                        st.markdown("### Download Options")
                        
                        download_options = st.radio(
                            "Download",
                            options=["Selected File", "All Files"],
                            horizontal=True
                        )
                        
                        if download_options == "Selected File":
                            if not files_df.empty:
                                selected_file = st.selectbox(
                                    "Select file to download",
                                    options=files_df["Path"].tolist()
                                )
                                
                                if st.button("Download Selected File"):
                                    with st.spinner(f"Downloading {selected_file}..."):
                                        try:
                                            # Create a temp directory to download to
                                            with tempfile.TemporaryDirectory() as tmp_dir:
                                                selected_artifact.download(root=tmp_dir)
                                                file_path = os.path.join(tmp_dir, selected_file)
                                                
                                                if os.path.exists(file_path):
                                                    with open(file_path, 'rb') as f:
                                                        file_content = f.read()
                                                        
                                                    st.download_button(
                                                        label=f"Save {selected_file}",
                                                        data=file_content,
                                                        file_name=os.path.basename(selected_file)
                                                    )
                                                else:
                                                    st.error(f"File {selected_file} not found in the artifact")
                                        except Exception as e:
                                            st.error(f"Error downloading file: {str(e)}")
                        else:  # All Files
                            if st.button("Download All Files"):
                                with st.spinner("Preparing artifact download..."):
                                    try:
                                        st.info("Artifacts will be downloaded to the current working directory")
                                        download_path = selected_artifact.download()
                                        st.success(f"Artifact downloaded successfully to {download_path}")
                                    except Exception as e:
                                        st.error(f"Error downloading artifact: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error loading artifact files: {str(e)}")
                
                # Display artifact lineage
                st.markdown("### Artifact Lineage")
                
                try:
                    # Collections that use this artifact
                    if hasattr(selected_artifact, 'used_by'):
                        used_by = selected_artifact.used_by
                        
                        if used_by:
                            st.markdown("#### Used By")
                            
                            used_by_data = []
                            for collection in used_by:
                                used_by_data.append({
                                    "Type": collection.type,
                                    "Name": collection.name,
                                    "Version": collection.version
                                })
                            
                            used_by_df = pd.DataFrame(used_by_data)
                            st.dataframe(used_by_df, use_container_width=True)
                        else:
                            st.info("This artifact is not used by any collection")
                    
                    # Collections that this artifact uses
                    if hasattr(selected_artifact, 'uses'):
                        uses = selected_artifact.uses
                        
                        if uses:
                            st.markdown("#### Uses")
                            
                            uses_data = []
                            for collection in uses:
                                uses_data.append({
                                    "Type": collection.type,
                                    "Name": collection.name,
                                    "Version": collection.version
                                })
                            
                            uses_df = pd.DataFrame(uses_data)
                            st.dataframe(uses_df, use_container_width=True)
                        else:
                            st.info("This artifact does not use any other collection")
                    
                except Exception as e:
                    st.info(f"Lineage information not available: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
