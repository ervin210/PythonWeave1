import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime

def data_export():
    """
    Export data from W&B runs and projects to various formats
    """
    if not st.session_state.authenticated or not st.session_state.api:
        st.warning("Please authenticate with W&B first")
        return
        
    if not st.session_state.selected_project:
        st.warning("Please select a project first")
        return
        
    st.header("Data Export")
    
    # Export options
    export_type = st.radio(
        "Select data to export",
        ["Run Metrics", "Run History", "Run Config", "Sweep Results", "Project Summary"],
        horizontal=True
    )
    
    try:
        # Get user and project
        user = st.session_state.api.viewer()['entity']
        project_name = st.session_state.selected_project
        
        # Run Metrics export
        if export_type == "Run Metrics":
            st.subheader("Export Run Metrics")
            
            # Get all runs for the project
            runs = st.session_state.api.runs(f"{user}/{project_name}")
            
            if not runs:
                st.info("No runs found in this project")
                return
                
            # Run selection
            run_options = []
            for run in runs:
                run_options.append(f"{run.id} - {run.name}")
            
            selected_runs = st.multiselect(
                "Select runs to export",
                options=run_options
            )
            
            if not selected_runs:
                st.warning("Please select at least one run")
                return
                
            # Get run IDs from selection
            selected_run_ids = [run.split(" - ")[0] for run in selected_runs]
            
            # Collect metrics from selected runs
            metrics_data = []
            all_metrics = set()
            
            for run_id in selected_run_ids:
                run = st.session_state.api.run(f"{user}/{project_name}/{run_id}")
                
                if not hasattr(run, 'summary') or not run.summary:
                    continue
                    
                run_metrics = {
                    "run_id": run.id,
                    "run_name": run.name,
                    "state": run.state
                }
                
                # Add summary metrics
                for key, value in run.summary.items():
                    if not key.startswith('_') and isinstance(value, (int, float, str, bool)) or value is None:
                        run_metrics[key] = value
                        all_metrics.add(key)
                
                metrics_data.append(run_metrics)
            
            if not metrics_data:
                st.warning("No metrics data found for selected runs")
                return
                
            # Create DataFrame with metrics
            metrics_df = pd.DataFrame(metrics_data)
            
            # Preview the data
            st.markdown("### Preview")
            st.dataframe(metrics_df, use_container_width=True)
            
            # Export options
            st.markdown("### Export Format")
            export_format = st.radio(
                "Select format",
                ["CSV", "Excel", "JSON"],
                horizontal=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                file_name = st.text_input("File name (without extension)", f"wandb_metrics_{project_name}_{datetime.now().strftime('%Y%m%d')}")
            
            with col2:
                if export_format == "CSV":
                    csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{file_name}.csv",
                        mime="text/csv"
                    )
                    
                elif export_format == "Excel":
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
                    
                    excel_data = buffer.getvalue()
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name=f"{file_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                elif export_format == "JSON":
                    json_str = metrics_df.to_json(orient='records')
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"{file_name}.json",
                        mime="application/json"
                    )
        
        # Run History export
        elif export_type == "Run History":
            st.subheader("Export Run History")
            
            # Must select a run first
            if not st.session_state.selected_run:
                # Let user select a run
                runs = st.session_state.api.runs(f"{user}/{project_name}")
                
                if not runs:
                    st.info("No runs found in this project")
                    return
                    
                run_options = []
                for run in runs:
                    run_options.append(f"{run.id} - {run.name}")
                
                selected_run_option = st.selectbox(
                    "Select a run",
                    options=run_options
                )
                
                selected_run_id = selected_run_option.split(" - ")[0]
            else:
                selected_run_id = st.session_state.selected_run
            
            # Load the run
            run = st.session_state.api.run(f"{user}/{project_name}/{selected_run_id}")
            
            st.markdown(f"### History for run: {run.name} ({run.id})")
            
            # Get run history
            try:
                history = run.scan_history()
                history_df = pd.DataFrame(history)
                
                if history_df.empty:
                    st.info("No history data available for this run")
                    return
                
                # Filter out internal columns and system metrics if requested
                include_internal = st.checkbox("Include internal metrics (starting with _)", value=False)
                
                if not include_internal:
                    history_df = history_df[[col for col in history_df.columns if not col.startswith('_') or col == '_step']]
                
                # Preview data
                st.markdown("### Preview")
                
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
                
                # Select specific columns to export
                all_columns = filtered_df.columns.tolist()
                selected_columns = st.multiselect(
                    "Select columns to export",
                    options=all_columns,
                    default=all_columns
                )
                
                if not selected_columns:
                    st.warning("Please select at least one column")
                    return
                
                export_df = filtered_df[selected_columns]
                
                # Show sample of data
                st.dataframe(export_df.head(10), use_container_width=True)
                
                # Export options
                st.markdown("### Export Format")
                export_format = st.radio(
                    "Select format",
                    ["CSV", "Excel", "JSON"],
                    horizontal=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    file_name = st.text_input("File name (without extension)", f"wandb_history_{run.id}_{datetime.now().strftime('%Y%m%d')}")
                
                with col2:
                    if export_format == "CSV":
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{file_name}.csv",
                            mime="text/csv"
                        )
                        
                    elif export_format == "Excel":
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            export_df.to_excel(writer, sheet_name="History", index=False)
                        
                        excel_data = buffer.getvalue()
                        st.download_button(
                            label="Download Excel",
                            data=excel_data,
                            file_name=f"{file_name}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                    elif export_format == "JSON":
                        json_str = export_df.to_json(orient='records')
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"{file_name}.json",
                            mime="application/json"
                        )
                
            except Exception as e:
                st.error(f"Error loading history data: {str(e)}")
        
        # Run Config export
        elif export_type == "Run Config":
            st.subheader("Export Run Configuration")
            
            # Get all runs for the project
            runs = st.session_state.api.runs(f"{user}/{project_name}")
            
            if not runs:
                st.info("No runs found in this project")
                return
                
            # Run selection
            run_options = []
            for run in runs:
                run_options.append(f"{run.id} - {run.name}")
            
            selected_runs = st.multiselect(
                "Select runs to export configurations",
                options=run_options
            )
            
            if not selected_runs:
                st.warning("Please select at least one run")
                return
                
            # Get run IDs from selection
            selected_run_ids = [run.split(" - ")[0] for run in selected_runs]
            
            # Whether to flatten nested configs
            flatten_config = st.checkbox("Flatten nested configurations", value=True)
            
            # Collect configs from selected runs
            configs_data = []
            all_config_keys = set()
            
            for run_id in selected_run_ids:
                run = st.session_state.api.run(f"{user}/{project_name}/{run_id}")
                
                if not run.config:
                    continue
                    
                if flatten_config:
                    # Flatten the config
                    flat_config = {"run_id": run.id, "run_name": run.name}
                    
                    def flatten_dict(d, parent_key=''):
                        for k, v in d.items():
                            key = f"{parent_key}.{k}" if parent_key else k
                            if isinstance(v, dict):
                                flatten_dict(v, key)
                            else:
                                if isinstance(v, (int, float, str, bool)) or v is None:
                                    flat_config[key] = v
                                    all_config_keys.add(key)
                    
                    flatten_dict(run.config)
                    configs_data.append(flat_config)
                else:
                    # Keep original structure but convert to JSON for export
                    configs_data.append({
                        "run_id": run.id,
                        "run_name": run.name,
                        "config": run.config
                    })
            
            if not configs_data:
                st.warning("No configuration data found for selected runs")
                return
            
            # Preview and export
            if flatten_config:
                # Create DataFrame with configs
                configs_df = pd.DataFrame(configs_data)
                
                # Preview the data
                st.markdown("### Preview")
                st.dataframe(configs_df, use_container_width=True)
                
                # Export options
                st.markdown("### Export Format")
                export_format = st.radio(
                    "Select format",
                    ["CSV", "Excel", "JSON"],
                    horizontal=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    file_name = st.text_input("File name (without extension)", f"wandb_configs_{project_name}_{datetime.now().strftime('%Y%m%d')}")
                
                with col2:
                    if export_format == "CSV":
                        csv = configs_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{file_name}.csv",
                            mime="text/csv"
                        )
                        
                    elif export_format == "Excel":
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            configs_df.to_excel(writer, sheet_name="Configs", index=False)
                        
                        excel_data = buffer.getvalue()
                        st.download_button(
                            label="Download Excel",
                            data=excel_data,
                            file_name=f"{file_name}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                    elif export_format == "JSON":
                        json_str = configs_df.to_json(orient='records')
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"{file_name}.json",
                            mime="application/json"
                        )
            else:
                # Export as JSON only for non-flattened configs
                json_str = json.dumps(configs_data, indent=2)
                
                st.markdown("### Preview")
                st.code(json_str[:5000] + ("..." if len(json_str) > 5000 else ""), language="json")
                
                file_name = st.text_input("File name (without extension)", f"wandb_configs_{project_name}_{datetime.now().strftime('%Y%m%d')}")
                
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"{file_name}.json",
                    mime="application/json"
                )
        
        # Sweep Results export
        elif export_type == "Sweep Results":
            st.subheader("Export Sweep Results")
            
            # Must select a sweep
            if not st.session_state.selected_sweep:
                # Let user select a sweep
                try:
                    sweeps = st.session_state.api.sweeps(f"{user}/{project_name}")
                    
                    if not sweeps:
                        st.info("No sweeps found in this project")
                        return
                        
                    sweep_options = []
                    for sweep in sweeps:
                        sweep_name = sweep.name or sweep.id
                        sweep_options.append(f"{sweep.id} - {sweep_name}")
                    
                    selected_sweep_option = st.selectbox(
                        "Select a sweep",
                        options=sweep_options
                    )
                    
                    selected_sweep_id = selected_sweep_option.split(" - ")[0]
                except Exception as e:
                    st.error(f"Error loading sweeps: {str(e)}")
                    return
            else:
                selected_sweep_id = st.session_state.selected_sweep
            
            # Load the sweep
            sweep = st.session_state.api.sweep(f"{user}/{project_name}/{selected_sweep_id}")
            
            st.markdown(f"### Results for sweep: {sweep.name or sweep.id}")
            
            # Load all runs in the sweep
            runs = list(sweep.runs)
            
            if not runs:
                st.info("No runs found in this sweep")
                return
                
            # Extract run data
            runs_data = []
            config_keys = set()
            metric_keys = set()
            
            for run in runs:
                run_data = {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": datetime.fromtimestamp(run.created_at).strftime('%Y-%m-%d %H:%M:%S'),
                    "runtime": round((run.runtime or 0) / 60, 2)  # minutes
                }
                
                # Extract config parameters
                if run.config:
                    # Option to flatten config
                    flatten_config = st.checkbox("Flatten nested configurations", value=True)
                    
                    if flatten_config:
                        def flatten_dict(d, parent_key=''):
                            for k, v in d.items():
                                key = f"{parent_key}.{k}" if parent_key else k
                                if isinstance(v, dict):
                                    flatten_dict(v, key)
                                else:
                                    if isinstance(v, (int, float, str, bool)) or v is None:
                                        run_data[f"config.{key}"] = v
                                        config_keys.add(f"config.{key}")
                        
                        flatten_dict(run.config)
                    else:
                        for key, value in run.config.items():
                            if isinstance(value, (int, float, str, bool)) or value is None:
                                run_data[f"config.{key}"] = value
                                config_keys.add(f"config.{key}")
                
                # Extract summary metrics
                if hasattr(run, 'summary') and run.summary:
                    for key, value in run.summary.items():
                        if not key.startswith('_') and isinstance(value, (int, float)) and not np.isnan(value):
                            run_data[key] = value
                            metric_keys.add(key)
                
                runs_data.append(run_data)
            
            # Create dataframe
            sweep_df = pd.DataFrame(runs_data)
            
            # Select columns to export
            st.markdown("### Select Data to Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                include_configs = st.checkbox("Include configurations", value=True)
            
            with col2:
                include_metrics = st.checkbox("Include metrics", value=True)
            
            # Filter columns based on selection
            columns_to_include = ["id", "name", "state", "created_at", "runtime"]
            
            if include_configs:
                config_columns = sorted(list(config_keys))
                columns_to_include.extend(config_columns)
            
            if include_metrics:
                metric_columns = sorted(list(metric_keys))
                columns_to_include.extend(metric_columns)
            
            # Filter the dataframe
            export_df = sweep_df[columns_to_include]
            
            # Preview the data
            st.markdown("### Preview")
            st.dataframe(export_df, use_container_width=True)
            
            # Export options
            st.markdown("### Export Format")
            export_format = st.radio(
                "Select format",
                ["CSV", "Excel", "JSON"],
                horizontal=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                file_name = st.text_input("File name (without extension)", f"wandb_sweep_{selected_sweep_id}_{datetime.now().strftime('%Y%m%d')}")
            
            with col2:
                if export_format == "CSV":
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{file_name}.csv",
                        mime="text/csv"
                    )
                    
                elif export_format == "Excel":
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        export_df.to_excel(writer, sheet_name="Sweep Results", index=False)
                    
                    excel_data = buffer.getvalue()
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name=f"{file_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                elif export_format == "JSON":
                    json_str = export_df.to_json(orient='records')
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"{file_name}.json",
                        mime="application/json"
                    )
        
        # Project Summary export
        elif export_type == "Project Summary":
            st.subheader("Export Project Summary")
            
            # Get all runs for the project
            runs = st.session_state.api.runs(f"{user}/{project_name}")
            
            if not runs:
                st.info("No runs found in this project")
                return
            
            # Get basic project info
            project = st.session_state.api.project(user, project_name)
            
            # Create project summary
            project_data = {
                "name": project.name,
                "entity": user,
                "description": project.description or "No description",
                "created_at": datetime.fromtimestamp(project.created_at).strftime('%Y-%m-%d %H:%M:%S') if hasattr(project, 'created_at') else "N/A",
                "total_runs": len(runs),
            }
            
            # Count runs by state
            run_states = {}
            for run in runs:
                state = run.state
                if state in run_states:
                    run_states[state] += 1
                else:
                    run_states[state] = 1
            
            for state, count in run_states.items():
                project_data[f"runs_{state}"] = count
            
            # Get sweeps
            try:
                sweeps = st.session_state.api.sweeps(f"{user}/{project_name}")
                project_data["total_sweeps"] = len(sweeps)
            except:
                project_data["total_sweeps"] = "N/A"
            
            # Extract all unique tags from runs
            all_tags = set()
            for run in runs:
                if hasattr(run, 'tags') and run.tags:
                    all_tags.update(run.tags)
            
            project_data["tags"] = list(all_tags)
            
            # Preview the summary
            st.markdown("### Project Summary")
            
            for key, value in project_data.items():
                if key != "tags":
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
            
            if project_data["tags"]:
                st.markdown(f"**Tags:** {', '.join(project_data['tags'])}")
            else:
                st.markdown("**Tags:** None")
            
            # Create a more detailed summary of runs
            st.markdown("### Runs Summary")
            
            # Get counts of different configurations and metrics
            config_counts = {}
            metric_counts = {}
            
            for run in runs:
                # Count configs
                if run.config:
                    for key in run.config.keys():
                        if key in config_counts:
                            config_counts[key] += 1
                        else:
                            config_counts[key] = 1
                
                # Count metrics
                if hasattr(run, 'summary') and run.summary:
                    for key in run.summary.keys():
                        if not key.startswith('_'):
                            if key in metric_counts:
                                metric_counts[key] += 1
                            else:
                                metric_counts[key] = 1
            
            # Display config usage
            if config_counts:
                st.markdown("#### Top Configuration Parameters")
                config_df = pd.DataFrame({
                    "Parameter": config_counts.keys(),
                    "Runs Count": config_counts.values()
                }).sort_values("Runs Count", ascending=False).head(10)
                
                st.dataframe(config_df, use_container_width=True)
            
            # Display metric usage
            if metric_counts:
                st.markdown("#### Top Metrics")
                metric_df = pd.DataFrame({
                    "Metric": metric_counts.keys(),
                    "Runs Count": metric_counts.values()
                }).sort_values("Runs Count", ascending=False).head(10)
                
                st.dataframe(metric_df, use_container_width=True)
            
            # Export options
            st.markdown("### Export Format")
            
            # Convert project data to JSON
            project_json = json.dumps(project_data, indent=2)
            
            file_name = st.text_input("File name (without extension)", f"wandb_project_{project_name}_summary_{datetime.now().strftime('%Y%m%d')}")
            
            st.download_button(
                label="Download Project Summary (JSON)",
                data=project_json,
                file_name=f"{file_name}.json",
                mime="application/json"
            )
            
            # Option to export all runs basic data
            if st.checkbox("Include all runs data in export"):
                # Extract basic run data
                all_runs_data = []
                
                for run in runs:
                    run_data = {
                        "id": run.id,
                        "name": run.name,
                        "state": run.state,
                        "created_at": datetime.fromtimestamp(run.created_at).strftime('%Y-%m-%d %H:%M:%S') if hasattr(run, 'created_at') else "N/A",
                        "tags": run.tags if hasattr(run, 'tags') else [],
                        "runtime": round((run.runtime or 0) / 60, 2) if hasattr(run, 'runtime') else "N/A"
                    }
                    all_runs_data.append(run_data)
                
                runs_df = pd.DataFrame(all_runs_data)
                
                # Export as CSV
                csv = runs_df.to_csv(index=False)
                st.download_button(
                    label="Download All Runs Data (CSV)",
                    data=csv,
                    file_name=f"{file_name}_runs.csv",
                    mime="text/csv"
                )
            
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")
