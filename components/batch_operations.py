import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import wandb
from datetime import datetime
import json
import os
import io
import zipfile
import time
import base64
import hashlib
import glob
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor

def batch_operations():
    """
    Perform batch operations across multiple runs
    """
    st.header("🔄 Batch Operations")
    
    st.markdown("""
    This component allows you to select multiple runs and perform operations on them as a batch,
    such as comparing metrics, exporting data, and analyzing patterns across experiments.
    """)
    
    # Check if we have a selected project
    if not st.session_state.selected_project:
        st.warning("Please select a project first.")
        if st.button("Go to Projects"):
            st.session_state.current_page = "projects"
            st.rerun()
        return
    
    project_id = st.session_state.selected_project["id"]
    
    # Get runs for the selected project if not already loaded
    if 'project_runs' not in st.session_state or st.button("Refresh Runs"):
        with st.spinner("Loading runs..."):
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
                
                st.session_state.project_runs = runs
            except Exception as e:
                st.error(f"Error fetching runs: {str(e)}")
                return
    
    if not st.session_state.project_runs:
        st.warning("No runs found for this project.")
        return
    
    # Display runs selection
    st.subheader("Select Runs")
    
    # Create a dataframe for easy viewing
    runs_data = []
    for run in st.session_state.project_runs:
        run_data = {
            "ID": run["id"],
            "Name": run["name"],
            "State": run["state"],
            "Created": pd.to_datetime(run["created_at"]).strftime('%Y-%m-%d %H:%M:%S') if "created_at" in run else ""
        }
        
        # Add key metrics if available
        if "summary" in run:
            for key, value in run["summary"].items():
                if isinstance(value, (int, float)) and not key.startswith('_'):
                    # Only add common metrics to avoid too many columns
                    if key in ["loss", "accuracy", "val_loss", "val_accuracy", "precision", "recall", "f1"]:
                        run_data[key] = value
        
        runs_data.append(run_data)
    
    # Show the runs in a dataframe
    runs_df = pd.DataFrame(runs_data)
    st.dataframe(runs_df, use_container_width=True)
    
    # Create multiselect for run selection
    run_options = [f"{run['name']} ({run['id']})" for run in st.session_state.project_runs]
    selected_run_options = st.multiselect(
        "Select runs for batch operations:",
        options=run_options,
        help="Select two or more runs to perform batch operations."
    )
    
    # Extract run IDs from the selected options
    selected_run_ids = [option.split("(")[-1].split(")")[0] for option in selected_run_options]
    
    # Only proceed if at least two runs are selected
    if len(selected_run_ids) >= 2:
        st.success(f"Selected {len(selected_run_ids)} runs for batch operations.")
        
        # Create tabs for different batch operations
        compare_tab, export_tab, analyze_tab, quantum_tab, bulk_actions_tab = st.tabs([
            "Compare Metrics", "Export Data", "Statistical Analysis", "Quantum Analysis", "Bulk Actions"
        ])
        
        # Get the selected runs data
        selected_runs = []
        for run_id in selected_run_ids:
            for run in st.session_state.project_runs:
                if run["id"] == run_id:
                    selected_runs.append(run)
        
        # Compare Metrics Tab
        with compare_tab:
            st.subheader("Compare Metrics Across Runs")
            
            # Collect all available metrics from the selected runs
            all_metrics = set()
            for run in selected_runs:
                if "summary" in run:
                    for key, value in run["summary"].items():
                        if isinstance(value, (int, float)) and not key.startswith('_'):
                            all_metrics.add(key)
            
            if all_metrics:
                # Let user select metrics to compare
                selected_metrics = st.multiselect(
                    "Select metrics to compare:",
                    options=sorted(list(all_metrics)),
                    default=list(all_metrics)[:min(3, len(all_metrics))]
                )
                
                if selected_metrics:
                    # Create a dataframe for comparison
                    comparison_data = []
                    for run in selected_runs:
                        run_data = {"Run": run["name"], "ID": run["id"]}
                        
                        for metric in selected_metrics:
                            run_data[metric] = run.get("summary", {}).get(metric, None)
                        
                        comparison_data.append(run_data)
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Show the comparison table
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Plot comparisons
                    st.subheader("Metric Comparison Charts")
                    
                    chart_type = st.radio(
                        "Select chart type:",
                        ["Bar Chart", "Radar Chart", "Heatmap"],
                        horizontal=True
                    )
                    
                    if chart_type == "Bar Chart":
                        # Create a bar chart for each selected metric
                        for metric in selected_metrics:
                            metric_df = comparison_df[["Run", metric]].dropna()
                            
                            if not metric_df.empty:
                                fig = px.bar(
                                    metric_df,
                                    x="Run",
                                    y=metric,
                                    title=f"{metric} Comparison",
                                    labels={"Run": "Run Name", metric: metric},
                                    text_auto='.3f',
                                    color="Run"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(f"No data available for metric: {metric}")
                    
                    elif chart_type == "Radar Chart":
                        # Create a radar chart for all metrics
                        if len(selected_metrics) >= 3:
                            # Prepare data for radar chart
                            fig = go.Figure()
                            
                            for i, run in enumerate(comparison_data):
                                values = [run.get(metric, 0) for metric in selected_metrics]
                                # Add the first value at the end to close the polygon
                                values.append(values[0])
                                
                                # Handle NaN values
                                values = [0 if pd.isna(v) else v for v in values]
                                
                                # Add trace for this run
                                fig.add_trace(go.Scatterpolar(
                                    r=values,
                                    theta=selected_metrics + [selected_metrics[0]],
                                    fill='toself',
                                    name=run["Run"]
                                ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                    )
                                ),
                                title="Multi-metric Comparison (Radar Chart)"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Radar chart requires at least 3 metrics. Please select more metrics.")
                    
                    elif chart_type == "Heatmap":
                        # Create a heatmap for all metrics
                        # Reshape data for heatmap
                        heatmap_df = comparison_df.set_index("Run")
                        heatmap_df = heatmap_df.drop(columns=["ID"])
                        
                        # Normalize the data to make metrics comparable
                        normalized_df = pd.DataFrame()
                        for col in heatmap_df.columns:
                            col_min = heatmap_df[col].min()
                            col_max = heatmap_df[col].max()
                            if col_max > col_min:
                                normalized_df[col] = (heatmap_df[col] - col_min) / (col_max - col_min)
                            else:
                                normalized_df[col] = heatmap_df[col] / heatmap_df[col]
                        
                        # Create heatmap
                        fig = px.imshow(
                            normalized_df,
                            labels=dict(x="Metric", y="Run", color="Normalized Value"),
                            x=normalized_df.columns,
                            y=normalized_df.index,
                            color_continuous_scale="Viridis",
                            title="Metrics Heatmap (Normalized Values)"
                        )
                        
                        # Add actual values as annotations
                        annotations = []
                        for i, run in enumerate(normalized_df.index):
                            for j, metric in enumerate(normalized_df.columns):
                                original_value = heatmap_df.loc[run, metric]
                                annotations.append(dict(
                                    x=j, y=i,
                                    text=f"{original_value:.3f}",
                                    showarrow=False,
                                    font=dict(color="white" if normalized_df.iloc[i, j] > 0.5 else "black")
                                ))
                        
                        fig.update_layout(annotations=annotations)
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("Please select at least one metric to compare.")
            else:
                st.info("No comparable metrics found across the selected runs.")
                
            # Config parameter comparison
            st.subheader("Compare Configuration Parameters")
            
            # Collect all config parameters
            all_params = set()
            for run in selected_runs:
                if "config" in run:
                    for key in run["config"].keys():
                        if not key.startswith('_'):
                            all_params.add(key)
            
            if all_params:
                # Let user select parameters to compare
                selected_params = st.multiselect(
                    "Select configuration parameters to compare:",
                    options=sorted(list(all_params)),
                    default=list(all_params)[:min(5, len(all_params))]
                )
                
                if selected_params:
                    # Create a dataframe for comparison
                    param_data = []
                    for run in selected_runs:
                        run_data = {"Run": run["name"], "ID": run["id"]}
                        
                        for param in selected_params:
                            run_data[param] = run.get("config", {}).get(param, None)
                        
                        param_data.append(run_data)
                    
                    param_df = pd.DataFrame(param_data)
                    
                    # Show the comparison table
                    st.dataframe(param_df, use_container_width=True)
                
                else:
                    st.info("Please select at least one parameter to compare.")
            else:
                st.info("No configuration parameters found across the selected runs.")
        
        # Export Data Tab
        with export_tab:
            st.subheader("Export Batch Data")
            
            export_type = st.radio(
                "Select export type:",
                ["Summary Metrics", "Configuration Parameters", "Complete Run Data", "Combined Report"],
                horizontal=True
            )
            
            # Prepare export options based on type
            if export_type == "Summary Metrics":
                # Collect all metrics
                all_metrics = set()
                for run in selected_runs:
                    if "summary" in run:
                        for key in run["summary"].keys():
                            if not key.startswith('_'):
                                all_metrics.add(key)
                
                if all_metrics:
                    metrics_to_export = st.multiselect(
                        "Select metrics to export:",
                        options=sorted(list(all_metrics)),
                        default=list(all_metrics)
                    )
                    
                    if metrics_to_export and st.button("Generate Metrics Export"):
                        # Create export dataframe
                        export_data = []
                        for run in selected_runs:
                            run_data = {
                                "run_id": run["id"],
                                "run_name": run["name"],
                                "state": run["state"],
                                "created_at": run.get("created_at", "")
                            }
                            
                            for metric in metrics_to_export:
                                run_data[metric] = run.get("summary", {}).get(metric, None)
                            
                            export_data.append(run_data)
                        
                        export_df = pd.DataFrame(export_data)
                        
                        # Convert to CSV
                        csv = export_df.to_csv(index=False)
                        
                        # Offer for download
                        st.download_button(
                            label="Download Metrics CSV",
                            data=csv,
                            file_name=f"batch_metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("No metrics available across the selected runs.")
            
            elif export_type == "Configuration Parameters":
                # Collect all config params
                all_params = set()
                for run in selected_runs:
                    if "config" in run:
                        for key in run["config"].keys():
                            if not key.startswith('_'):
                                all_params.add(key)
                
                if all_params:
                    params_to_export = st.multiselect(
                        "Select configuration parameters to export:",
                        options=sorted(list(all_params)),
                        default=list(all_params)
                    )
                    
                    if params_to_export and st.button("Generate Config Export"):
                        # Create export dataframe
                        export_data = []
                        for run in selected_runs:
                            run_data = {
                                "run_id": run["id"],
                                "run_name": run["name"],
                                "state": run["state"],
                                "created_at": run.get("created_at", "")
                            }
                            
                            for param in params_to_export:
                                run_data[param] = run.get("config", {}).get(param, None)
                            
                            export_data.append(run_data)
                        
                        export_df = pd.DataFrame(export_data)
                        
                        # Convert to CSV
                        csv = export_df.to_csv(index=False)
                        
                        # Offer for download
                        st.download_button(
                            label="Download Config CSV",
                            data=csv,
                            file_name=f"batch_config_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("No configuration parameters available across the selected runs.")
            
            elif export_type == "Complete Run Data":
                if st.button("Generate Complete Export"):
                    # Create a ZIP file with JSON data for each run
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                        for run in selected_runs:
                            # Convert run data to JSON
                            run_json = json.dumps(run, default=str, indent=2)
                            
                            # Add to zip file
                            zip_file.writestr(f"run_{run['id']}.json", run_json)
                    
                    # Reset buffer position
                    zip_buffer.seek(0)
                    
                    # Offer for download
                    st.download_button(
                        label="Download Complete Run Data (ZIP)",
                        data=zip_buffer,
                        file_name=f"batch_runs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
            
            elif export_type == "Combined Report":
                report_format = st.selectbox(
                    "Report format:",
                    ["CSV", "Excel"]
                )
                
                if st.button("Generate Combined Report"):
                    # Collect metrics and params
                    all_metrics = set()
                    all_params = set()
                    
                    for run in selected_runs:
                        if "summary" in run:
                            for key in run["summary"].keys():
                                if not key.startswith('_') and isinstance(run["summary"][key], (int, float)):
                                    all_metrics.add(key)
                        
                        if "config" in run:
                            for key in run["config"].keys():
                                if not key.startswith('_'):
                                    all_params.add(key)
                    
                    # Create dataframes
                    run_info_data = []
                    metrics_data = []
                    config_data = []
                    
                    for run in selected_runs:
                        # Basic run info
                        run_info = {
                            "run_id": run["id"],
                            "run_name": run["name"],
                            "state": run["state"],
                            "created_at": pd.to_datetime(run.get("created_at", "")).strftime('%Y-%m-%d %H:%M:%S')
                        }
                        run_info_data.append(run_info)
                        
                        # Metrics
                        metrics = {"run_id": run["id"], "run_name": run["name"]}
                        for metric in sorted(all_metrics):
                            metrics[metric] = run.get("summary", {}).get(metric, None)
                        metrics_data.append(metrics)
                        
                        # Config
                        config = {"run_id": run["id"], "run_name": run["name"]}
                        for param in sorted(all_params):
                            config[param] = run.get("config", {}).get(param, None)
                        config_data.append(config)
                    
                    run_info_df = pd.DataFrame(run_info_data)
                    metrics_df = pd.DataFrame(metrics_data)
                    config_df = pd.DataFrame(config_data)
                    
                    if report_format == "CSV":
                        # Create CSV files
                        info_csv = run_info_df.to_csv(index=False)
                        metrics_csv = metrics_df.to_csv(index=False)
                        config_csv = config_df.to_csv(index=False)
                        
                        # Create ZIP with all CSVs
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                            zip_file.writestr("01_run_info.csv", info_csv)
                            zip_file.writestr("02_metrics.csv", metrics_csv)
                            zip_file.writestr("03_config.csv", config_csv)
                        
                        # Reset buffer position
                        zip_buffer.seek(0)
                        
                        # Offer for download
                        st.download_button(
                            label="Download Combined Report (ZIP of CSVs)",
                            data=zip_buffer,
                            file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
                    
                    elif report_format == "Excel":
                        # Create Excel file with multiple sheets
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            run_info_df.to_excel(writer, sheet_name='Run Info', index=False)
                            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                            config_df.to_excel(writer, sheet_name='Configuration', index=False)
                            
                            # Add some formatting
                            workbook = writer.book
                            
                            # Format for Run Info sheet
                            info_sheet = writer.sheets['Run Info']
                            header_format = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
                            
                            for col_num, value in enumerate(run_info_df.columns.values):
                                info_sheet.write(0, col_num, value, header_format)
                            
                            # Format for Metrics sheet
                            metrics_sheet = writer.sheets['Metrics']
                            for col_num, value in enumerate(metrics_df.columns.values):
                                metrics_sheet.write(0, col_num, value, header_format)
                            
                            # Format for Configuration sheet
                            config_sheet = writer.sheets['Configuration']
                            for col_num, value in enumerate(config_df.columns.values):
                                config_sheet.write(0, col_num, value, header_format)
                        
                        excel_data = excel_buffer.getvalue()
                        
                        # Offer for download
                        st.download_button(
                            label="Download Combined Report (Excel)",
                            data=excel_data,
                            file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        
        # Statistical Analysis Tab
        with analyze_tab:
            st.subheader("Statistical Analysis")
            
            # Collect all numeric metrics
            all_metrics = set()
            for run in selected_runs:
                if "summary" in run:
                    for key, value in run["summary"].items():
                        if isinstance(value, (int, float)) and not key.startswith('_'):
                            all_metrics.add(key)
            
            if all_metrics:
                # Select a metric for detailed analysis
                selected_metric = st.selectbox(
                    "Select a metric for statistical analysis:",
                    options=sorted(list(all_metrics))
                )
                
                if selected_metric:
                    # Extract values for the selected metric
                    metric_values = []
                    run_names = []
                    
                    for run in selected_runs:
                        value = run.get("summary", {}).get(selected_metric)
                        if value is not None and not pd.isna(value):
                            metric_values.append(value)
                            run_names.append(run["name"])
                    
                    if metric_values:
                        # Basic statistics
                        metric_series = pd.Series(metric_values)
                        
                        st.markdown("### Basic Statistics")
                        stats_cols = st.columns(4)
                        
                        with stats_cols[0]:
                            st.metric("Mean", f"{metric_series.mean():.4f}")
                        
                        with stats_cols[1]:
                            st.metric("Median", f"{metric_series.median():.4f}")
                        
                        with stats_cols[2]:
                            st.metric("Min", f"{metric_series.min():.4f}")
                        
                        with stats_cols[3]:
                            st.metric("Max", f"{metric_series.max():.4f}")
                        
                        # More detailed statistics
                        stats_df = pd.DataFrame({
                            "Statistic": ["Count", "Mean", "Median", "Std Dev", "Min", "25%", "50%", "75%", "Max", "Range"],
                            "Value": [
                                len(metric_values),
                                metric_series.mean(),
                                metric_series.median(),
                                metric_series.std(),
                                metric_series.min(),
                                metric_series.quantile(0.25),
                                metric_series.quantile(0.5),
                                metric_series.quantile(0.75),
                                metric_series.max(),
                                metric_series.max() - metric_series.min()
                            ]
                        })
                        
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                        # Distribution plots
                        st.markdown("### Distribution Analysis")
                        
                        plot_type = st.radio(
                            "Select plot type:",
                            ["Histogram", "Box Plot", "Violin Plot"],
                            horizontal=True
                        )
                        
                        if plot_type == "Histogram":
                            fig = px.histogram(
                                x=metric_values,
                                labels={"x": selected_metric},
                                title=f"Distribution of {selected_metric}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif plot_type == "Box Plot":
                            # Prepare data for box plot
                            box_df = pd.DataFrame({
                                "Run": run_names,
                                "Value": metric_values
                            })
                            
                            fig = px.box(
                                box_df,
                                y="Value",
                                title=f"Box Plot of {selected_metric}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif plot_type == "Violin Plot":
                            # Prepare data for violin plot
                            violin_df = pd.DataFrame({
                                "Metric": [selected_metric] * len(metric_values),
                                "Value": metric_values
                            })
                            
                            fig = px.violin(
                                violin_df,
                                y="Value",
                                box=True,
                                points="all",
                                title=f"Violin Plot of {selected_metric}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Correlation analysis if we have configuration parameters
                        st.markdown("### Correlation Analysis")
                        st.markdown("Analyze correlation between configuration parameters and metric value.")
                        
                        # Collect numeric config parameters
                        numeric_params = {}
                        for run in selected_runs:
                            if "config" in run and selected_metric in run.get("summary", {}):
                                for param, value in run["config"].items():
                                    if isinstance(value, (int, float)) and not param.startswith('_'):
                                        if param not in numeric_params:
                                            numeric_params[param] = []
                                        
                                        # Pair config value with metric value
                                        metric_value = run["summary"][selected_metric]
                                        numeric_params[param].append((value, metric_value))
                        
                        if numeric_params:
                            # Let user select a parameter to correlate
                            param_options = [p for p in numeric_params.keys() if len(numeric_params[p]) >= 3]
                            
                            if param_options:
                                selected_param = st.selectbox(
                                    "Select a configuration parameter to correlate with metric:",
                                    options=sorted(param_options)
                                )
                                
                                if selected_param:
                                    # Get the paired values
                                    paired_values = numeric_params[selected_param]
                                    param_values = [p[0] for p in paired_values]
                                    paired_metric_values = [p[1] for p in paired_values]
                                    
                                    # Create scatter plot
                                    fig = px.scatter(
                                        x=param_values,
                                        y=paired_metric_values,
                                        labels={"x": selected_param, "y": selected_metric},
                                        title=f"Correlation between {selected_param} and {selected_metric}",
                                        trendline="ols"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Calculate correlation
                                    if len(paired_values) >= 3:
                                        corr = np.corrcoef(param_values, paired_metric_values)[0, 1]
                                        st.metric("Correlation Coefficient", f"{corr:.4f}")
                                        
                                        if abs(corr) < 0.3:
                                            st.info("Weak correlation detected.")
                                        elif abs(corr) < 0.7:
                                            st.info("Moderate correlation detected.")
                                        else:
                                            st.success("Strong correlation detected!")
                            else:
                                st.info("Not enough data points to perform correlation analysis.")
                        else:
                            st.info("No numeric configuration parameters found for correlation analysis.")
                    else:
                        st.warning("No valid values found for the selected metric.")
                
            else:
                st.info("No metrics available for statistical analysis.")
        
        # Quantum Analysis Tab
        with quantum_tab:
            st.subheader("Quantum Analysis")
            
            st.markdown("""
            This tab provides specialized analysis tools for quantum computing experiments,
            including parameter space visualization and quantum circuit comparison.
            """)
            
            # Check if selected runs have quantum-related parameters or metrics
            has_quantum_params = False
            quantum_param_prefixes = ["circuit", "qubit", "gate", "entangle", "rotate", "phase", "noise", "ansatz"]
            
            for run in selected_runs:
                for param in run.get("config", {}):
                    if any(prefix in param.lower() for prefix in quantum_param_prefixes):
                        has_quantum_params = True
                        break
            
            if has_quantum_params:
                st.success("Quantum parameters detected in the selected runs.")
                
                # Quantum parameter space visualization
                st.markdown("### Quantum Parameter Space")
                
                # Collect quantum-specific parameters
                quantum_params = {}
                for run in selected_runs:
                    if "config" in run:
                        for param, value in run["config"].items():
                            if any(prefix in param.lower() for prefix in quantum_param_prefixes) and isinstance(value, (int, float)):
                                if param not in quantum_params:
                                    quantum_params[param] = []
                                
                                # Store the parameter value and run ID
                                quantum_params[param].append((value, run["id"]))
                
                if quantum_params:
                    # Get parameters with at least 2 different values
                    valid_params = {p: v for p, v in quantum_params.items() if len(set([x[0] for x in v])) >= 2}
                    
                    if valid_params:
                        # Select parameters for visualization
                        x_param = st.selectbox(
                            "Select X-axis parameter:",
                            options=sorted(valid_params.keys())
                        )
                        
                        y_options = [p for p in valid_params.keys() if p != x_param]
                        if y_options:
                            y_param = st.selectbox(
                                "Select Y-axis parameter:",
                                options=sorted(y_options)
                            )
                            
                            # Select a metric for color coding
                            all_metrics = set()
                            for run in selected_runs:
                                if "summary" in run:
                                    for key, value in run["summary"].items():
                                        if isinstance(value, (int, float)) and not key.startswith('_'):
                                            all_metrics.add(key)
                            
                            if all_metrics:
                                color_metric = st.selectbox(
                                    "Select metric for color coding:",
                                    options=sorted(list(all_metrics))
                                )
                                
                                # Create visualization dataset
                                plot_data = []
                                for run in selected_runs:
                                    if x_param in run.get("config", {}) and y_param in run.get("config", {}) and color_metric in run.get("summary", {}):
                                        plot_data.append({
                                            "run_id": run["id"],
                                            "run_name": run["name"],
                                            "x_value": run["config"][x_param],
                                            "y_value": run["config"][y_param],
                                            "metric": run["summary"][color_metric]
                                        })
                                
                                if plot_data:
                                    plot_df = pd.DataFrame(plot_data)
                                    
                                    # Create scatter plot
                                    fig = px.scatter(
                                        plot_df,
                                        x="x_value",
                                        y="y_value",
                                        color="metric",
                                        hover_name="run_name",
                                        labels={
                                            "x_value": x_param,
                                            "y_value": y_param,
                                            "metric": color_metric
                                        },
                                        title=f"Quantum Parameter Space: {x_param} vs {y_param} (colored by {color_metric})"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Not enough data for selected parameters and metric.")
                            else:
                                st.info("No metrics available for color coding.")
                        else:
                            st.info("Need at least two quantum parameters with different values.")
                    else:
                        st.info("Not enough variation in quantum parameters for meaningful visualization.")
                else:
                    st.info("No numeric quantum parameters found.")
                
                # Circuit complexity comparison
                st.markdown("### Circuit Complexity Analysis")
                
                # Check for circuit depth/width metrics
                circuit_metrics = ["circuit_depth", "circuit_width", "n_qubits", "gate_count", "entangler_count"]
                found_circuit_metrics = []
                
                for metric in circuit_metrics:
                    for run in selected_runs:
                        if metric in run.get("summary", {}):
                            found_circuit_metrics.append(metric)
                            break
                
                if found_circuit_metrics:
                    # Create a circuit complexity comparison
                    complexity_data = []
                    for run in selected_runs:
                        run_data = {"run_id": run["id"], "run_name": run["name"]}
                        
                        for metric in found_circuit_metrics:
                            run_data[metric] = run.get("summary", {}).get(metric, None)
                        
                        # Also look for a performance metric
                        performance_metrics = ["accuracy", "fidelity", "success_rate"]
                        for metric in performance_metrics:
                            if metric in run.get("summary", {}):
                                run_data["performance"] = run["summary"][metric]
                                run_data["performance_metric"] = metric
                                break
                        
                        complexity_data.append(run_data)
                    
                    if complexity_data:
                        # Convert to dataframe
                        complexity_df = pd.DataFrame(complexity_data)
                        
                        # Show table
                        st.dataframe(complexity_df, use_container_width=True)
                        
                        # Create visualization if we have enough data
                        if len(found_circuit_metrics) >= 2 and "performance" in complexity_df.columns:
                            # Select metrics for x and y
                            x_metric = st.selectbox(
                                "Select X-axis circuit metric:",
                                options=found_circuit_metrics
                            )
                            
                            y_options = [m for m in found_circuit_metrics if m != x_metric]
                            if y_options:
                                y_metric = st.selectbox(
                                    "Select Y-axis circuit metric:",
                                    options=y_options
                                )
                                
                                # Create bubble chart
                                fig = px.scatter(
                                    complexity_df,
                                    x=x_metric,
                                    y=y_metric,
                                    size="performance",
                                    color="performance",
                                    hover_name="run_name",
                                    text="run_name",
                                    title=f"Circuit Complexity vs Performance: {x_metric} vs {y_metric}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No circuit complexity metrics found in the selected runs.")
            else:
                st.info("No quantum-specific parameters detected in the selected runs.")
        
        # Bulk Actions Tab
        with bulk_actions_tab:
            st.subheader("Bulk Actions")
            
            st.markdown("""
            This tab provides tools for performing bulk operations across multiple runs simultaneously,
            including batch tagging, status management, metadata operations, and more.
            """)
            
            # Create sub-tabs for different bulk operations
            batch_tagging_tab, status_management_tab, metadata_tab, artifact_tab, deletion_tab = st.tabs([
                "Batch Tagging", "Status Management", "Metadata Operations", "Artifact Management", "Deletion & Archiving"
            ])
            
            # Batch Tagging Tab
            with batch_tagging_tab:
                st.markdown("### Tag Management System")
                
                # Display current tags for selected runs
                all_tags = set()
                run_tags = {}
                
                for run in selected_runs:
                    tags = run.get("tags", [])
                    run_tags[run["id"]] = tags
                    all_tags.update(tags)
                
                # Create tabs for different tag operations
                tag_overview_tab, add_tags_tab, remove_tags_tab, tag_templates_tab = st.tabs([
                    "Tag Overview", "Add Tags", "Remove Tags", "Tag Templates"
                ])
                
                with tag_overview_tab:
                    st.markdown("#### Current Tag Distribution")
                
                    # Show current tags table
                    if all_tags:
                        # Create a dataframe to show which runs have which tags
                        tag_data = []
                        for run in selected_runs:
                            run_data = {"Run": run["name"], "ID": run["id"]}
                            
                            for tag in sorted(all_tags):
                                run_data[tag] = "✓" if tag in run_tags.get(run["id"], []) else ""
                            
                            tag_data.append(run_data)
                        
                        tag_df = pd.DataFrame(tag_data)
                        st.dataframe(tag_df, use_container_width=True)
                        
                        # Visualize tag distribution
                        st.markdown("#### Tag Distribution Analysis")
                        
                        # Count tags across runs
                        tag_counts = {tag: sum(1 for run_id in run_tags if tag in run_tags[run_id]) for tag in all_tags}
                        
                        # Create bar chart for tag distribution
                        fig = px.bar(
                            x=list(tag_counts.keys()), 
                            y=list(tag_counts.values()),
                            labels={"x": "Tag", "y": "Count"},
                            title="Tag Distribution Across Selected Runs",
                            color=list(tag_counts.keys())
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No tags found on the selected runs.")
                
                # Add new tags tab
                with add_tags_tab:
                    st.markdown("#### Add New Tags to Selected Runs")
                    
                    # Standard tag entry
                    new_tags = st.text_input("Enter new tags (comma-separated):")
                    
                    # Suggested category tags as checkboxes
                    st.markdown("#### Suggested Category Tags")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        add_status_tag = st.checkbox("📊 Status", key="status_tag")
                        add_verified_tag = st.checkbox("✅ Verified", key="verified_tag")
                        add_production_tag = st.checkbox("🏭 Production", key="production_tag")
                    
                    with col2:
                        add_experiment_tag = st.checkbox("🧪 Experiment", key="experiment_tag")
                        add_baseline_tag = st.checkbox("📏 Baseline", key="baseline_tag")
                        add_debug_tag = st.checkbox("🐞 Debug", key="debug_tag")
                    
                    with col3:
                        add_review_tag = st.checkbox("👁️ Review", key="review_tag")
                        add_quantum_tag = st.checkbox("⚛️ Quantum", key="quantum_tag")
                        add_ml_tag = st.checkbox("🧠 Machine Learning", key="ml_tag")
                    
                    # Custom tag templates
                    st.markdown("#### Custom Tag Prefix")
                    tag_prefix = st.text_input("Enter a prefix for all tags (optional):")
                    
                    if st.button("Add Tags to Selected Runs"):
                        # Collect all selected tags
                        tag_list = [tag.strip() for tag in new_tags.split(",") if tag.strip()]
                        
                        # Add suggested tags if selected
                        if add_status_tag:
                            tag_list.append("status")
                        if add_verified_tag:
                            tag_list.append("verified")
                        if add_production_tag:
                            tag_list.append("production")
                        if add_experiment_tag:
                            tag_list.append("experiment")
                        if add_baseline_tag:
                            tag_list.append("baseline")
                        if add_debug_tag:
                            tag_list.append("debug")
                        if add_review_tag:
                            tag_list.append("review")
                        if add_quantum_tag:
                            tag_list.append("quantum")
                        if add_ml_tag:
                            tag_list.append("ml")
                        
                        # Apply prefix if provided
                        if tag_prefix:
                            tag_list = [f"{tag_prefix}_{tag}" if not tag.startswith(tag_prefix) else tag for tag in tag_list]
                        
                        if tag_list:
                            try:
                                # Use the W&B API to add tags to each run
                                api = wandb.Api()
                                
                                success_count = 0
                                for run_id in selected_run_ids:
                                    try:
                                        run = api.run(f"{st.session_state.wandb_entity}/{st.session_state.selected_project}/{run_id}")
                                        run.tags = list(set(run.tags + tag_list))
                                        run.update()
                                        success_count += 1
                                    except Exception as e:
                                        st.error(f"Error adding tags to run {run_id}: {str(e)}")
                                
                                if success_count > 0:
                                    st.success(f"Added tags to {success_count} runs. Refresh the page to see updates.")
                                    # Update session state runs data
                                    st.session_state.refresh_required = True
                            except Exception as e:
                                st.error(f"Error accessing W&B API: {str(e)}")
                        else:
                            st.warning("No valid tags provided. Please enter tags or select suggested categories.")
                    else:
                        pass  # Don't show any warning until the button is clicked
                
                # Remove tags tab
                with remove_tags_tab:
                    st.markdown("#### Remove Tags from Selected Runs")
                    
                    if all_tags:
                        st.markdown("Select tags to remove from the runs:")
                        
                        # Create a multi-column layout for tag checkboxes
                        num_cols = 3
                        cols = st.columns(num_cols)
                        
                        # Organize tags into columns
                        sorted_tags = sorted(list(all_tags))
                        tags_per_col = (len(sorted_tags) + num_cols - 1) // num_cols
                        
                        # Dictionary to store checkbox states
                        tags_to_remove = []
                        
                        # Distribute tags across columns
                        for i, tag in enumerate(sorted_tags):
                            col_idx = i // tags_per_col
                            with cols[min(col_idx, num_cols-1)]:
                                if st.checkbox(tag, key=f"tag_{tag}_remove"):
                                    tags_to_remove.append(tag)
                        
                        # Add a "Select All" and "Deselect All" option
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Select All Tags"):
                                # We can't directly modify checkbox states, but this informs the user
                                st.info("Please manually check all boxes, or use 'Remove All Tags' button below.")
                        
                        with col2:
                            remove_all_tags = st.button("Remove ALL Tags")
                        
                        # Button to remove selected tags
                        if st.button("Remove Selected Tags"):
                            if tags_to_remove:
                                try:
                                    # Use the W&B API to remove tags from each run
                                    api = wandb.Api()
                                    
                                    success_count = 0
                                    for run_id in selected_run_ids:
                                        try:
                                            run = api.run(f"{st.session_state.wandb_entity}/{st.session_state.selected_project}/{run_id}")
                                            run.tags = [tag for tag in run.tags if tag not in tags_to_remove]
                                            run.update()
                                            success_count += 1
                                        except Exception as e:
                                            st.error(f"Error removing tags from run {run_id}: {str(e)}")
                                    
                                    if success_count > 0:
                                        st.success(f"Removed tags from {success_count} runs. Refresh the page to see updates.")
                                        # Update session state runs data
                                        st.session_state.refresh_required = True
                                except Exception as e:
                                    st.error(f"Error accessing W&B API: {str(e)}")
                            else:
                                st.warning("Please select at least one tag to remove.")
                        
                        # Handle the "Remove All Tags" action
                        if remove_all_tags:
                            try:
                                # Use the W&B API to remove all tags from each run
                                api = wandb.Api()
                                
                                success_count = 0
                                for run_id in selected_run_ids:
                                    try:
                                        run = api.run(f"{st.session_state.wandb_entity}/{st.session_state.selected_project}/{run_id}")
                                        run.tags = []
                                        run.update()
                                        success_count += 1
                                    except Exception as e:
                                        st.error(f"Error removing all tags from run {run_id}: {str(e)}")
                                
                                if success_count > 0:
                                    st.success(f"Removed all tags from {success_count} runs. Refresh the page to see updates.")
                                    # Update session state runs data
                                    st.session_state.refresh_required = True
                            except Exception as e:
                                st.error(f"Error accessing W&B API: {str(e)}")
                    else:
                        st.info("No tags available to remove.")
                
                # Tag templates tab
                with tag_templates_tab:
                    st.markdown("#### Tag Templates & Bulk Operations")
                    
                    st.markdown("""
                    Apply predefined tag sets to selected runs. These templates help maintain
                    consistent tagging across your experiments.
                    """)
                    
                    # Define some common tag templates
                    tag_templates = {
                        "Production Ready": ["production", "verified", "stable"],
                        "Development": ["dev", "experimental", "in-progress"],
                        "Quantum ML": ["quantum", "ml", "hybrid"],
                        "Review Required": ["review", "pending", "needs-attention"],
                        "Baseline Model": ["baseline", "reference", "standard"],
                        "Deprecated": ["deprecated", "archived", "obsolete"]
                    }
                    
                    # Let user select a template
                    selected_template = st.selectbox(
                        "Select a tag template:",
                        options=list(tag_templates.keys())
                    )
                    
                    # Display tags in the selected template
                    st.markdown(f"**Tags in template:** {', '.join(tag_templates[selected_template])}")
                    
                    # Apply template button
                    if st.button(f"Apply '{selected_template}' Template"):
                        try:
                            # Use the W&B API to add template tags to each run
                            api = wandb.Api()
                            
                            success_count = 0
                            for run_id in selected_run_ids:
                                try:
                                    run = api.run(f"{st.session_state.wandb_entity}/{st.session_state.selected_project}/{run_id}")
                                    run.tags = list(set(run.tags + tag_templates[selected_template]))
                                    run.update()
                                    success_count += 1
                                except Exception as e:
                                    st.error(f"Error applying tag template to run {run_id}: {str(e)}")
                            
                            if success_count > 0:
                                st.success(f"Applied template to {success_count} runs. Refresh the page to see updates.")
                                # Update session state runs data
                                st.session_state.refresh_required = True
                        except Exception as e:
                            st.error(f"Error accessing W&B API: {str(e)}")
                    
                    # Create custom template section
                    st.markdown("#### Create Custom Template")
                    custom_template_name = st.text_input("Template Name:")
                    custom_template_tags = st.text_input("Template Tags (comma-separated):")
                    
                    if st.button("Save Custom Template"):
                        if custom_template_name and custom_template_tags:
                            tag_list = [tag.strip() for tag in custom_template_tags.split(",") if tag.strip()]
                            if tag_list:
                                # This is a UI demo - in a real app, you'd persist templates to a database
                                st.success(f"Template '{custom_template_name}' created with tags: {', '.join(tag_list)}")
                                st.info("Note: Custom templates are not permanently saved in this demo version.")
                            else:
                                st.warning("Please enter valid tags for your template.")
            
            # Status Management Tab
            with status_management_tab:
                st.markdown("### Manage Run Status")
                
                # Display current status
                st.markdown("#### Current Status")
                status_data = []
                for run in selected_runs:
                    status_data.append({
                        "Run": run["name"],
                        "ID": run["id"],
                        "State": run.get("state", "unknown"),
                        "Created": pd.to_datetime(run["created_at"]).strftime('%Y-%m-%d %H:%M:%S') if "created_at" in run else "",
                        "Created At": run.get("created_at", "")
                    })
                
                status_df = pd.DataFrame(status_data)
                st.dataframe(status_df, use_container_width=True)
                
                # Status change operations
                st.markdown("#### Change Status")
                
                status_action = st.selectbox(
                    "Select action:",
                    ["Archive Runs", "Unarchive Runs", "Mark as Failed", "Mark as Finished"]
                )
                
                if st.button(f"Apply {status_action}"):
                    try:
                        # Use the W&B API to change run status
                        api = wandb.Api()
                        
                        success_count = 0
                        for run_id in selected_run_ids:
                            try:
                                run = api.run(f"{st.session_state.wandb_entity}/{st.session_state.selected_project}/{run_id}")
                                
                                if status_action == "Archive Runs":
                                    run.archive()
                                elif status_action == "Unarchive Runs":
                                    run.unarchive()
                                elif status_action == "Mark as Failed":
                                    run.state = "failed"
                                    run.update()
                                elif status_action == "Mark as Finished":
                                    run.state = "finished"
                                    run.update()
                                
                                success_count += 1
                            except Exception as e:
                                st.error(f"Error changing status for run {run_id}: {str(e)}")
                        
                        if success_count > 0:
                            st.success(f"Changed status for {success_count} runs. Refresh the page to see updates.")
                            # Update session state runs data
                            st.session_state.refresh_required = True
                    except Exception as e:
                        st.error(f"Error accessing W&B API: {str(e)}")
            
            # Metadata Operations Tab
            with metadata_tab:
                st.markdown("### Metadata Operations")
                
                # Add or update notes
                st.markdown("#### Run Notes")
                batch_notes = st.text_area("Enter notes to add to all selected runs:", height=100)
                
                # Add a button to apply notes
                if st.button("Apply Notes to Selected Runs"):
                    if batch_notes:
                        try:
                            api = wandb.Api()
                            success_count = 0
                            
                            for run_id in selected_run_ids:
                                try:
                                    run = api.run(f"{project_id}/{run_id}")
                                    
                                    # Get existing notes or initialize empty string
                                    existing_notes = run.notes if hasattr(run, "notes") else ""
                                    
                                    # Append or set new notes
                                    if existing_notes:
                                        # Add a divider if there are existing notes
                                        run.notes = f"{existing_notes}\n\n---\n{batch_notes}"
                                    else:
                                        run.notes = batch_notes
                                    
                                    run.update()
                                    success_count += 1
                                except Exception as e:
                                    st.error(f"Error updating notes for run {run_id}: {str(e)}")
                            
                            if success_count > 0:
                                st.success(f"Updated notes for {success_count} runs. Refresh to see changes.")
                        except Exception as e:
                            st.error(f"Error accessing W&B API: {str(e)}")
                    else:
                        st.warning("Please enter notes to add.")
                
                # Custom metadata fields
                st.markdown("#### Custom Metadata Fields")
                st.markdown("Add or update custom metadata fields across all selected runs.")
                
                col1, col2 = st.columns(2)
                with col1:
                    custom_field_key = st.text_input("Field Key", placeholder="e.g. experiment_name")
                with col2:
                    custom_field_value = st.text_input("Field Value", placeholder="e.g. Quantum Experiment 1")
                
                add_metadata_button = st.button("Add/Update Metadata Field")
                
                if add_metadata_button and custom_field_key and custom_field_value:
                    try:
                        api = wandb.Api()
                        success_count = 0
                        
                        for run_id in selected_run_ids:
                            try:
                                run = api.run(f"{project_id}/{run_id}")
                                if not hasattr(run, "summary"):
                                    run.summary = {}
                                
                                # Set the custom metadata field
                                run.summary[custom_field_key] = custom_field_value
                                run.update()
                                success_count += 1
                            except Exception as e:
                                st.error(f"Error updating metadata for run {run_id}: {str(e)}")
                        
                        if success_count > 0:
                            st.success(f"Updated metadata for {success_count} runs. Refresh to see changes.")
                    except Exception as e:
                        st.error(f"Error accessing W&B API: {str(e)}")
                elif add_metadata_button:
                    st.warning("Please provide both a field key and value.")
                
                append_mode = st.checkbox("Append to existing notes (unchecked will replace notes)")
                
                if st.button("Update Notes"):
                    if batch_notes:
                        try:
                            # Use the W&B API to update notes for each run
                            api = wandb.Api()
                            
                            success_count = 0
                            for run_id in selected_run_ids:
                                try:
                                    run = api.run(f"{st.session_state.wandb_entity}/{st.session_state.selected_project}/{run_id}")
                                    
                                    if append_mode and run.notes:
                                        run.notes = f"{run.notes}\n\n{batch_notes}"
                                    else:
                                        run.notes = batch_notes
                                    
                                    run.update()
                                    success_count += 1
                                except Exception as e:
                                    st.error(f"Error updating notes for run {run_id}: {str(e)}")
                            
                            if success_count > 0:
                                st.success(f"Updated notes for {success_count} runs. Refresh the page to see updates.")
                                # Update session state runs data
                                st.session_state.refresh_required = True
                        except Exception as e:
                            st.error(f"Error accessing W&B API: {str(e)}")
                    else:
                        st.warning("Please enter notes to update.")
                
                # Add custom metadata
                st.markdown("#### Add Custom Metadata")
                st.info("Custom metadata will be added to the summary section of each run.")
                
                col1, col2 = st.columns(2)
                with col1:
                    metadata_key = st.text_input("Metadata Key:")
                with col2:
                    metadata_value = st.text_input("Metadata Value:")
                
                if st.button("Add Metadata"):
                    if metadata_key and metadata_value:
                        try:
                            # Use the W&B API to add metadata to each run
                            api = wandb.Api()
                            
                            success_count = 0
                            for run_id in selected_run_ids:
                                try:
                                    run = api.run(f"{st.session_state.wandb_entity}/{st.session_state.selected_project}/{run_id}")
                                    
                                    # Try to convert value to the appropriate type
                                    try:
                                        if metadata_value.lower() == "true":
                                            typed_value = True
                                        elif metadata_value.lower() == "false":
                                            typed_value = False
                                        elif metadata_value.isdigit():
                                            typed_value = int(metadata_value)
                                        else:
                                            try:
                                                typed_value = float(metadata_value)
                                            except ValueError:
                                                typed_value = metadata_value
                                    except:
                                        typed_value = metadata_value
                                    
                                    # Update the run's summary with the new metadata
                                    run.summary[metadata_key] = typed_value
                                    run.update()
                                    success_count += 1
                                except Exception as e:
                                    st.error(f"Error adding metadata to run {run_id}: {str(e)}")
                            
                            if success_count > 0:
                                st.success(f"Added metadata to {success_count} runs. Refresh the page to see updates.")
                                # Update session state runs data
                                st.session_state.refresh_required = True
                        except Exception as e:
                            st.error(f"Error accessing W&B API: {str(e)}")
                    else:
                        st.warning("Please enter both a metadata key and value.")
                
            # Artifact Management Tab
            with artifact_tab:
                st.markdown("### Artifact Management")
                
                st.markdown("""
                This tab allows you to view and download artifacts from multiple runs.
                You can batch download files, compare artifacts across runs, and more.
                """)
                
                # Collect all artifacts from the selected runs
                all_artifacts = []
                
                with st.spinner("Loading artifacts..."):
                    for run in selected_runs:
                        run_id = run["id"]
                        try:
                            api = wandb.Api()
                            full_run = api.run(f"{project_id}/{run_id}")
                            
                            # Get files from this run
                            files = []
                            for file in full_run.files():
                                files.append({
                                    "name": file.name,
                                    "size": file.size,
                                    "updated_at": file.updatedAt,
                                    "run_id": run_id,
                                    "run_name": run["name"]
                                })
                            
                            all_artifacts.extend(files)
                        except Exception as e:
                            st.error(f"Error loading artifacts for run {run_id}: {str(e)}")
                
                if all_artifacts:
                    # Convert to DataFrame for easier filtering and display
                    artifacts_df = pd.DataFrame(all_artifacts)
                    
                    # Convert file sizes to human-readable format
                    def format_size(size_bytes):
                        for unit in ['B', 'KB', 'MB', 'GB']:
                            if size_bytes < 1024.0 or unit == 'GB':
                                return f"{size_bytes:.2f} {unit}"
                            size_bytes /= 1024.0
                    
                    artifacts_df['readable_size'] = artifacts_df['size'].apply(format_size)
                    
                    # Add formatted date
                    artifacts_df['date'] = pd.to_datetime(artifacts_df['updated_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Display artifact table
                    st.subheader("All Artifacts")
                    st.dataframe(artifacts_df[['name', 'readable_size', 'date', 'run_name', 'run_id']], use_container_width=True)
                    
                    # Filter artifacts
                    st.subheader("Filter Artifacts")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Filter by file extension
                        file_extensions = ['All'] + sorted(list({name.split('.')[-1] if '.' in name else 'No extension' 
                                                        for name in artifacts_df['name']}))
                        selected_extension = st.selectbox("Filter by file type:", file_extensions)
                    
                    with col2:
                        # Filter by run
                        run_options = ['All'] + sorted(list(set(artifacts_df['run_name'])))
                        selected_run = st.selectbox("Filter by run:", run_options)
                    
                    # Apply filters
                    filtered_df = artifacts_df.copy()
                    if selected_extension != 'All':
                        filtered_df = filtered_df[filtered_df['name'].str.endswith(f'.{selected_extension}')]
                    if selected_run != 'All':
                        filtered_df = filtered_df[filtered_df['run_name'] == selected_run]
                    
                    # Search filter
                    search_term = st.text_input("Search artifacts by name:", placeholder="Enter search term...")
                    if search_term:
                        filtered_df = filtered_df[filtered_df['name'].str.contains(search_term, case=False)]
                    
                    # Display filtered artifacts
                    st.subheader("Filtered Artifacts")
                    if not filtered_df.empty:
                        st.dataframe(filtered_df[['name', 'readable_size', 'date', 'run_name', 'run_id']], use_container_width=True)
                        
                        # Batch download section
                        st.subheader("Batch Download")
                        
                        # Multi-select for files to download
                        file_options = [f"{row['name']} ({row['run_name']})" for _, row in filtered_df.iterrows()]
                        selected_files = st.multiselect("Select files to download:", file_options)
                        
                        if selected_files:
                            if st.button("Download Selected Files"):
                                try:
                                    # Create a temporary directory to store files
                                    with tempfile.TemporaryDirectory() as tmp_dir:
                                        downloaded_files = []
                                        
                                        for file_option in selected_files:
                                            file_name = file_option.split(" (")[0]
                                            run_name = file_option.split("(")[1].split(")")[0]
                                            
                                            # Find run_id for this file
                                            file_info = filtered_df[(filtered_df['name'] == file_name) & 
                                                                   (filtered_df['run_name'] == run_name)].iloc[0]
                                            run_id = file_info['run_id']
                                            
                                            try:
                                                # Download the file
                                                api = wandb.Api()
                                                run = api.run(f"{project_id}/{run_id}")
                                                
                                                file_path = os.path.join(tmp_dir, f"{run_name}_{file_name}")
                                                run.file(file_name).download(root=tmp_dir, replace=True)
                                                actual_path = os.path.join(tmp_dir, file_name)
                                                
                                                # Create a more descriptive filename that includes the run name
                                                renamed_path = os.path.join(tmp_dir, f"{run_name}_{file_name}")
                                                if os.path.exists(actual_path):
                                                    # Rename the file to include the run name for clarity
                                                    os.rename(actual_path, renamed_path)
                                                    downloaded_files.append(renamed_path)
                                            except Exception as e:
                                                st.error(f"Error downloading {file_name} from run {run_name}: {str(e)}")
                                        
                                        if downloaded_files:
                                            # Create ZIP file of all downloaded files
                                            zip_path = os.path.join(tmp_dir, "wandb_artifacts.zip")
                                            self.create_zip_file(downloaded_files, zip_path)
                                            
                                            # Provide download link for the ZIP file
                                            with open(zip_path, "rb") as f:
                                                bytes_data = f.read()
                                                st.download_button(
                                                    label="Download ZIP File",
                                                    data=bytes_data,
                                                    file_name="wandb_artifacts.zip",
                                                    mime="application/zip"
                                                )
                                    
                                except Exception as e:
                                    st.error(f"Error processing files: {str(e)}")
                            
                            # Option to download files individually
                            st.markdown("#### Individual File Downloads")
                            for file_option in selected_files:
                                file_name = file_option.split(" (")[0]
                                run_name = file_option.split("(")[1].split(")")[0]
                                
                                # Find run_id for this file
                                file_info = filtered_df[(filtered_df['name'] == file_name) & 
                                                       (filtered_df['run_name'] == run_name)].iloc[0]
                                run_id = file_info['run_id']
                                
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**{file_name}** from run *{run_name}*")
                                with col2:
                                    if st.button(f"Download", key=f"dl_{file_name}_{run_id}"):
                                        try:
                                            api = wandb.Api()
                                            run = api.run(f"{project_id}/{run_id}")
                                            
                                            with tempfile.TemporaryDirectory() as tmp_dir:
                                                actual_path = os.path.join(tmp_dir, file_name)
                                                
                                                if os.path.exists(actual_path):
                                                    os.rename(actual_path, file_path)
                                                    downloaded_files.append((file_path, f"{run_name}_{file_name}"))
                                            except Exception as e:
                                                st.error(f"Error downloading {file_name} from run {run_id}: {str(e)}")
                                        
                                        if downloaded_files:
                                            # Create a zip file with all downloaded files
                                            zip_path = os.path.join(tmp_dir, "batch_artifacts.zip")
                                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                                for file_path, arcname in downloaded_files:
                                                    zipf.write(file_path, arcname=arcname)
                                            
                                            # Read the zip file and provide download link
                                            with open(zip_path, "rb") as f:
                                                zip_data = f.read()
                                                
                                            b64 = base64.b64encode(zip_data).decode()
                                            href = f'<a href="data:application/zip;base64,{b64}" download="batch_artifacts.zip">Download ZIP File</a>'
                                            st.markdown(href, unsafe_allow_html=True)
                                            st.success(f"Successfully downloaded {len(downloaded_files)} files.")
                                        else:
                                            st.error("No files were successfully downloaded.")
                                except Exception as e:
                                    st.error(f"Error creating batch download: {str(e)}")
                    else:
                        st.info("No artifacts match the selected filters.")
                else:
                    st.info("No artifacts found for the selected runs.")
                
            # Deletion & Archiving Tab
            with deletion_tab:
                st.markdown("### Deletion & Archiving")
                
                st.markdown("""
                This tab allows you to perform batch deletion or archiving operations.
                Please use with caution as these operations cannot be easily undone.
                """)
                
                # Create tabs for different operations
                delete_tab, archive_tab = st.tabs([
                    "Delete Runs", "Archive Runs"
                ])
                
                # Delete Runs Tab
                with delete_tab:
                    st.markdown("#### Delete Selected Runs")
                    st.warning("⚠️ **Warning**: Deleting runs is permanent and cannot be undone.")
                    
                    # Display runs to be deleted
                    delete_data = []
                    for run in selected_runs:
                        delete_data.append({
                            "Run": run["name"],
                            "ID": run["id"],
                            "Created": pd.to_datetime(run["created_at"]).strftime('%Y-%m-%d %H:%M:%S') if "created_at" in run else "",
                            "State": run["state"]
                        })
                    
                    delete_df = pd.DataFrame(delete_data)
                    st.dataframe(delete_df, use_container_width=True)
                    
                    # Confirm deletion
                    confirm_text = st.text_input("Type 'DELETE' to confirm deletion:")
                    
                    if st.button("Delete Selected Runs"):
                        if confirm_text == "DELETE":
                            try:
                                api = wandb.Api()
                                success_count = 0
                                
                                for run_id in selected_run_ids:
                                    try:
                                        run = api.run(f"{project_id}/{run_id}")
                                        run.delete()
                                        success_count += 1
                                    except Exception as e:
                                        st.error(f"Error deleting run {run_id}: {str(e)}")
                                
                                if success_count > 0:
                                    st.success(f"Successfully deleted {success_count} runs.")
                                    # We need to refresh the project runs
                                    st.session_state.pop('project_runs', None)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error accessing W&B API: {str(e)}")
                        else:
                            st.error("Please type 'DELETE' to confirm the deletion.")
                
                # Archive Runs Tab
                with archive_tab:
                    st.markdown("#### Archive Selected Runs")
                    st.info("Archiving runs marks them as archived in Weights & Biases without deleting data.")
                    
                    # Display runs to be archived
                    archive_data = []
                    for run in selected_runs:
                        archive_data.append({
                            "Run": run["name"],
                            "ID": run["id"],
                            "Created": pd.to_datetime(run["created_at"]).strftime('%Y-%m-%d %H:%M:%S') if "created_at" in run else "",
                            "State": run["state"]
                        })
                    
                    archive_df = pd.DataFrame(archive_data)
                    st.dataframe(archive_df, use_container_width=True)
                    
                    if st.button("Archive Selected Runs"):
                        try:
                            api = wandb.Api()
                            success_count = 0
                            
                            for run_id in selected_run_ids:
                                try:
                                    run = api.run(f"{project_id}/{run_id}")
                                    # Add an archive tag to the run
                                    run.tags = list(set(run.tags + ["archived"]))
                                    # Also update its state if possible
                                    if hasattr(run, "state"):
                                        run.state = "archived"
                                    run.update()
                                    success_count += 1
                                except Exception as e:
                                    st.error(f"Error archiving run {run_id}: {str(e)}")
                            
                            if success_count > 0:
                                st.success(f"Successfully archived {success_count} runs by adding 'archived' tag.")
                                # We should refresh the project runs
                                st.session_state.refresh_required = True
                        except Exception as e:
                            st.error(f"Error accessing W&B API: {str(e)}")
    
    else:
        if selected_run_ids:
            st.info("Please select at least one more run to perform batch operations (minimum 2 runs required).")
        else:
            st.info("Please select at least 2 runs to perform batch operations.")