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
    
    else:
        if selected_run_ids:
            st.info("Please select at least one more run to perform batch operations (minimum 2 runs required).")
        else:
            st.info("Please select at least 2 runs to perform batch operations.")