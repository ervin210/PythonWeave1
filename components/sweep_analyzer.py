import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils.visualization import create_parallel_coordinates_plot

def sweep_analyzer():
    """
    Analyze and visualize W&B parameter sweeps
    """
    if not st.session_state.authenticated or not st.session_state.api:
        st.warning("Please authenticate with W&B first")
        return
        
    if not st.session_state.selected_project:
        st.warning("Please select a project first")
        return
        
    if not st.session_state.selected_sweep:
        st.warning("Please select a sweep to analyze")
        return
        
    st.header("Sweep Analysis")
    
    try:
        # Get user and project
        user = st.session_state.api.viewer()['entity']
        project_name = st.session_state.selected_project
        sweep_id = st.session_state.selected_sweep
        
        # Load the sweep
        sweep = st.session_state.api.sweep(f"{user}/{project_name}/{sweep_id}")
        
        # Display sweep info
        st.subheader("Sweep Overview")
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown(f"**Sweep ID:** {sweep.id}")
            st.markdown(f"**Name:** {sweep.name or 'N/A'}")
            st.markdown(f"**Status:** {sweep.state}")
            
        with cols[1]:
            st.markdown(f"**Created:** {datetime.fromtimestamp(sweep.created).strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Method:** {sweep.config.get('method', 'N/A')}")
            st.markdown(f"**Metric:** {sweep.config.get('metric', {}).get('name', 'N/A')}")
            
        with cols[2]:
            st.markdown(f"**Total Runs:** {sweep.runs_count}")
            direction = sweep.config.get('metric', {}).get('goal', 'N/A')
            st.markdown(f"**Optimization Goal:** {direction}")
            
            # Generate W&B URL to the sweep
            wandb_url = f"https://wandb.ai/{user}/{project_name}/sweeps/{sweep_id}"
            st.markdown(f"[View in W&B]({wandb_url})")
        
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
                for key, value in run.config.items():
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        config_key = f"config.{key}"
                        run_data[config_key] = value
                        config_keys.add(config_key)
            
            # Extract summary metrics
            if hasattr(run, 'summary') and run.summary:
                for key, value in run.summary.items():
                    if not key.startswith('_') and isinstance(value, (int, float)) and not np.isnan(value):
                        run_data[key] = value
                        metric_keys.add(key)
            
            runs_data.append(run_data)
        
        # Create dataframe
        sweep_df = pd.DataFrame(runs_data)
        
        # Remove rows with missing metric values
        if len(metric_keys) > 0:
            metric_columns = list(metric_keys)
            sweep_df = sweep_df.dropna(subset=metric_columns)
        
        if sweep_df.empty:
            st.warning("No valid runs with complete metrics found")
            return
        
        # Create tabs for different visualizations
        summary_tab, parallel_tab, table_tab, best_runs_tab = st.tabs([
            "Summary", "Parallel Coordinates", "All Runs", "Best Runs"
        ])
        
        # Summary tab
        with summary_tab:
            st.subheader("Sweep Summary")
            
            # Summary statistics
            st.markdown(f"**Total Runs:** {len(sweep_df)}")
            st.markdown(f"**Completed Runs:** {len(sweep_df[sweep_df['state'] == 'finished'])}")
            
            # Get optimization metric
            metric_name = sweep.config.get('metric', {}).get('name', None)
            
            if metric_name and metric_name in sweep_df.columns:
                # Statistics for the optimization metric
                st.markdown(f"### Optimization Metric: {metric_name}")
                
                metric_stats = {
                    "Min": sweep_df[metric_name].min(),
                    "Max": sweep_df[metric_name].max(),
                    "Mean": sweep_df[metric_name].mean(),
                    "Median": sweep_df[metric_name].median(),
                    "Std Dev": sweep_df[metric_name].std()
                }
                
                # Display metric stats
                stats_df = pd.DataFrame({
                    "Statistic": metric_stats.keys(),
                    "Value": metric_stats.values()
                })
                
                st.dataframe(stats_df, use_container_width=True)
                
                # Histogram of the metric
                fig = px.histogram(
                    sweep_df,
                    x=metric_name,
                    nbins=20,
                    title=f"Distribution of {metric_name}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution of other metrics
                other_metrics = [m for m in metric_keys if m != metric_name]
                
                if other_metrics:
                    st.markdown("### Other Metrics")
                    
                    selected_metric = st.selectbox(
                        "Select metric to visualize",
                        options=other_metrics
                    )
                    
                    if selected_metric:
                        fig = px.histogram(
                            sweep_df,
                            x=selected_metric,
                            nbins=20,
                            title=f"Distribution of {selected_metric}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Parameter importance
            if len(config_keys) > 0 and metric_name and metric_name in sweep_df.columns:
                st.markdown("### Parameter Importance")
                
                # Display scatter plots for each parameter vs the metric
                config_params = list(config_keys)
                selected_params = st.multiselect(
                    "Select parameters to visualize",
                    options=config_params,
                    default=config_params[:min(3, len(config_params))]
                )
                
                if selected_params:
                    for param in selected_params:
                        if param in sweep_df.columns and pd.api.types.is_numeric_dtype(sweep_df[param]):
                            fig = px.scatter(
                                sweep_df,
                                x=param,
                                y=metric_name,
                                color='state',
                                title=f"{param} vs {metric_name}",
                                trendline="ols"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        elif param in sweep_df.columns:
                            # For categorical parameters
                            fig = px.box(
                                sweep_df,
                                x=param,
                                y=metric_name,
                                title=f"{param} vs {metric_name}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        # Parallel Coordinates tab
        with parallel_tab:
            st.subheader("Parallel Coordinates Plot")
            
            # Choose optimization metric for coloring
            if len(metric_keys) > 0:
                color_metric = st.selectbox(
                    "Select metric for coloring",
                    options=list(metric_keys),
                    index=list(metric_keys).index(metric_name) if metric_name in metric_keys else 0
                )
                
                # Select parameters for parallel coordinates
                parameter_cols = list(config_keys)
                selected_params = st.multiselect(
                    "Select parameters to include",
                    options=parameter_cols,
                    default=parameter_cols[:min(5, len(parameter_cols))]
                )
                
                if selected_params and color_metric:
                    dimensions = selected_params + [color_metric]
                    
                    # Remove non-numeric columns for parallel coordinates
                    numeric_dimensions = []
                    for dim in dimensions:
                        if dim in sweep_df.columns and pd.api.types.is_numeric_dtype(sweep_df[dim]):
                            numeric_dimensions.append(dim)
                    
                    if len(numeric_dimensions) >= 2:  # Need at least 2 dimensions
                        fig = create_parallel_coordinates_plot(
                            sweep_df, 
                            dimensions=numeric_dimensions,
                            color_col=color_metric
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need at least 2 numeric dimensions for parallel coordinates plot")
            else:
                st.info("No numeric metrics available for visualization")
        
        # Table of all runs
        with table_tab:
            st.subheader("All Runs")
            
            # Allow filtering by run state
            states = sweep_df['state'].unique().tolist()
            selected_states = st.multiselect(
                "Filter by state",
                options=states,
                default=states
            )
            
            filtered_df = sweep_df[sweep_df['state'].isin(selected_states)]
            
            # Allow sorting
            if len(metric_keys) > 0:
                sort_by = st.selectbox(
                    "Sort by",
                    options=['created_at'] + list(metric_keys)
                )
                
                sort_order = st.radio(
                    "Sort order",
                    options=["Ascending", "Descending"],
                    horizontal=True
                )
                
                ascending = sort_order == "Ascending"
                filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
            
            # Display the runs table
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download as CSV
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Runs CSV",
                data=csv,
                file_name=f"wandb_sweep_{sweep_id}_runs.csv",
                mime="text/csv"
            )
        
        # Best runs tab
        with best_runs_tab:
            st.subheader("Best Performing Runs")
            
            if len(metric_keys) > 0:
                # Select metric for ranking
                ranking_metric = st.selectbox(
                    "Select metric for ranking",
                    options=list(metric_keys),
                    index=list(metric_keys).index(metric_name) if metric_name in metric_keys else 0
                )
                
                # Select goal (minimize or maximize)
                goal = st.radio(
                    "Optimization Goal",
                    options=["Minimize", "Maximize"],
                    horizontal=True,
                    index=0 if sweep.config.get('metric', {}).get('goal', '') == 'minimize' else 1
                )
                
                # Number of top runs to show
                top_n = st.slider("Number of top runs to show", min_value=1, max_value=min(20, len(sweep_df)), value=5)
                
                # Filter to completed runs only
                completed_df = sweep_df[sweep_df['state'] == 'finished']
                
                if completed_df.empty:
                    st.warning("No completed runs found")
                else:
                    # Sort by the selected metric
                    ascending = goal == "Minimize"
                    top_runs = completed_df.sort_values(by=ranking_metric, ascending=ascending).head(top_n)
                    
                    # Display top runs
                    st.markdown(f"### Top {top_n} runs by {ranking_metric} ({goal.lower()})")
                    st.dataframe(top_runs, use_container_width=True)
                    
                    # Visualize metrics for top runs
                    st.markdown("### Metrics Comparison")
                    
                    # Select metrics to compare
                    compare_metrics = st.multiselect(
                        "Select metrics to compare",
                        options=list(metric_keys),
                        default=[ranking_metric]
                    )
                    
                    if compare_metrics:
                        # Create a bar chart for each metric
                        for metric in compare_metrics:
                            fig = px.bar(
                                top_runs,
                                x='name',
                                y=metric,
                                title=f"{metric} for Top Runs",
                                labels={'name': 'Run Name', 'y': metric}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # View details of a selected run
                    st.markdown("### View Run Details")
                    
                    selected_run_id = st.selectbox(
                        "Select a run to view details",
                        options=top_runs['id'].tolist(),
                        format_func=lambda x: f"{x} - {top_runs[top_runs['id']==x]['name'].iloc[0]}"
                    )
                    
                    if st.button("View Run Details"):
                        st.session_state.selected_run = selected_run_id
                        st.session_state.active_tab = "Run Details"
                        st.rerun()
            else:
                st.info("No metrics available for ranking runs")
    
    except Exception as e:
        st.error(f"Error analyzing sweep: {str(e)}")
