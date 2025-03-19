import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
# Import visualization functions directly from the correct module
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
        summary_tab, parallel_tab, hyperparameter_tab, table_tab, best_runs_tab, quantum_tab = st.tabs([
            "Summary", "Parallel Coordinates", "Hyperparameter Analysis", "All Runs", "Best Runs", "Quantum Analysis"
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

        # Hyperparameter Analysis tab
        with hyperparameter_tab:
            st.subheader("Advanced Hyperparameter Analysis")
            
            if len(config_keys) > 0 and len(metric_keys) > 0:
                # Get optimization metric
                if metric_name and metric_name in sweep_df.columns:
                    target_metric = st.selectbox(
                        "Select target metric for analysis",
                        options=list(metric_keys),
                        index=list(metric_keys).index(metric_name) if metric_name in metric_keys else 0
                    )
                else:
                    target_metric = st.selectbox(
                        "Select target metric for analysis",
                        options=list(metric_keys)
                    )
                
                # Filter to completed runs only for better analysis
                completed_df = sweep_df[sweep_df['state'] == 'finished']
                
                if completed_df.empty:
                    st.warning("No completed runs found for hyperparameter analysis")
                else:
                    # Determine hyperparameter importance
                    st.markdown("### Hyperparameter Importance")
                    
                    # Select hyperparameters to analyze
                    hp_params = [p for p in config_keys if p in completed_df.columns]
                    
                    # Filter out non-numeric or constant parameters
                    numeric_params = []
                    for p in hp_params:
                        if pd.api.types.is_numeric_dtype(completed_df[p]) and completed_df[p].nunique() > 1:
                            numeric_params.append(p)
                    
                    if not numeric_params:
                        st.info("No numeric hyperparameters with variation found for importance analysis")
                    else:
                        # Basic correlation analysis
                        st.markdown("#### Correlation Analysis")
                        corr_data = []
                        
                        for param in numeric_params:
                            # Calculate correlation with target metric
                            correlation = completed_df[[param, target_metric]].corr().iloc[0, 1]
                            abs_corr = abs(correlation)
                            corr_data.append({
                                "Parameter": param,
                                "Correlation": correlation,
                                "Absolute Correlation": abs_corr
                            })
                        
                        # Sort by absolute correlation
                        corr_df = pd.DataFrame(corr_data).sort_values("Absolute Correlation", ascending=False)
                        
                        # Display correlation table
                        st.dataframe(corr_df, use_container_width=True)
                        
                        # Visualize top parameter correlations
                        top_params = corr_df.head(min(5, len(corr_df)))["Parameter"].tolist()
                        
                        if top_params:
                            st.markdown("#### Top Parameter Effects")
                            
                            for param in top_params:
                                # Create scatter plot with trend line
                                fig = px.scatter(
                                    completed_df,
                                    x=param,
                                    y=target_metric,
                                    trendline="ols",
                                    title=f"Effect of {param} on {target_metric}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Hyperparameter interaction analysis
                    st.markdown("### Parameter Interactions")
                    
                    # Select parameters for interaction analysis
                    interaction_params = st.multiselect(
                        "Select parameters to analyze interactions",
                        options=hp_params,
                        default=hp_params[:min(2, len(hp_params))] if len(hp_params) >= 2 else []
                    )
                    
                    if len(interaction_params) >= 2:
                        # Pair of parameters to analyze
                        param1 = interaction_params[0]
                        param2 = interaction_params[1]
                        
                        # Check if both are numeric
                        if pd.api.types.is_numeric_dtype(completed_df[param1]) and pd.api.types.is_numeric_dtype(completed_df[param2]):
                            # Create 3D surface plot
                            fig = go.Figure(data=[go.Scatter3d(
                                x=completed_df[param1],
                                y=completed_df[param2],
                                z=completed_df[target_metric],
                                mode='markers',
                                marker=dict(
                                    size=5,
                                    color=completed_df[target_metric],
                                    colorscale='Viridis',
                                    opacity=0.8,
                                    colorbar=dict(title=target_metric)
                                )
                            )])
                            
                            fig.update_layout(
                                title=f"Interaction between {param1} and {param2} on {target_metric}",
                                scene=dict(
                                    xaxis_title=param1,
                                    yaxis_title=param2,
                                    zaxis_title=target_metric
                                ),
                                height=700
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # At least one parameter is categorical
                            # Create grouped box plot
                            if not pd.api.types.is_numeric_dtype(completed_df[param1]):
                                # param1 is categorical, param2 is numeric
                                fig = px.box(
                                    completed_df,
                                    x=param1,
                                    y=target_metric,
                                    color=param2 if pd.api.types.is_numeric_dtype(completed_df[param2]) else None,
                                    title=f"Effect of {param1} and {param2} on {target_metric}"
                                )
                            elif not pd.api.types.is_numeric_dtype(completed_df[param2]):
                                # param2 is categorical, param1 is numeric
                                fig = px.box(
                                    completed_df,
                                    x=param2,
                                    y=target_metric,
                                    color=param1,
                                    title=f"Effect of {param1} and {param2} on {target_metric}"
                                )
                            else:
                                # Both are categorical
                                # Create grouped bar chart
                                fig = px.bar(
                                    completed_df.groupby([param1, param2])[target_metric].mean().reset_index(),
                                    x=param1,
                                    y=target_metric,
                                    color=param2,
                                    title=f"Effect of {param1} and {param2} on {target_metric}",
                                    barmode="group"
                                )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Search space visualization
                    st.markdown("### Search Space Analysis")
                    
                    # Get parameter search ranges
                    param_ranges = {}
                    for param in numeric_params:
                        param_ranges[param] = {
                            "min": completed_df[param].min(),
                            "max": completed_df[param].max(),
                            "unique_values": completed_df[param].nunique()
                        }
                    
                    # Display parameter ranges
                    ranges_data = []
                    for param, range_info in param_ranges.items():
                        ranges_data.append({
                            "Parameter": param,
                            "Min": range_info["min"],
                            "Max": range_info["max"],
                            "Range": range_info["max"] - range_info["min"],
                            "Unique Values": range_info["unique_values"]
                        })
                    
                    if ranges_data:
                        ranges_df = pd.DataFrame(ranges_data)
                        st.dataframe(ranges_df, use_container_width=True)
                        
                        # Visualize parameter exploration
                        st.markdown("#### Parameter Exploration")
                        
                        # Select parameter to visualize
                        if numeric_params:
                            vis_param = st.selectbox(
                                "Select parameter to visualize exploration",
                                options=numeric_params
                            )
                            
                            # Create scatter plot of parameter vs run index
                            param_df = completed_df.sort_values("created_at").reset_index(drop=True)
                            param_df["Run Index"] = param_df.index
                            
                            fig = px.scatter(
                                param_df,
                                x="Run Index",
                                y=vis_param,
                                color=target_metric,
                                size=abs(param_df[target_metric] - param_df[target_metric].mean()),
                                title=f"Exploration of {vis_param} Over Sweep"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No numeric parameters found for search space analysis")
            else:
                st.info("Insufficient data for hyperparameter analysis")
        
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
