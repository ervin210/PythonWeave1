import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_metric_over_time(history_df, metric_name):
    """Plot a single metric over time."""
    if metric_name not in history_df.columns or history_df.empty:
        st.warning(f"Metric '{metric_name}' not found in history data.")
        return None
    
    # Filter out NaN values for the metric
    filtered_df = history_df.dropna(subset=[metric_name])
    
    if filtered_df.empty:
        st.warning(f"No valid data points for metric '{metric_name}'.")
        return None
    
    # Try to find a suitable x-axis
    x_column = "_step"
    if x_column not in filtered_df.columns:
        x_column = "step" if "step" in filtered_df.columns else None
    
    if x_column is None:
        # If no step column found, use row index as steps
        filtered_df = filtered_df.reset_index()
        x_column = "index"
    
    # Create the line plot
    fig = px.line(
        filtered_df, 
        x=x_column, 
        y=metric_name,
        title=f"{metric_name} over time",
        labels={x_column: "Step", metric_name: metric_name.capitalize()},
    )
    
    # Improve layout
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig

def plot_multiple_metrics(history_df, metric_names):
    """Plot multiple metrics over time."""
    if not metric_names or history_df.empty:
        st.warning("No metrics selected or no history data available.")
        return None
    
    # Check which metrics are actually in the dataframe
    available_metrics = [m for m in metric_names if m in history_df.columns]
    
    if not available_metrics:
        st.warning("None of the selected metrics are available in the history data.")
        return None
    
    # Try to find a suitable x-axis
    x_column = "_step"
    if x_column not in history_df.columns:
        x_column = "step" if "step" in history_df.columns else None
    
    if x_column is None:
        # If no step column found, use row index as steps
        history_df = history_df.reset_index()
        x_column = "index"
    
    # Create subplots if there are multiple metrics
    if len(available_metrics) > 1:
        fig = make_subplots(
            rows=len(available_metrics), 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=available_metrics
        )
        
        for i, metric in enumerate(available_metrics):
            # Filter out NaN values for this metric
            filtered_df = history_df.dropna(subset=[metric])
            
            if not filtered_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df[x_column],
                        y=filtered_df[metric],
                        mode='lines',
                        name=metric
                    ),
                    row=i+1, 
                    col=1
                )
        
        # Update layout
        fig.update_layout(
            height=300 * len(available_metrics),
            title_text="Metrics over time",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
    else:
        # Single metric, use the simpler plot_metric_over_time function
        metric = available_metrics[0]
        fig = plot_metric_over_time(history_df, metric)
    
    return fig

def plot_parameter_importance(runs_data, metric_name):
    """Plot parameter importance for a given metric."""
    if not runs_data or not metric_name:
        st.warning("No runs data or metric specified.")
        return None
    
    # Extract runs that have the metric in their summary
    valid_runs = []
    for run in runs_data:
        if "summary" in run and metric_name in run["summary"] and "config" in run:
            valid_runs.append({
                "id": run["id"],
                "metric_value": run["summary"][metric_name],
                **{f"param_{k}": v for k, v in run["config"].items() if not isinstance(v, (dict, list))}
            })
    
    if not valid_runs:
        st.warning(f"No runs with metric '{metric_name}' found.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(valid_runs)
    
    # Find parameters that vary across runs
    param_columns = [col for col in df.columns if col.startswith("param_")]
    varying_params = []
    for param in param_columns:
        if df[param].nunique() > 1:
            varying_params.append(param)
    
    if not varying_params:
        st.warning("No varying parameters found across runs.")
        return None
    
    # Create parallel coordinates plot
    dimensions = [
        dict(
            label=param.replace("param_", ""),
            values=df[param]
        )
        for param in varying_params
    ]
    
    # Add the metric as the final dimension
    dimensions.append(
        dict(
            label=metric_name,
            values=df["metric_value"],
            range=[min(df["metric_value"]), max(df["metric_value"])]
        )
    )
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=df["metric_value"],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=metric_name
            )
        ),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        title=f"Parameter Importance for {metric_name}",
        height=600,
        margin=dict(l=100, r=20, t=60, b=20),
    )
    
    return fig

def plot_run_comparisons(runs_data, metric_names):
    """Plot comparisons of selected metrics across multiple runs."""
    if not runs_data or not metric_names:
        st.warning("No runs data or metrics specified.")
        return None
    
    # Prepare data for plotting
    comparison_data = []
    for run in runs_data:
        if "summary" in run:
            run_data = {"run_id": run["id"], "run_name": run.get("name", run["id"])}
            for metric in metric_names:
                if metric in run["summary"]:
                    value = run["summary"][metric]
                    # Make sure the value is numeric
                    if isinstance(value, (int, float)):
                        run_data[metric] = value
            
            if len(run_data) > 2:  # Only add if at least one metric is present
                comparison_data.append(run_data)
    
    if not comparison_data:
        st.warning("No valid data for comparison.")
        return None
    
    df = pd.DataFrame(comparison_data)
    
    # Create plots based on the number of metrics
    if len(metric_names) == 1:
        # Single metric - bar chart
        metric = metric_names[0]
        if metric in df.columns:
            fig = px.bar(
                df, 
                x="run_name", 
                y=metric,
                title=f"Comparison of {metric} across runs",
                labels={"run_name": "Run", metric: metric.capitalize()},
                hover_data=["run_id"]
            )
            
            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=100),
                xaxis_tickangle=-45
            )
            
            return fig
        else:
            st.warning(f"Metric '{metric}' not found in any run.")
            return None
    else:
        # Multiple metrics - grouped bar chart
        # First, reshape dataframe to long format
        metrics_in_df = [m for m in metric_names if m in df.columns]
        
        if not metrics_in_df:
            st.warning("None of the selected metrics are available in any run.")
            return None
        
        df_long = pd.melt(
            df,
            id_vars=["run_id", "run_name"],
            value_vars=metrics_in_df,
            var_name="metric",
            value_name="value"
        )
        
        fig = px.bar(
            df_long,
            x="run_name",
            y="value",
            color="metric",
            barmode="group",
            title="Comparison of metrics across runs",
            labels={"run_name": "Run", "value": "Value", "metric": "Metric"},
            hover_data=["run_id"]
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=100),
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig

def plot_sweep_results(sweep_data, metric_name):
    """Plot the results of a parameter sweep for a specific metric."""
    if not sweep_data or "runs" not in sweep_data or not sweep_data["runs"]:
        st.warning("No sweep data available.")
        return None
    
    # Extract runs that have the metric in their summary
    runs = []
    for run in sweep_data["runs"]:
        if "summary" in run and metric_name in run["summary"]:
            runs.append({
                "id": run["id"],
                "name": run.get("name", run["id"]),
                "metric_value": run["summary"][metric_name],
                **{k: v for k, v in run["config"].items() if not isinstance(v, (dict, list))}
            })
    
    if not runs:
        st.warning(f"No runs with metric '{metric_name}' found in the sweep.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(runs)
    
    # Sort by metric value
    df = df.sort_values("metric_value", ascending=False)
    
    # Create a bar chart of the metric values
    fig = px.bar(
        df,
        x="name",
        y="metric_value",
        title=f"{metric_name} values across sweep runs",
        labels={"name": "Run", "metric_value": metric_name},
        hover_data=["id"]
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=100),
        xaxis_tickangle=-45
    )
    
    return fig

def plot_confusion_matrix(confusion_matrix_data, class_names=None):
    """Plot a confusion matrix."""
    if confusion_matrix_data is None or not isinstance(confusion_matrix_data, (list, np.ndarray)):
        st.warning("Invalid confusion matrix data.")
        return None
    
    # Convert to numpy array if it's a list
    if isinstance(confusion_matrix_data, list):
        confusion_matrix_data = np.array(confusion_matrix_data)
    
    # Check if the matrix is square
    if confusion_matrix_data.shape[0] != confusion_matrix_data.shape[1]:
        st.warning("Confusion matrix must be square.")
        return None
    
    # If no class names provided, generate default ones
    num_classes = confusion_matrix_data.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Create heatmap
    fig = px.imshow(
        confusion_matrix_data,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale="Blues"
    )
    
    fig.update_layout(
        title="Confusion Matrix",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig
