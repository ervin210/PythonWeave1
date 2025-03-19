import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_metrics_history(history_df, metrics, smoothing=None):
    """
    Create line plots for selected metrics over time
    
    Args:
        history_df: DataFrame with metrics history
        metrics: List of metric names to plot
        smoothing: Optional smoothing factor (0-1)
    
    Returns:
        Plotly figure object
    """
    if '_step' not in history_df.columns:
        # If no step column, use row index
        x_col = history_df.index
        x_title = 'Row Index'
    else:
        x_col = '_step'
        x_title = 'Step'
    
    fig = go.Figure()
    
    for metric in metrics:
        y_data = history_df[metric]
        
        # Apply smoothing if requested
        if smoothing and smoothing > 0 and smoothing < 1:
            y_data = y_data.ewm(alpha=smoothing).mean()
        
        fig.add_trace(
            go.Scatter(
                x=history_df[x_col] if x_col != history_df.index else history_df.index,
                y=y_data,
                mode='lines',
                name=metric
            )
        )
    
    fig.update_layout(
        title='Metrics Over Time',
        xaxis_title=x_title,
        yaxis_title='Value',
        legend_title='Metrics',
        hovermode='x unified'
    )
    
    return fig

def create_parallel_coordinates_plot(df, dimensions, color_col):
    """
    Create a parallel coordinates plot for hyperparameter visualization
    
    Args:
        df: DataFrame with run data
        dimensions: List of columns to include as dimensions
        color_col: Column to use for coloring
    
    Returns:
        Plotly figure object
    """
    # Determine dimension ranges for categorical vs. numerical data
    dims = []
    
    for dim in dimensions:
        if dim in df.columns:
            if pd.api.types.is_numeric_dtype(df[dim]):
                # Numerical dimension
                dims.append(
                    dict(
                        range=[df[dim].min(), df[dim].max()],
                        label=dim,
                        values=df[dim]
                    )
                )
            else:
                # Categorical dimension
                values = df[dim].astype(str)
                dims.append(
                    dict(
                        tickvals=list(range(len(values.unique()))),
                        ticktext=values.unique(),
                        label=dim,
                        values=values.map(dict(zip(values.unique(), range(len(values.unique())))))
                    )
                )
    
    # Create figure
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df[color_col],
                colorscale='Viridis',
                showscale=True,
                cmin=df[color_col].min(),
                cmax=df[color_col].max(),
                colorbar=dict(title=color_col)
            ),
            dimensions=dims
        )
    )
    
    fig.update_layout(
        title='Parallel Coordinates Plot',
        height=600
    )
    
    return fig

def create_metric_comparison_plot(runs_df, x_metric, y_metric, color_by=None, size_by=None):
    """
    Create a scatter plot to compare two metrics across runs
    
    Args:
        runs_df: DataFrame with run data
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        color_by: Column to use for coloring points
        size_by: Column to use for point sizes
    
    Returns:
        Plotly figure object
    """
    if color_by and size_by:
        fig = px.scatter(
            runs_df,
            x=x_metric,
            y=y_metric,
            color=color_by,
            size=size_by,
            hover_name="name",
            hover_data=["id", "state"],
            title=f"{y_metric} vs {x_metric}",
            labels={
                x_metric: x_metric,
                y_metric: y_metric,
                color_by: color_by,
                size_by: size_by
            }
        )
    elif color_by:
        fig = px.scatter(
            runs_df,
            x=x_metric,
            y=y_metric,
            color=color_by,
            hover_name="name",
            hover_data=["id", "state"],
            title=f"{y_metric} vs {x_metric}",
            labels={
                x_metric: x_metric,
                y_metric: y_metric,
                color_by: color_by
            }
        )
    elif size_by:
        fig = px.scatter(
            runs_df,
            x=x_metric,
            y=y_metric,
            size=size_by,
            hover_name="name",
            hover_data=["id", "state"],
            title=f"{y_metric} vs {x_metric}",
            labels={
                x_metric: x_metric,
                y_metric: y_metric,
                size_by: size_by
            }
        )
    else:
        fig = px.scatter(
            runs_df,
            x=x_metric,
            y=y_metric,
            hover_name="name",
            hover_data=["id", "state"],
            title=f"{y_metric} vs {x_metric}",
            labels={
                x_metric: x_metric,
                y_metric: y_metric
            }
        )
    
    # Add trend line
    fig.update_layout(
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        height=600
    )
    
    return fig
