import wandb
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

def format_timestamp(timestamp):
    """Convert Unix timestamp to readable date format"""
    if timestamp:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return "N/A"

def get_run_summary(run):
    """Extract summary metrics from a run"""
    summary_data = {"ID": run.id, "Name": run.name}
    
    if hasattr(run, 'summary') and run.summary:
        for key, value in run.summary.items():
            if not key.startswith('_') and isinstance(value, (int, float, str, bool)) or value is None:
                summary_data[key] = value
    
    return summary_data

def get_run_config(run, flatten=True):
    """Extract configuration from a run"""
    if not run.config:
        return {"ID": run.id, "Name": run.name}
    
    if not flatten:
        return {"ID": run.id, "Name": run.name, "config": run.config}
    
    # Flatten the config
    config_data = {"ID": run.id, "Name": run.name}
    
    def flatten_dict(d, parent_key=''):
        for k, v in d.items():
            key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                flatten_dict(v, key)
            else:
                if isinstance(v, (int, float, str, bool)) or v is None:
                    config_data[key] = v
    
    flatten_dict(run.config)
    return config_data

def get_filtered_history(run, columns=None, start_step=None, end_step=None):
    """Get filtered history data from a run"""
    try:
        history = run.scan_history()
        history_df = pd.DataFrame(history)
        
        if history_df.empty:
            return pd.DataFrame()
        
        # Filter by steps if specified
        if '_step' in history_df.columns:
            if start_step is not None:
                history_df = history_df[history_df['_step'] >= start_step]
            if end_step is not None:
                history_df = history_df[history_df['_step'] <= end_step]
        
        # Filter columns if specified
        if columns:
            # Add _step column if it exists and not already included
            if '_step' in history_df.columns and '_step' not in columns:
                columns = ['_step'] + columns
            
            # Filter to specified columns
            available_columns = [col for col in columns if col in history_df.columns]
            history_df = history_df[available_columns]
        
        return history_df
    
    except Exception as e:
        st.error(f"Error fetching history: {str(e)}")
        return pd.DataFrame()

def get_best_runs_from_sweep(sweep, metric_name=None, top_n=5, goal="minimize"):
    """Get the best performing runs from a sweep"""
    if not metric_name:
        metric_name = sweep.config.get('metric', {}).get('name')
        if not metric_name:
            return pd.DataFrame()
    
    # Get all runs from the sweep
    runs = list(sweep.runs)
    
    # Filter to only completed runs
    completed_runs = [run for run in runs if run.state == "finished"]
    
    if not completed_runs:
        return pd.DataFrame()
    
    # Extract run data with the metric
    runs_data = []
    
    for run in completed_runs:
        if hasattr(run, 'summary') and run.summary and metric_name in run.summary:
            metric_value = run.summary[metric_name]
            
            if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                run_data = {
                    "id": run.id,
                    "name": run.name,
                    metric_name: metric_value,
                    "created_at": format_timestamp(run.created_at),
                    "runtime": round((run.runtime or 0) / 60, 2) if run.runtime else "N/A"
                }
                
                # Add config parameters
                if run.config:
                    for key, value in run.config.items():
                        if isinstance(value, (int, float, str, bool)) or value is None:
                            run_data[f"config.{key}"] = value
                
                runs_data.append(run_data)
    
    if not runs_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    runs_df = pd.DataFrame(runs_data)
    
    # Sort based on goal
    ascending = goal.lower() == "minimize"
    runs_df = runs_df.sort_values(by=metric_name, ascending=ascending)
    
    # Return top N runs
    return runs_df.head(top_n)

def download_artifact_files(artifact, output_dir=None, file_paths=None):
    """
    Download files from an artifact
    
    Args:
        artifact: W&B artifact object
        output_dir: Directory to save files (if None, use current directory)
        file_paths: List of specific file paths to download (if None, download all)
    
    Returns:
        List of downloaded file paths
    """
    try:
        # Download artifact
        artifact_dir = artifact.download(root=output_dir)
        
        # If specific files requested, return those paths
        if file_paths:
            return [f"{artifact_dir}/{path}" for path in file_paths]
        
        # Otherwise return the artifact directory
        return artifact_dir
        
    except Exception as e:
        st.error(f"Error downloading artifact: {str(e)}")
        return None
