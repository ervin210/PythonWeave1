import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
# Import visualization functions directly from the correct module
from utils.visualization import plot_metrics_history, create_parallel_coordinates_plot

def run_details():
    """
    View detailed information about a selected W&B run
    """
    if not st.session_state.authenticated or not st.session_state.api:
        st.warning("Please authenticate with W&B first")
        return
        
    if not st.session_state.selected_project:
        st.warning("Please select a project first")
        return
        
    if not st.session_state.selected_run:
        st.warning("Please select a run to view its details")
        return
        
    st.header("Run Details")
    
    try:
        # Get user and project
        user = st.session_state.api.viewer()['entity']
        project_name = st.session_state.selected_project
        run_id = st.session_state.selected_run
        
        # Load the run
        run = st.session_state.api.run(f"{user}/{project_name}/{run_id}")
        
        # Overview section
        st.subheader("Overview")
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown(f"**Run Name:** {run.name}")
            st.markdown(f"**Run ID:** {run.id}")
            st.markdown(f"**Status:** {run.state}")
            
        with cols[1]:
            created_time = datetime.fromtimestamp(run.created_at).strftime('%Y-%m-%d %H:%M:%S') if run.created_at else "N/A"
            runtime = f"{round(run.runtime / 60, 2)} minutes" if run.runtime else "N/A"
            
            st.markdown(f"**Created:** {created_time}")
            st.markdown(f"**Runtime:** {runtime}")
            st.markdown(f"**Heartbeat:** {datetime.fromtimestamp(run.heartbeat_at).strftime('%Y-%m-%d %H:%M:%S') if run.heartbeat_at else 'N/A'}")
            
        with cols[2]:
            st.markdown(f"**Tags:** {', '.join(run.tags) if run.tags else 'None'}")
            st.markdown(f"**User:** {run.user.name if hasattr(run, 'user') and run.user else 'N/A'}")
            
            # Generate W&B URL to the run
            wandb_url = f"https://wandb.ai/{user}/{project_name}/runs/{run_id}"
            st.markdown(f"[View in W&B]({wandb_url})")
        
        # Create tabs for different sections
        config_tab, metrics_tab, history_tab, performance_tab, files_tab = st.tabs([
            "Configuration", "Metrics", "History", "Model Performance", "Files & Artifacts"
        ])
        
        # Configuration tab
        with config_tab:
            st.subheader("Run Configuration")
            
            # Get run config
            config = run.config
            
            if not config:
                st.info("No configuration data available for this run")
            else:
                # Convert config to flat structure
                flat_config = {}
                
                def flatten_dict(d, parent_key=''):
                    for k, v in d.items():
                        key = f"{parent_key}.{k}" if parent_key else k
                        if isinstance(v, dict):
                            flatten_dict(v, key)
                        else:
                            flat_config[key] = v
                
                flatten_dict(config)
                
                # Group configs by categories
                config_categories = {}
                
                for key, value in flat_config.items():
                    category = key.split('.')[0] if '.' in key else 'general'
                    if category not in config_categories:
                        config_categories[category] = {}
                    config_categories[category][key] = value
                
                # Display config by categories
                for category, params in config_categories.items():
                    st.markdown(f"#### {category.capitalize()}")
                    
                    # Convert to DataFrame for better display
                    params_df = pd.DataFrame({
                        "Parameter": params.keys(),
                        "Value": params.values()
                    })
                    
                    st.dataframe(params_df, use_container_width=True)
                
                # Option to download config as JSON
                config_json = pd.io.json.dumps(config)
                st.download_button(
                    label="Download Config as JSON",
                    data=config_json,
                    file_name=f"wandb_run_{run.id}_config.json",
                    mime="application/json"
                )
        
        # Metrics tab
        with metrics_tab:
            st.subheader("Summary Metrics")
            
            summary = run.summary
            
            if not summary or all(k.startswith('_') for k in summary.keys()):
                st.info("No summary metrics available for this run")
            else:
                # Filter out internal keys and non-numeric values
                metrics = {k: v for k, v in summary.items() 
                          if not k.startswith('_') and isinstance(v, (int, float))}
                
                if not metrics:
                    st.info("No numeric metrics available for this run")
                else:
                    # Convert to DataFrame for better display
                    metrics_df = pd.DataFrame({
                        "Metric": metrics.keys(),
                        "Value": metrics.values()
                    })
                    
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Allow user to select metrics to visualize
                    selected_metrics = st.multiselect(
                        "Select metrics to visualize",
                        options=list(metrics.keys()),
                        default=list(metrics.keys())[:min(3, len(metrics))]
                    )
                    
                    if selected_metrics:
                        # Create a bar chart for selected metrics
                        fig = px.bar(
                            x=selected_metrics,
                            y=[metrics[m] for m in selected_metrics],
                            labels={'x': 'Metric', 'y': 'Value'},
                            title="Summary Metrics"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # History tab
        with history_tab:
            st.subheader("Metrics History")
            
            # Get run history
            try:
                history = run.scan_history()
                history_df = pd.DataFrame(history)
                
                if history_df.empty:
                    st.info("No history data available for this run")
                else:
                    # Filter out internal columns and keep only step and numeric columns
                    valid_columns = ['_step']
                    numeric_columns = []
                    
                    for col in history_df.columns:
                        if col == '_step':
                            continue
                        if col.startswith('_'):
                            continue
                        if pd.api.types.is_numeric_dtype(history_df[col]):
                            valid_columns.append(col)
                            numeric_columns.append(col)
                    
                    if not numeric_columns:
                        st.info("No numeric metrics found in history data")
                    else:
                        # Metric selection
                        selected_metrics = st.multiselect(
                            "Select metrics to plot",
                            options=numeric_columns,
                            default=numeric_columns[:min(2, len(numeric_columns))]
                        )
                        
                        if selected_metrics:
                            # Create metric history plots
                            fig = plot_metrics_history(history_df, selected_metrics)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show data table with history
                            st.subheader("Metrics Table")
                            
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
                            
                            # Allow downloading history data
                            csv = filtered_df.to_csv(index=False)
                            st.download_button(
                                label="Download History CSV",
                                data=csv,
                                file_name=f"wandb_run_{run.id}_history.csv",
                                mime="text/csv"
                            )
                            
                            # Display history table
                            display_columns = ['_step'] + selected_metrics
                            st.dataframe(filtered_df[display_columns], use_container_width=True)
            except Exception as e:
                st.error(f"Error loading history data: {str(e)}")
        
        # Model Performance tab
        with performance_tab:
            st.subheader("Model Performance Analysis")
            
            st.markdown("""
            This tab provides advanced visualizations and metrics for evaluating model performance.
            Depending on the type of ML task, different metrics and visualizations will be available.
            """)
            
            # Get run summary
            summary = run.summary
            
            # Check for common ML metrics in summary
            has_accuracy = any(k in str(summary.keys()).lower() for k in ['accuracy', 'acc'])
            has_loss = any(k in str(summary.keys()).lower() for k in ['loss'])
            has_precision = any(k in str(summary.keys()).lower() for k in ['precision'])
            has_recall = any(k in str(summary.keys()).lower() for k in ['recall', 'sensitivity'])
            has_f1 = any(k in str(summary.keys()).lower() for k in ['f1', 'f_score', 'f-score'])
            has_auc = any(k in str(summary.keys()).lower() for k in ['auc', 'roc'])
            has_mae = any(k in str(summary.keys()).lower() for k in ['mae', 'mean_absolute_error'])
            has_mse = any(k in str(summary.keys()).lower() for k in ['mse', 'mean_squared_error'])
            has_rmse = any(k in str(summary.keys()).lower() for k in ['rmse', 'root_mean_squared_error'])
            
            # Try to determine model type based on metrics
            is_classification = has_accuracy or has_precision or has_recall or has_f1 or has_auc
            is_regression = has_mae or has_mse or has_rmse
            
            # Let user select or confirm model type
            selected_model_type = st.selectbox(
                "Select Model Type",
                options=["Classification", "Regression", "Other"],
                index=0 if is_classification else (1 if is_regression else 2)
            )
            
            if selected_model_type == "Classification":
                # Classification performance metrics and visualizations
                st.markdown("### Classification Performance")
                
                # Metrics section
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.markdown("#### Key Metrics")
                    
                    # Check for and display classification metrics
                    metrics_found = False
                    
                    # Display accuracy
                    for k in summary.keys():
                        if 'accuracy' in k.lower() or k.lower() == 'acc':
                            st.metric("Accuracy", f"{summary[k]:.4f}")
                            metrics_found = True
                            break
                    
                    # Display precision
                    for k in summary.keys():
                        if 'precision' in k.lower():
                            st.metric("Precision", f"{summary[k]:.4f}")
                            metrics_found = True
                            break
                    
                    # Display recall
                    for k in summary.keys():
                        if 'recall' in k.lower() or 'sensitivity' in k.lower():
                            st.metric("Recall", f"{summary[k]:.4f}")
                            metrics_found = True
                            break
                    
                    # Display F1
                    for k in summary.keys():
                        if 'f1' in k.lower() or 'f_score' in k.lower() or 'f-score' in k.lower():
                            st.metric("F1 Score", f"{summary[k]:.4f}")
                            metrics_found = True
                            break
                    
                    # Display AUC
                    for k in summary.keys():
                        if 'auc' in k.lower() or 'roc' in k.lower():
                            st.metric("AUC-ROC", f"{summary[k]:.4f}")
                            metrics_found = True
                            break
                    
                    if not metrics_found:
                        st.info("No classification metrics found in run summary.")
                
                with metrics_col2:
                    # Check for confusion matrix in summary
                    confusion_matrix_found = False
                    for k in summary.keys():
                        if 'confusion' in k.lower() or 'cm' in k.lower():
                            st.markdown("#### Confusion Matrix")
                            try:
                                # Try to parse confusion matrix from summary
                                cm_value = summary[k]
                                if isinstance(cm_value, str):
                                    import json
                                    try:
                                        cm_data = json.loads(cm_value)
                                        # Plot confusion matrix
                                        fig = go.Figure(data=go.Heatmap(
                                            z=cm_data,
                                            text=cm_data,
                                            texttemplate="%{text}",
                                            colorscale="Blues"
                                        ))
                                        
                                        fig.update_layout(
                                            title="Confusion Matrix",
                                            height=300,
                                            xaxis=dict(title="Predicted Class"),
                                            yaxis=dict(title="True Class")
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        confusion_matrix_found = True
                                    except:
                                        pass
                            except:
                                pass
                    
                    if not confusion_matrix_found:
                        st.info("No confusion matrix data found.")
                
                # Check for class metrics in summary
                class_metrics_found = False
                for k in summary.keys():
                    if 'class' in k.lower() and any(m in k.lower() for m in ['precision', 'recall', 'f1']):
                        try:
                            class_data = summary[k]
                            if isinstance(class_data, str):
                                try:
                                    import json
                                    class_data = json.loads(class_data)
                                except:
                                    continue
                            
                            if isinstance(class_data, dict):
                                st.markdown("#### Per-Class Metrics")
                                
                                # Convert to DataFrame
                                classes = list(class_data.keys())
                                metrics_list = []
                                
                                for cls in classes:
                                    metrics_dict = {"Class": cls}
                                    metrics_dict.update(class_data[cls])
                                    metrics_list.append(metrics_dict)
                                
                                if metrics_list:
                                    df = pd.DataFrame(metrics_list)
                                    st.dataframe(df, use_container_width=True)
                                    
                                    # Create a visualization
                                    metrics_to_plot = [col for col in df.columns if col != "Class"]
                                    
                                    if metrics_to_plot:
                                        st.markdown("#### Class Performance Comparison")
                                        # Melt the DataFrame for easier plotting
                                        melted_df = pd.melt(df, id_vars=["Class"], value_vars=metrics_to_plot, 
                                                          var_name="Metric", value_name="Value")
                                        
                                        fig = px.bar(
                                            melted_df, 
                                            x="Class", 
                                            y="Value", 
                                            color="Metric",
                                            barmode="group",
                                            title="Per-Class Performance Metrics"
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    class_metrics_found = True
                                    break
                        except:
                            continue
                
                if not class_metrics_found:
                    # Generate generic performance over time chart from history if available
                    try:
                        history = run.scan_history()
                        history_df = pd.DataFrame(history)
                        
                        if not history_df.empty:
                            metric_columns = []
                            for col in history_df.columns:
                                if not col.startswith('_') and pd.api.types.is_numeric_dtype(history_df[col]):
                                    if any(m in col.lower() for m in ['accuracy', 'loss', 'precision', 'recall', 'f1', 'auc']):
                                        metric_columns.append(col)
                            
                            if metric_columns:
                                st.markdown("#### Performance Over Time")
                                fig = plot_metrics_history(history_df, metric_columns)
                                st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass
                    
            elif selected_model_type == "Regression":
                # Regression performance metrics and visualizations
                st.markdown("### Regression Performance")
                
                # Metrics section
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.markdown("#### Key Metrics")
                    
                    # Check for and display regression metrics
                    metrics_found = False
                    
                    # Display MAE
                    for k in summary.keys():
                        if 'mae' in k.lower() or 'mean_absolute_error' in k.lower():
                            st.metric("Mean Absolute Error", f"{summary[k]:.4f}")
                            metrics_found = True
                            break
                    
                    # Display MSE
                    for k in summary.keys():
                        if 'mse' in k.lower() or 'mean_squared_error' in k.lower():
                            st.metric("Mean Squared Error", f"{summary[k]:.4f}")
                            metrics_found = True
                            break
                    
                    # Display RMSE
                    for k in summary.keys():
                        if 'rmse' in k.lower() or 'root_mean_squared_error' in k.lower():
                            st.metric("Root Mean Squared Error", f"{summary[k]:.4f}")
                            metrics_found = True
                            break
                    
                    # Display R²
                    for k in summary.keys():
                        if 'r2' in k.lower() or 'r_squared' in k.lower() or 'r^2' in k.lower():
                            st.metric("R² Score", f"{summary[k]:.4f}")
                            metrics_found = True
                            break
                    
                    if not metrics_found:
                        st.info("No regression metrics found in run summary.")
                
                with metrics_col2:
                    # Scatter plot (if predicted vs actual values available)
                    st.markdown("#### Prediction Analysis")
                    
                    # Look for predictions and actual values in summary
                    has_predictions = False
                    for k in summary.keys():
                        if 'prediction' in k.lower() or 'pred' in k.lower() or 'y_pred' in k.lower():
                            try:
                                # Try to find corresponding actual values
                                for k2 in summary.keys():
                                    if 'actual' in k2.lower() or 'true' in k2.lower() or 'y_true' in k2.lower():
                                        # Try to create scatter plot
                                        try:
                                            import json
                                            preds = summary[k]
                                            actuals = summary[k2]
                                            
                                            if isinstance(preds, str):
                                                preds = json.loads(preds)
                                            if isinstance(actuals, str):
                                                actuals = json.loads(actuals)
                                                
                                            if len(preds) > 0 and len(actuals) > 0:
                                                df = pd.DataFrame({
                                                    'Predicted': preds[:min(len(preds), len(actuals))],
                                                    'Actual': actuals[:min(len(preds), len(actuals))]
                                                })
                                                
                                                fig = px.scatter(
                                                    df, x='Actual', y='Predicted',
                                                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                                    title="Predicted vs Actual Values"
                                                )
                                                
                                                # Add diagonal line
                                                fig.add_trace(
                                                    go.Scatter(
                                                        x=[df['Actual'].min(), df['Actual'].max()],
                                                        y=[df['Actual'].min(), df['Actual'].max()],
                                                        mode='lines',
                                                        name='Ideal',
                                                        line=dict(dash='dash', color='gray')
                                                    )
                                                )
                                                
                                                st.plotly_chart(fig, use_container_width=True)
                                                has_predictions = True
                                        except:
                                            pass
                                        
                                        break
                            except:
                                pass
                            
                            break
                    
                    if not has_predictions:
                        st.info("No prediction data found for visualization.")
                
                # Generate error distribution chart from residuals if available
                for k in summary.keys():
                    if 'residual' in k.lower() or 'error' in k.lower():
                        try:
                            errors = summary[k]
                            if isinstance(errors, str):
                                import json
                                errors = json.loads(errors)
                            
                            if isinstance(errors, list) and len(errors) > 0:
                                st.markdown("#### Error Distribution")
                                
                                fig = px.histogram(
                                    errors,
                                    title="Error Distribution",
                                    labels={'x': 'Error', 'y': 'Count'}
                                )
                                
                                fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                                break
                        except:
                            pass
                
                # Generate performance over time chart from history if available
                try:
                    history = run.scan_history()
                    history_df = pd.DataFrame(history)
                    
                    if not history_df.empty:
                        metric_columns = []
                        for col in history_df.columns:
                            if not col.startswith('_') and pd.api.types.is_numeric_dtype(history_df[col]):
                                if any(m in col.lower() for m in ['mae', 'mse', 'rmse', 'loss', 'r2']):
                                    metric_columns.append(col)
                        
                        if metric_columns:
                            st.markdown("#### Performance Over Time")
                            fig = plot_metrics_history(history_df, metric_columns)
                            st.plotly_chart(fig, use_container_width=True)
                except:
                    pass
                    
            else:  # Other model types
                st.markdown("### Model Analysis")
                
                # Create a generic performance view for other model types
                st.markdown("""
                This is a generic analysis view for models that don't fit standard classification or regression patterns.
                Select metrics to visualize from the available data.
                """)
                
                # Get metrics from summary
                metrics = {k: v for k, v in summary.items() 
                         if not k.startswith('_') and isinstance(v, (int, float))}
                
                if not metrics:
                    st.info("No numeric metrics found in run summary.")
                else:
                    # Allow user to select metrics
                    selected_metrics = st.multiselect(
                        "Select metrics to analyze",
                        options=list(metrics.keys()),
                        default=list(metrics.keys())[:min(4, len(metrics))]
                    )
                    
                    if selected_metrics:
                        # Create comparison chart
                        fig = px.bar(
                            x=selected_metrics,
                            y=[metrics[m] for m in selected_metrics],
                            labels={'x': 'Metric', 'y': 'Value'},
                            title="Model Performance Metrics"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Try to find learning curve data
                learning_curve_found = False
                for k in summary.keys():
                    if 'learning_curve' in k.lower() or 'learning' in k.lower() or 'curve' in k.lower():
                        try:
                            curve_data = summary[k]
                            if isinstance(curve_data, str):
                                import json
                                curve_data = json.loads(curve_data)
                                
                            if isinstance(curve_data, dict) and 'train_sizes' in curve_data and 'train_scores' in curve_data and 'test_scores' in curve_data:
                                st.markdown("#### Learning Curve")
                                
                                # Extract data
                                train_sizes = curve_data['train_sizes']
                                train_scores = np.mean(curve_data['train_scores'], axis=1) if isinstance(curve_data['train_scores'][0], list) else curve_data['train_scores']
                                test_scores = np.mean(curve_data['test_scores'], axis=1) if isinstance(curve_data['test_scores'][0], list) else curve_data['test_scores']
                                
                                # Create DataFrame
                                df = pd.DataFrame({
                                    'Training Size': train_sizes,
                                    'Training Score': train_scores,
                                    'Validation Score': test_scores
                                })
                                
                                # Create plot
                                fig = px.line(
                                    df, x='Training Size', 
                                    y=['Training Score', 'Validation Score'],
                                    title="Learning Curve"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                learning_curve_found = True
                                break
                        except:
                            pass
                
                # If no specific visualizations were found, try to visualize history
                if not learning_curve_found:
                    try:
                        history = run.scan_history()
                        history_df = pd.DataFrame(history)
                        
                        if not history_df.empty:
                            # Find columns that might be important for training
                            metric_columns = []
                            for col in history_df.columns:
                                if not col.startswith('_') and pd.api.types.is_numeric_dtype(history_df[col]):
                                    metric_columns.append(col)
                            
                            if metric_columns:
                                st.markdown("#### Training Metrics")
                                # Let user select metrics
                                selected_cols = st.multiselect(
                                    "Select metrics to plot",
                                    options=metric_columns,
                                    default=metric_columns[:min(3, len(metric_columns))]
                                )
                                
                                if selected_cols:
                                    # Create metric history plots
                                    fig = plot_metrics_history(history_df, selected_cols)
                                    st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass
            
            # Add a quantum computing integration section
            st.markdown("### Quantum Computing Integration")
            st.markdown("""
            Enhance your model analysis with quantum computing techniques:
            
            - **Quantum Feature Importance**: Apply quantum algorithms to better understand feature importance
            - **Quantum Optimization**: Explore parameter optimization using quantum annealing or QAOA
            - **Dataset Complexity Analysis**: Use quantum entropy measures to analyze dataset complexity
            """)
            
            if st.button("Go to Quantum Assistant"):
                st.session_state.active_tab = "Quantum AI"
                st.rerun()
        
        # Files tab
        with files_tab:
            st.subheader("Files & Artifacts")
            
            # Show files
            files = run.files()
            
            if not files:
                st.info("No files found for this run")
            else:
                st.markdown("#### Files")
                
                # Create DataFrame of files
                files_data = []
                for file in files:
                    files_data.append({
                        "Name": file.name,
                        "Size (KB)": round(file.size / 1024, 2) if hasattr(file, 'size') else "N/A",
                        "Updated": datetime.fromtimestamp(file.updated_at).strftime('%Y-%m-%d %H:%M:%S') if hasattr(file, 'updated_at') else "N/A"
                    })
                
                files_df = pd.DataFrame(files_data)
                st.dataframe(files_df, use_container_width=True)
                
                # Allow downloading selected file
                if not files_df.empty:
                    selected_file = st.selectbox(
                        "Select file to download",
                        options=files_df["Name"].tolist()
                    )
                    
                    if st.button("Download Selected File"):
                        with st.spinner(f"Downloading {selected_file}..."):
                            try:
                                for file in files:
                                    if file.name == selected_file:
                                        file.download(replace=True)
                                        st.success(f"File {selected_file} downloaded successfully to the current directory")
                                        break
                            except Exception as e:
                                st.error(f"Error downloading file: {str(e)}")
            
            # Show artifacts
            try:
                artifacts = run.logged_artifacts()
                
                if not artifacts:
                    st.info("No artifacts found for this run")
                else:
                    st.markdown("#### Artifacts")
                    
                    # Create DataFrame of artifacts
                    artifacts_data = []
                    for artifact in artifacts:
                        artifacts_data.append({
                            "Name": artifact.name,
                            "Type": artifact.type,
                            "Version": artifact.version,
                            "Size (MB)": round(artifact.size / (1024 * 1024), 2) if hasattr(artifact, 'size') else "N/A",
                            "Created": datetime.fromtimestamp(artifact.created_at).strftime('%Y-%m-%d %H:%M:%S') if hasattr(artifact, 'created_at') else "N/A"
                        })
                    
                    artifacts_df = pd.DataFrame(artifacts_data)
                    st.dataframe(artifacts_df, use_container_width=True)
                    
                    # Button to go to Artifact Manager for more detailed handling
                    if st.button("Go to Artifact Manager"):
                        st.session_state.active_tab = "Artifacts"
                        st.rerun()
            except Exception as e:
                st.info(f"Artifact information not available: {str(e)}")
        
    except Exception as e:
        st.error(f"Error loading run details: {str(e)}")
