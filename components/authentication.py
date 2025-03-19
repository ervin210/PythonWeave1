import streamlit as st
import wandb
import os

def authenticate_wandb():
    """
    Handle authentication with Weights & Biases API
    """
    st.header("Connect to Weights & Biases")
    
    # API Key authentication
    auth_method = st.radio(
        "Authentication Method",
        ["API Key", "Environment Variable"]
    )
    
    if auth_method == "API Key":
        api_key = st.text_input("Enter your W&B API Key", type="password")
        
        if st.button("Connect"):
            if not api_key:
                st.error("Please enter a valid API key")
                return
                
            try:
                # Initialize the W&B API with the provided key
                api = wandb.Api(api_key=api_key)
                # Test authentication by making a simple API call
                _ = api.viewer()
                
                # Store API instance in session state
                st.session_state.api = api
                st.session_state.authenticated = True
                st.success("Successfully authenticated with W&B!")
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {str(e)}")
                
    else:  # Environment Variable
        api_key = os.getenv("WANDB_API_KEY")
        if st.button("Connect using WANDB_API_KEY environment variable"):
            if not api_key:
                st.error("WANDB_API_KEY environment variable not found")
                return
                
            try:
                # Initialize the W&B API with the environment variable
                api = wandb.Api()
                # Test authentication by making a simple API call
                _ = api.viewer()
                
                # Store API instance in session state
                st.session_state.api = api
                st.session_state.authenticated = True
                st.success("Successfully authenticated with W&B!")
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {str(e)}")
    
    st.markdown("""
    ### About W&B API Keys
    
    Your API key grants access to your W&B account. To get your API key:
    1. Log in to [wandb.ai](https://wandb.ai)
    2. Click on your profile icon in the top right corner
    3. Select "Settings"
    4. Find your API key under the "API keys" section
    
    Your key will never be stored on our servers.
    """)
