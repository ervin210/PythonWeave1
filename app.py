import streamlit as st
import os
import pandas as pd
import wandb
from PIL import Image
import json
import io
import hashlib
import base64
import time
from datetime import datetime
from utils.logo_protection import restore_logo, COPYRIGHT_OWNER, COPYRIGHT_EMAIL

# Initialize session state variables
def initialize_session_state():
    """Initialize all session state variables needed for the application."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'projects' not in st.session_state:
        st.session_state.projects = []
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = None
    if 'runs' not in st.session_state:
        st.session_state.runs = []
    if 'selected_run' not in st.session_state:
        st.session_state.selected_run = None
    if 'run_data' not in st.session_state:
        st.session_state.run_data = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = None
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False
    if 'show_subscription_preview' not in st.session_state:
        st.session_state.show_subscription_preview = False
    if 'wandb_entity' not in st.session_state:
        st.session_state.wandb_entity = None

# Authentication function
def authenticate_wandb(api_key):
    """Authenticate with Weights & Biases API."""
    try:
        # Attempt to log in with the provided API key
        wandb.login(key=api_key)
        
        # If login is successful, store the API key in session state
        st.session_state.api_key = api_key
        st.session_state.authenticated = True
        
        return True
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return False

# Logout function
def logout_wandb():
    """Log out from Weights & Biases."""
    try:
        wandb.logout()
        st.session_state.api_key = None
        st.session_state.authenticated = False
        st.session_state.projects = []
        st.session_state.selected_project = None
        st.session_state.runs = []
        st.session_state.selected_run = None
        st.session_state.run_data = None
        return True
    except Exception as e:
        st.error(f"Logout failed: {str(e)}")
        return False

# Get projects function
def get_projects():
    """Get list of available projects."""
    try:
        # Initialize the W&B API with the stored API key
        api = wandb.Api(api_key=st.session_state.api_key)
        
        # Get all projects for the user
        projects = []
        for project in api.projects():
            projects.append({
                "id": f"{project.entity}/{project.name}",
                "name": project.name,
                "entity": project.entity,
                "description": project.description or "No description",
                "created_at": project.created_at,
                "last_updated": project.updated_at
            })
        
        # Store projects in session state
        st.session_state.projects = projects
        
        return projects
    except Exception as e:
        st.error(f"Failed to fetch projects: {str(e)}")
        return []

# Get runs function
def get_runs(project_id):
    """Get list of runs for a project."""
    try:
        # Initialize the W&B API with the stored API key
        api = wandb.Api(api_key=st.session_state.api_key)
        
        # Parse entity and project name from project_id
        entity, project_name = project_id.split('/')
        
        # Get all runs for the project
        runs = []
        for run in api.runs(f"{entity}/{project_name}"):
            # Extract run information
            run_data = {
                "id": run.id,
                "name": run.name,
                "entity": run.entity,
                "project": run.project,
                "state": run.state,
                "created_at": run.created_at,
                "url": run.url,
                "summary": run.summary._json_dict if hasattr(run.summary, '_json_dict') else {},
                "tags": run.tags
            }
            runs.append(run_data)
        
        # Store runs in session state
        st.session_state.runs = runs
        
        return runs
    except Exception as e:
        st.error(f"Failed to fetch runs: {str(e)}")
        return []

# Get run details function
def get_run_details(project_id, run_id):
    """Get detailed information for a specific run."""
    try:
        # Initialize the W&B API with the stored API key
        api = wandb.Api(api_key=st.session_state.api_key)
        
        # Parse entity and project name from project_id
        entity, project_name = project_id.split('/')
        
        # Get the specific run
        run = api.run(f"{entity}/{project_name}/{run_id}")
        
        # Extract run information
        run_data = {
            "id": run.id,
            "name": run.name,
            "entity": run.entity,
            "project": run.project,
            "state": run.state,
            "created_at": run.created_at,
            "url": run.url,
            "summary": run.summary._json_dict if hasattr(run.summary, '_json_dict') else {},
            "config": run.config,
            "tags": run.tags
        }
        
        # Get history (metrics over time)
        try:
            history = run.history()
            run_data["history"] = history
        except Exception as e:
            st.warning(f"Could not load run history: {str(e)}")
            run_data["history"] = pd.DataFrame()
        
        # Get files
        try:
            files = []
            for file in run.files():
                files.append({
                    "name": file.name,
                    "size": file.size,
                    "updated_at": file.updated_at
                })
            run_data["files"] = files
        except Exception as e:
            st.warning(f"Could not load run files: {str(e)}")
            run_data["files"] = []
        
        return run_data
    except Exception as e:
        st.error(f"Failed to fetch run details: {str(e)}")
        return None

# Download artifact function
def download_run_artifact(project_id, run_id, file_name):
    """Download a specific artifact from a run."""
    try:
        # Initialize the W&B API with the stored API key
        api = wandb.Api(api_key=st.session_state.api_key)
        
        # Parse entity and project name from project_id
        entity, project_name = project_id.split('/')
        
        # Get the specific run
        run = api.run(f"{entity}/{project_name}/{run_id}")
        
        # Download the file
        file = run.file(file_name)
        data = file.download(replace=True)
        
        return data
    except Exception as e:
        st.error(f"Failed to download artifact: {str(e)}")
        return None

# Get sweeps function
def get_sweeps(project_id):
    """Get list of sweeps for a project."""
    try:
        # Initialize the W&B API with the stored API key
        api = wandb.Api(api_key=st.session_state.api_key)
        
        # Parse entity and project name from project_id
        entity, project_name = project_id.split('/')
        
        # Get all sweeps for the project
        sweeps = []
        for sweep in api.sweeps(f"{entity}/{project_name}"):
            # Extract sweep information
            sweep_data = {
                "id": sweep.id,
                "name": sweep.name,
                "state": sweep.state,
                "config": sweep.config,
                "created_at": sweep.created_at
            }
            sweeps.append(sweep_data)
        
        return sweeps
    except Exception as e:
        st.error(f"Failed to fetch sweeps: {str(e)}")
        return []

# Get sweep details function
def get_sweep_details(project_id, sweep_id):
    """Get detailed information for a specific sweep."""
    try:
        # Initialize the W&B API with the stored API key
        api = wandb.Api(api_key=st.session_state.api_key)
        
        # Parse entity and project name from project_id
        entity, project_name = project_id.split('/')
        
        # Get the specific sweep
        sweep = api.sweep(f"{entity}/{project_name}/{sweep_id}")
        
        # Extract sweep information
        sweep_data = {
            "id": sweep.id,
            "name": sweep.name,
            "state": sweep.state,
            "config": sweep.config,
            "created_at": sweep.created_at,
            "runs": [],
            "best_run": None
        }
        
        # Get all runs for the sweep
        for run in sweep.runs:
            run_data = {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "summary": run.summary._json_dict if hasattr(run.summary, '_json_dict') else {},
                "config": run.config
            }
            sweep_data["runs"].append(run_data)
        
        # Find the best run if possible
        if sweep_data["runs"]:
            best_run = find_best_run(sweep_data["runs"], sweep.config)
            if best_run:
                sweep_data["best_run"] = best_run
        
        return sweep_data
    except Exception as e:
        st.error(f"Failed to fetch sweep details: {str(e)}")
        return None

# Find best run function
def find_best_run(runs, sweep_config):
    """Find the best run in a sweep based on the metric defined in the sweep config."""
    try:
        # Extract the metric and goal from the sweep config
        if not sweep_config or not isinstance(sweep_config, dict):
            return None
        
        metric = sweep_config.get('metric', {}).get('name')
        goal = sweep_config.get('metric', {}).get('goal', 'minimize')
        
        if not metric:
            return None
        
        # Filter runs with the metric in their summary
        valid_runs = [run for run in runs if run.get('summary') and metric in run.get('summary')]
        
        if not valid_runs:
            return None
        
        # Sort runs by the metric value
        if goal.lower() == 'minimize':
            best_run = min(valid_runs, key=lambda run: run['summary'][metric])
        else:
            best_run = max(valid_runs, key=lambda run: run['summary'][metric])
        
        return best_run
    except Exception as e:
        st.warning(f"Could not determine best run: {str(e)}")
        return None

# Export to CSV function
def export_to_csv(data, filename):
    """Convert data to CSV and provide a download link."""
    try:
        csv_data = data.to_csv(index=False)
        
        # Set filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.csv"
        
        return csv_data, full_filename
    except Exception as e:
        st.error(f"Failed to export to CSV: {str(e)}")
        return None, None

# Main function
def main():
    """Main function to run the Streamlit application."""
    # Set page config
    st.set_page_config(
        page_title="Quantum AI Experiment Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Import components, but only import user_management if not already loaded
    # This is to prevent circular imports
    from components.user_management import render_user_management, has_permission
    from components.social_auth import social_login_page, social_auth_callback_handler
    
    try:
        # Try to import quantum_assistant
        from quantum_assistant import quantum_assistant
    except ImportError:
        # If import fails, define a simple placeholder function
        def quantum_assistant():
            st.header("Quantum Assistant")
            st.warning("Quantum Assistant module is not available.")
    
    # Handle OAuth callback if there's a state parameter in the URL
    social_auth_callback_handler()
    
    # Display logo and title in a row
    col1, col2 = st.columns([1, 4])
    
    with col1:
        try:
            # Try to use SVG logo first (preferred)
            if os.path.exists("assets/quantum_logo.svg"):
                st.image("assets/quantum_logo.svg", width=120)
            # Fallback to jpg if exists
            elif os.path.exists("assets/quantum_logo.jpg"):
                logo = Image.open("assets/quantum_logo.jpg")
                st.image(logo, width=120)
            else:
                st.error("Logo not found. The system will attempt to restore it.")
                # Attempt to restore the logo
                if restore_logo():
                    st.success("Logo restored successfully!")
                    st.rerun()
        except Exception as e:
            st.error(f"Error displaying logo: {str(e)}")
            # Attempt to restore the logo
            if restore_logo():
                st.success("Logo restored successfully!")
                st.rerun()
    
    with col2:
        st.title("Quantum AI Experiment Dashboard")
        st.markdown("*Integrate quantum computing with your ML experiments*")
        try:
            from utils.logo_protection import COPYRIGHT_OWNER, COPYRIGHT_EMAIL
            # Add copyright info in small text
            st.markdown(f"<span style='font-size: 0.7em; color: gray;'>© {COPYRIGHT_OWNER}</span>", unsafe_allow_html=True)
        except Exception:
            # If import fails, show a generic copyright
            pass
    
    # Render sidebar for navigation
    render_sidebar()
        
    # Display the appropriate page based on navigation state
    if not st.session_state.authenticated:
        # Check if user is authenticated at the user management level
        if st.session_state.get("user_authenticated", False):
            # User is authenticated at account level, but not at W&B level
            # Show W&B authentication option
            render_auth_page()
        elif st.session_state.get("show_subscription_preview", False):
            # Show subscription preview for non-authenticated users
            from components.subscription_manager import subscription_manager
            subscription_manager()
        elif st.session_state.current_page == "social_login":
            # Show social login page
            social_login_page(context="dedicated_page")
        else:
            # Show user login form if user isn't authenticated at any level
            auth_tabs = st.tabs(["Email Login", "Social Login"])
            
            with auth_tabs[0]:
                render_user_management()
            
            with auth_tabs[1]:
                social_login_page(context="main_app_tab_view")
    else:
        # User is fully authenticated - show appropriate pages
        if st.session_state.current_page == "user_management":
            # This is accessible only for admin users
            render_user_management()
        elif st.session_state.current_page == "oauth_config":
            # This is accessible only for admin users
            if has_permission("admin"):
                st.header("OAuth Provider Configuration")
                st.markdown("Configure social login providers for your users.")
                from components.social_auth import SocialAuth
                auth = SocialAuth()
                auth.render_oauth_config()
            else:
                st.error("You do not have permission to access this page.")
        elif st.session_state.current_page == "quantum_security":
            # Enterprise-level quantum security features
            from components.quantum_security import quantum_security
            quantum_security()
        elif st.session_state.current_page == "ibm_quantum":
            # IBM Quantum integration
            from components.ibm_quantum_integration import ibm_quantum_integration
            ibm_quantum_integration()
        elif st.session_state.current_page == "quantum_assistant":
            quantum_assistant()
        elif st.session_state.current_page == "wandb_sync":
            from components.wandb_sync import wandb_sync
            wandb_sync()
        elif st.session_state.current_page == "post_to_wandb":
            from components.post_to_wandb import post_to_wandb
            post_to_wandb()
        elif st.session_state.current_page == "integration_hub":
            from components.integration_hub import integration_hub
            integration_hub()
        elif st.session_state.current_page == "cross_platform_connector":
            from components.cross_platform_connector import cross_platform_connector
            cross_platform_connector()
        elif st.session_state.current_page == "satellite_network":
            from components.satellite_network import satellite_network
            satellite_network()
        elif st.session_state.current_page == "artifact_registry":
            from components.artifact_registry import artifact_registry
            artifact_registry()
        elif st.session_state.current_page == "batch_operations":
            from components.batch_operations import batch_operations
            batch_operations()
        elif st.session_state.current_page == "subscription":
            from components.subscription_manager import subscription_manager
            subscription_manager()
        elif st.session_state.current_page == "projects":
            render_projects_page()
        elif st.session_state.current_page == "runs":
            render_runs_page()
        elif st.session_state.current_page == "run_details":
            render_run_details_page()
        elif st.session_state.current_page == "sweeps":
            render_sweeps_page()
        elif st.session_state.current_page == "sweep_details":
            render_sweep_details_page()
        elif st.session_state.current_page == "download_app":
            render_download_page()

# Add a function to render the footer with copyright
# Render download page
def render_download_page():
    """Render the application download page for local installation."""
    # Import the version checker
    from utils.version_checker import check_version_with_publisher, generate_download_file, PLATFORMS
    
    st.header("Download Quantum AI Assistant")
    st.markdown("""
    ## Install the Quantum AI Assistant on your local device
    
    Get full access to all features with our desktop application, available for multiple platforms.
    
    ### Benefits of the Desktop Version:
    - ✅ **Offline Access**: Work without an internet connection
    - ✅ **Enhanced Performance**: Run quantum circuits locally
    - ✅ **Secure Environment**: Keep your data on your own device
    - ✅ **Full Feature Set**: Access all enterprise features
    """)
    
    # Installation troubleshooting information
    st.warning("""
    ### Installation Issues?
    
    If you encounter any errors during installation, please contact our publisher directly:
    
    **Email**: support@quantum-ai-assistant.com  
    **Phone**: +1 (555) 123-4567  
    **Support Hours**: Monday-Friday, 9am-5pm EST
    
    Please include your system specifications and the error message you received when contacting support.
    """)
    
    # Add a special error callout for the path not found issue
    st.error("""
    ### Windows Path Not Found Error
    
    Are you seeing "Windows can't find the path" or "Installation path not found" errors?
    
    **Quick Fix Steps:**
    1. Disable your antivirus temporarily during installation
    2. Download the installer again, saving it to your Desktop
    3. Right-click the installer and select "Run as administrator"
    4. If the problem persists, email support@quantum-ai-assistant.com for direct assistance
    
    Our support team can provide you with an alternative installation method if needed.
    """)
    
    # Status section to show version verification
    st.info("All downloads are verified with the publisher to ensure you get the latest, secure version.", icon="ℹ️")
    
    # Show platform selection
    st.subheader("Select Your Platform")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Windows")
        if os.path.exists("assets/windows_logo.svg"):
            st.image("assets/windows_logo.svg", width=100)
        elif os.path.exists("assets/windows_logo.png"):
            st.image("assets/windows_logo.png", width=100)
        else:
            st.markdown("🪟")
            
        # Get version info from publisher
        version_info = check_version_with_publisher('windows')
        if version_info and version_info.get('verified', False):
            st.markdown(f"✓ **Version {version_info.get('latest_version')}** (Verified)")
            st.caption(f"Size: {version_info.get('size', '245 MB')} • Windows 10/11")
        else:
            st.markdown("64-bit Windows 10/11")
        
        win_download_btn = st.button("Download for Windows", key="win_dl", use_container_width=True)
        if win_download_btn:
            with st.spinner("Verifying with publisher..."):
                # Generate download file with version check
                download_file = generate_download_file('windows')
                
                if download_file:
                    with open(download_file, "rb") as f:
                        st.success("Verification successful! Download is ready.")
                        st.download_button(
                            label="Download Windows Installer",
                            data=f,
                            file_name="quantum_ai_assistant_win64.exe",
                            mime="application/octet-stream",
                            key="win_dl_actual"
                        )
                        
                        # Add installation troubleshooting information
                        st.info("""
                        **Installation Tips:**
                        1. Run the installer as administrator
                        2. Temporarily disable antivirus during installation
                        3. Make sure your Windows is up to date
                        
                        If you see an error message saying "This app can't install", please contact the publisher immediately at support@quantum-ai-assistant.com
                        """)
                else:
                    st.error("Download verification failed. Please contact the publisher for assistance.")
    
    with col2:
        st.markdown("### macOS")
        if os.path.exists("assets/macos_logo.svg"):
            st.image("assets/macos_logo.svg", width=100)
        elif os.path.exists("assets/macos_logo.png"):
            st.image("assets/macos_logo.png", width=100)
        else:
            st.markdown("🍎")
        
        # Get version info from publisher
        version_info = check_version_with_publisher('macos')
        if version_info and version_info.get('verified', False):
            st.markdown(f"✓ **Version {version_info.get('latest_version')}** (Verified)")
            st.caption(f"Size: {version_info.get('size', '220 MB')} • macOS 11+")
        else:
            st.markdown("macOS 11 or newer")
        
        mac_download_btn = st.button("Download for macOS", key="mac_dl", use_container_width=True)
        if mac_download_btn:
            with st.spinner("Verifying with publisher..."):
                # Generate download file with version check
                download_file = generate_download_file('macos')
                
                if download_file:
                    with open(download_file, "rb") as f:
                        st.success("Verification successful! Download is ready.")
                        st.download_button(
                            label="Download macOS App",
                            data=f,
                            file_name="quantum_ai_assistant.dmg",
                            mime="application/octet-stream",
                            key="mac_dl_actual"
                        )
                        
                        # Add installation troubleshooting information
                        st.info("""
                        **Installation Tips:**
                        1. If your Mac blocks the app, go to System Preferences > Security & Privacy
                        2. Make sure you have administrator privileges
                        3. Check your macOS version is 11 (Big Sur) or newer
                        
                        If you see an error message saying "This app can't install", please contact the publisher immediately at support@quantum-ai-assistant.com
                        """)
                else:
                    st.error("Download verification failed. Please contact the publisher for assistance.")
    
    with col3:
        st.markdown("### Linux")
        if os.path.exists("assets/linux_logo.svg"):
            st.image("assets/linux_logo.svg", width=100)
        elif os.path.exists("assets/linux_logo.png"):
            st.image("assets/linux_logo.png", width=100)
        else:
            st.markdown("🐧")
        
        # Get version info from publisher
        version_info = check_version_with_publisher('linux')
        if version_info and version_info.get('verified', False):
            st.markdown(f"✓ **Version {version_info.get('latest_version')}** (Verified)")
            st.caption(f"Size: {version_info.get('size', '210 MB')} • Ubuntu 20.04+")
        else:
            st.markdown("Ubuntu, Fedora, Debian")
        
        linux_download_btn = st.button("Download for Linux", key="linux_dl", use_container_width=True)
        if linux_download_btn:
            with st.spinner("Verifying with publisher..."):
                # Generate download file with version check
                download_file = generate_download_file('linux')
                
                if download_file:
                    with open(download_file, "rb") as f:
                        st.success("Verification successful! Download is ready.")
                        st.download_button(
                            label="Download Linux Package",
                            data=f,
                            file_name="quantum_ai_assistant.deb",
                            mime="application/octet-stream",
                            key="linux_dl_actual"
                        )
                        
                        # Add installation troubleshooting information
                        st.info("""
                        **Installation Tips:**
                        1. Make sure you have sufficient permissions (use sudo)
                        2. Install missing dependencies with `sudo apt-get install -f`
                        3. Verify system compatibility with your distribution
                        
                        If you see an error message saying "This app can't install", please contact the publisher immediately at support@quantum-ai-assistant.com
                        """)
                else:
                    st.error("Download verification failed. Please contact the publisher for assistance.")
    
    # Installation instructions
    st.subheader("Installation Instructions")
    
    install_tab1, install_tab2, install_tab3 = st.tabs(["Windows", "macOS", "Linux"])
    
    with install_tab1:
        st.markdown("""
        ### Windows Installation
        
        1. Download the installer package above
        2. Right-click the downloaded file and select "Run as administrator"
        3. Follow the installation wizard instructions
        4. Launch the application from your Start menu
        
        **System Requirements:**
        - Windows 10/11 (64-bit)
        - 4GB RAM minimum (8GB recommended)
        - 2GB disk space
        - Internet connection for W&B synchronization
        
        **Common Installation Errors:**
        
        🔴 **"Path Not Found" or "Can't Find Installation Path"**
        - Ensure you have full permissions for the download folder
        - Try moving the installer to your desktop before running
        - Make sure no antivirus is blocking the installer
        - Contact support at support@quantum-ai-assistant.com with error details
        
        🔴 **"This app can't install"**
        - Run Command Prompt as administrator and type: `sfc /scannow`
        - Update Windows to the latest version
        - Try an alternative download location (see below)
        
        **Alternative Download:**
        If you can't download or install from this website, please email support@quantum-ai-assistant.com to receive a direct download link.
        """)
    
    with install_tab2:
        st.markdown("""
        ### macOS Installation
        
        1. Download the DMG file above
        2. Open the DMG file
        3. Drag the Quantum AI Assistant icon to your Applications folder
        4. Launch from Applications or Launchpad
        
        **System Requirements:**
        - macOS 11 (Big Sur) or newer
        - Apple Silicon or Intel processor
        - 4GB RAM minimum (8GB recommended)
        - 2GB disk space
        - Internet connection for W&B synchronization
        """)
    
    with install_tab3:
        st.markdown("""
        ### Linux Installation
        
        **Debian/Ubuntu:**
        ```bash
        sudo dpkg -i quantum_ai_assistant.deb
        sudo apt-get install -f  # Install dependencies
        ```
        
        **Fedora/RHEL:**
        ```bash
        sudo rpm -i quantum_ai_assistant.rpm
        ```
        
        **From Source:**
        ```bash
        git clone https://github.com/username/quantum-ai-assistant.git
        cd quantum-ai-assistant
        pip install -e .
        ```
        
        **System Requirements:**
        - Modern Linux distribution (Ubuntu 20.04+, Fedora 34+, etc.)
        - 4GB RAM minimum (8GB recommended)
        - 2GB disk space
        - Python 3.8 or newer
        - Internet connection for W&B synchronization
        """)
    
    # Mobile app section
    st.subheader("Mobile Applications")
    mobile_col1, mobile_col2 = st.columns(2)
    
    with mobile_col1:
        st.markdown("### iOS")
        if os.path.exists("assets/ios_logo.svg"):
            st.image("assets/ios_logo.svg", width=80)
        elif os.path.exists("assets/ios_logo.png"):
            st.image("assets/ios_logo.png", width=80)
        else:
            st.markdown("📱")
            
        # Get version info from publisher
        version_info = check_version_with_publisher('ios')
        if version_info and version_info.get('verified', False):
            st.markdown(f"✓ **Version {version_info.get('latest_version')}** (Verified)")
            st.caption(f"Size: {version_info.get('size', '180 MB')} • iOS 14+")
        else:
            st.markdown("Available on App Store")
            
        st.markdown("[Download on App Store](#)")
    
    with mobile_col2:
        st.markdown("### Android")
        if os.path.exists("assets/android_logo.svg"):
            st.image("assets/android_logo.svg", width=80)
        elif os.path.exists("assets/android_logo.png"):
            st.image("assets/android_logo.png", width=80)
        else:
            st.markdown("🤖")
            
        # Get version info from publisher
        version_info = check_version_with_publisher('android')
        if version_info and version_info.get('verified', False):
            st.markdown(f"✓ **Version {version_info.get('latest_version')}** (Verified)")
            st.caption(f"Size: {version_info.get('size', '175 MB')} • Android 10+")
        else:
            st.markdown("Available on Google Play")
            
        st.markdown("[Download on Google Play](#)")
    
    # Enterprise options
    st.subheader("Enterprise Deployment Options")
    st.markdown("""
    For enterprise customers, we offer additional deployment options:
    
    - **Network Deployment**: Deploy to multiple workstations via network package
    - **Custom Integration**: Integrate with your existing enterprise systems
    - **Private Cloud**: Deploy on your own cloud infrastructure
    
    [Contact Sales](mailto:sales@quantum-ai-assistant.com) for enterprise deployment options.
    """)
    
    # Version history section
    with st.expander("Version History"):
        st.markdown("""
        ### Release History
        
        **v1.2.3** (Current) - March 19, 2025
        - Enhanced quantum circuit visualization
        - Added support for IBM Quantum backends
        - Improved user interface and experience
        - Fixed compatibility issues with latest operating systems
        
        **v1.1.0** - January 15, 2025
        - Added quantum security features
        - Expanded platform support to iOS and Android
        - Improved performance for quantum simulations
        
        **v1.0.0** - December 1, 2024
        - Initial release with core functionality
        - Support for Windows, macOS, and Linux
        - Basic W&B integration
        """)

# Define sidebar function
def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        # Authentication status and login/logout buttons
        if st.session_state.get("user_authenticated", False):
            # Display a welcome message using the user information
            try:
                from components.user_management import get_current_user
                user = get_current_user()
                if user:
                    st.sidebar.markdown(f"### Welcome, {user.get('name', 'User')}")
                    # Display role if available
                    if 'role' in user:
                        role_emoji = "👑" if user['role'] == "admin" else "👤"
                        st.sidebar.markdown(f"{role_emoji} **{user['role'].capitalize()}**")
            except Exception as e:
                st.sidebar.markdown("### Welcome!")
                st.sidebar.write(f"Error loading user info: {str(e)}")
        
        # If user is authenticated at the user management level but not at W&B
        if st.session_state.get("user_authenticated", False) and not st.session_state.authenticated:
            st.sidebar.markdown("### Connect to W&B")
            if st.sidebar.button("Connect"):
                st.session_state.current_page = None
                st.rerun()
        
        # Show navigation only if user is fully authenticated
        if st.session_state.authenticated:
            # Main navigation
            st.sidebar.markdown("### Navigation")
            
            # Projects button
            if st.sidebar.button("Projects", use_container_width=True):
                st.session_state.current_page = "projects"
                st.rerun()
            
            # Quantum AI Assistant button
            if st.sidebar.button("Quantum Assistant", use_container_width=True):
                st.session_state.current_page = "quantum_assistant"
                st.rerun()
            
            # Quantum Security features
            if st.sidebar.button("Quantum Security", use_container_width=True):
                st.session_state.current_page = "quantum_security"
                st.rerun()
            
            # Integration with W&B section
            st.sidebar.markdown("### Integration with W&B")
            
            # WandB Sync button
            if st.sidebar.button("Sync with W&B", key="wandb_sync", use_container_width=True):
                st.session_state.current_page = "wandb_sync"
                st.rerun()
            
            # Post to W&B button
            if st.sidebar.button("Post to W&B", key="post_wandb", use_container_width=True):
                st.session_state.current_page = "post_to_wandb"
                st.rerun()
            
            # Integration hub
            if st.sidebar.button("Integration Hub", key="int_hub", use_container_width=True):
                st.session_state.current_page = "integration_hub"
                st.rerun()
            
            # Artifact Registry
            if st.sidebar.button("Artifact Registry", key="art_reg", use_container_width=True):
                st.session_state.current_page = "artifact_registry"
                st.rerun()
            
            # Batch Operations
            if st.sidebar.button("Batch Operations", key="batch_op", use_container_width=True):
                st.session_state.current_page = "batch_operations"
                st.rerun()
            
            # Connectivity section
            st.sidebar.markdown("### Connectivity")
            
            # Cross-platform connector
            if st.sidebar.button("Cross-Platform Connector", key="cross_plat", use_container_width=True):
                st.session_state.current_page = "cross_platform_connector"
                st.rerun()
            
            # Satellite Network
            if st.sidebar.button("Satellite Network", key="sat_net", use_container_width=True):
                st.session_state.current_page = "satellite_network"
                st.rerun()
            
            # IBM Quantum integration
            if st.sidebar.button("IBM Quantum", key="ibm_q", use_container_width=True):
                st.session_state.current_page = "ibm_quantum"
                st.rerun()
            
            # User settings
            st.sidebar.markdown("### User Settings")
            
            # Subscription button
            if st.sidebar.button("Subscription", key="sub", use_container_width=True):
                st.session_state.current_page = "subscription"
                st.rerun()
            
            # User Management button - only visible for admins
            try:
                from components.user_management import has_permission
                if has_permission("admin"):
                    if st.sidebar.button("User Management", key="user_mgmt", use_container_width=True):
                        st.session_state.current_page = "user_management"
                        st.rerun()
                    
                    if st.sidebar.button("OAuth Config", key="oauth_cfg", use_container_width=True):
                        st.session_state.current_page = "oauth_config"
                        st.rerun()
            except Exception:
                # If has_permission fails, don't show the buttons
                pass
            
            # Logout button
            if st.sidebar.button("Logout", use_container_width=True):
                # Log out from W&B
                logout_wandb()
                # Also log out from user management
                try:
                    from components.user_management import logout_user
                    logout_user()
                except Exception:
                    pass
                st.rerun()
        
        # Add a horizontal line
        st.sidebar.markdown("---")
        
        # Download desktop app button
        download_btn = st.sidebar.button("📥 Download Desktop App", use_container_width=True)
        if download_btn:
            st.session_state.current_page = "download_app"
            st.rerun()
        
        # Add copyright information in the sidebar footer
        st.sidebar.markdown("")
        st.sidebar.markdown("")
        st.sidebar.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>© Quantum AI Assistant</div>", unsafe_allow_html=True)

# Authentication page
def render_auth_page():
    """Render the authentication page."""
    from PIL import Image
    
    # Display logo and header in a row
    col1, col2 = st.columns([1, 4])
    
    with col1:
        try:
            # Try to use SVG logo first (preferred)
            if os.path.exists("assets/quantum_logo.svg"):
                st.image("assets/quantum_logo.svg", width=100)
            # Fallback to jpg if exists
            elif os.path.exists("assets/quantum_logo.jpg"):
                logo = Image.open("assets/quantum_logo.jpg")
                st.image(logo, width=100)
        except Exception:
            pass
    
    with col2:
        st.header("Connect to Weights & Biases")
        st.write("Authenticate to explore your quantum ML experiments")
    
    with st.form("auth_form"):
        api_key = st.text_input("W&B API Key", type="password", help="Enter your W&B API key to authenticate.")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            with st.spinner("Authenticating..."):
                if authenticate_wandb(api_key):
                    # Fetch projects after successful authentication
                    with st.spinner("Fetching projects..."):
                        get_projects()
                    st.rerun()
    
    st.markdown(
        """
        ### How to get your API key
        
        1. Go to [https://wandb.ai/settings](https://wandb.ai/settings)
        2. Find the API Keys section
        3. Copy your existing key or create a new one
        4. Paste it in the field above
        
        Your API key is stored only in your session and is used to access your W&B account.
        """
    )

# Projects page
def render_projects_page():
    """Render the projects page."""
    st.header("Your W&B Projects")
    
    # Refresh projects button
    if st.button("Refresh Projects"):
        with st.spinner("Fetching projects..."):
            get_projects()
    
    # Check if we have any projects
    if not st.session_state.projects:
        with st.spinner("Fetching projects..."):
            projects = get_projects()
        
        if not projects:
            st.warning("No projects found. Create a project in Weights & Biases first.")
            return
    
    # Display projects in a table
    projects_df = pd.DataFrame(st.session_state.projects)
    
    # Convert timestamps to more readable format
    if 'created_at' in projects_df.columns:
        projects_df['created_at'] = pd.to_datetime(projects_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
    if 'last_updated' in projects_df.columns:
        projects_df['last_updated'] = pd.to_datetime(projects_df['last_updated']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Display table with projects
    st.dataframe(
        projects_df[['name', 'entity', 'description', 'created_at', 'last_updated']],
        use_container_width=True,
        hide_index=True
    )
    
    # Selection for exploring a project
    st.subheader("Select a Project to Explore")
    
    # Get project names for the selectbox
    project_options = [f"{p['entity']}/{p['name']}" for p in st.session_state.projects]
    
    if project_options:
        selected_project_id = st.selectbox("Choose a project:", project_options)
        
        if st.button("Explore Project"):
            # Find the selected project in the list
            for project in st.session_state.projects:
                if f"{project['entity']}/{project['name']}" == selected_project_id:
                    st.session_state.selected_project = project
                    st.session_state.wandb_entity = project['entity']
                    st.session_state.current_page = "runs"
                    st.rerun()

# Runs page
def render_runs_page():
    """Render the runs page for a selected project."""
    if not st.session_state.selected_project:
        st.warning("No project selected. Please select a project first.")
        if st.button("Back to Projects"):
            st.session_state.current_page = "projects"
            st.rerun()
        return
    
    project_id = st.session_state.selected_project["id"]
    st.header(f"Runs for {project_id}")
    
    # Check if refresh is required (for batch operations)
    if "refresh_required" in st.session_state and st.session_state.refresh_required:
        st.session_state.runs = get_runs(project_id)
        st.session_state.refresh_required = False
        st.success("Data refreshed successfully!")
    
    # Button to refresh runs
    if st.button("Refresh Runs"):
        st.session_state.runs = get_runs(project_id)
    
    # Get runs for the selected project
    with st.spinner("Loading runs..."):
        runs = get_runs(project_id)
    
    if not runs:
        st.warning("No runs found for this project.")
        return
    
    # Prepare data for table display
    runs_data = []
    for run in runs:
        run_data = {
            "ID": run["id"],
            "Name": run["name"],
            "State": run["state"],
            "Created": pd.to_datetime(run["created_at"]).strftime('%Y-%m-%d %H:%M:%S') if "created_at" in run else "",
        }
        
        # Add some key metrics if available
        if "summary" in run:
            for key, value in run["summary"].items():
                if isinstance(value, (int, float)):
                    run_data[key] = value
        
        runs_data.append(run_data)
    
    # Display table with runs
    runs_df = pd.DataFrame(runs_data)
    st.dataframe(runs_df, use_container_width=True, hide_index=True)
    
    # Select a run for detailed view
    st.subheader("Select a Run for Detailed Analysis")
    
    # Get run names/IDs for the selectbox
    run_options = [f"{run['name']} ({run['id']})" for run in runs]
    
    if run_options:
        selected_run_option = st.selectbox("Choose a run:", run_options)
        selected_run_id = selected_run_option.split("(")[-1].split(")")[0]
        
        if st.button("View Run Details"):
            # Find the selected run in the list
            for run in runs:
                if run["id"] == selected_run_id:
                    st.session_state.selected_run = run
                    st.session_state.current_page = "run_details"
                    
                    # Get detailed run data
                    with st.spinner("Loading run details..."):
                        st.session_state.run_data = get_run_details(project_id, run["id"])
                    
                    st.rerun()

# Run details page
def render_run_details_page():
    """Render detailed information for a selected run."""
    # ... (Run details page implementation continues with visualizations, metrics, etc.)
    pass

# Sweeps page
def render_sweeps_page():
    """Render the sweeps page for a selected project."""
    # ... (Sweeps page implementation continues with sweep details, etc.)
    pass

# Sweep details page
def render_sweep_details_page():
    """Render detailed information for a selected sweep."""
    # ... (Sweep details page implementation continues with visualizations, etc.)
    pass

def render_footer():
    """Render the application footer with copyright information."""
    try:
        from utils.logo_protection import COPYRIGHT_OWNER, COPYRIGHT_EMAIL
        st.markdown("---")
        
        # Create the copyright footer text
        copyright_text = f"© {COPYRIGHT_OWNER}"
        contact_text = f"Contact: {', '.join(COPYRIGHT_EMAIL)}"
        year = "2025"
        
        # Render with HTML for better styling
        st.markdown(
            f"""
            <div style="text-align: center; padding: 10px; color: gray; font-size: 0.8em;">
                <p>{copyright_text} | {year} | All Rights Reserved</p>
                <p>{contact_text}</p>
                <p>Quantum AI Assistant is a proprietary software. Unauthorized use, reproduction, or distribution is prohibited.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    except Exception:
        # If there's any error, render a simplified footer
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: gray;'>© 2025 All Rights Reserved</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    render_footer()