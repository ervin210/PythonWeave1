import streamlit as st
import time
import threading
from quantum_ai_assistant import __version__

# Try to import the auto-updater (it might not be available in development environments)
try:
    from quantum_ai_assistant.auto_updater import (
        check_for_updates_now,
        download_and_install_update,
        start_background_update_checks
    )
    UPDATER_AVAILABLE = True
except ImportError:
    UPDATER_AVAILABLE = False

def check_for_update_worker():
    """Check for updates in a background thread to avoid blocking the UI"""
    if not UPDATER_AVAILABLE:
        return False, None, None, None
    
    # Mark check as in progress
    st.session_state.update_check_in_progress = True
    
    # Check for updates
    update_available, latest_version, update_url, update_notes = check_for_updates_now()
    
    # Store results in session state
    st.session_state.update_available = update_available
    st.session_state.latest_version = latest_version
    st.session_state.update_url = update_url
    st.session_state.update_notes = update_notes
    st.session_state.update_check_in_progress = False
    st.session_state.last_update_check = time.time()
    
    return update_available, latest_version, update_url, update_notes

def install_update_worker(callback_key):
    """Download and install update in a background thread"""
    if not UPDATER_AVAILABLE:
        st.session_state[callback_key] = "Auto-updater not available."
        return False
    
    # Mark installation as in progress
    st.session_state.update_install_in_progress = True
    
    # Define callback to update UI
    def update_callback(message):
        st.session_state[callback_key] = message
    
    # Start the installation
    success = download_and_install_update(update_callback)
    
    # Mark installation as complete
    st.session_state.update_install_in_progress = False
    
    if success:
        st.session_state.update_installed = True
    
    return success

def initialize_update_state():
    """Initialize the session state variables for updates"""
    if 'update_check_in_progress' not in st.session_state:
        st.session_state.update_check_in_progress = False
    
    if 'update_install_in_progress' not in st.session_state:
        st.session_state.update_install_in_progress = False
    
    if 'update_available' not in st.session_state:
        st.session_state.update_available = False
    
    if 'latest_version' not in st.session_state:
        st.session_state.latest_version = __version__
    
    if 'update_url' not in st.session_state:
        st.session_state.update_url = None
    
    if 'update_notes' not in st.session_state:
        st.session_state.update_notes = None
    
    if 'last_update_check' not in st.session_state:
        st.session_state.last_update_check = 0
    
    if 'update_status_message' not in st.session_state:
        st.session_state.update_status_message = ""
    
    if 'update_installed' not in st.session_state:
        st.session_state.update_installed = False

def render_update_ui():
    """Render the update UI in the sidebar"""
    initialize_update_state()
    
    with st.sidebar.expander("Updates", expanded=st.session_state.update_available):
        st.write(f"Current version: v{__version__}")
        
        if st.session_state.update_check_in_progress:
            st.info("Checking for updates...")
        
        if st.session_state.update_available:
            st.success(f"Update available: v{st.session_state.latest_version}")
            
            if st.session_state.update_notes:
                with st.expander("What's new"):
                    st.markdown(st.session_state.update_notes)
            
            if st.session_state.update_install_in_progress:
                st.info(st.session_state.update_status_message)
                st.progress(0.5)  # Indeterminate progress
            elif st.session_state.update_installed:
                st.success("Update installed! Please restart the application.")
            else:
                # Use a timestamp-based unique key for the install button as well
                install_button_key = f"updates_manager_install_btn_{int(time.time())}"
                if st.button("Install Update", key=install_button_key):
                    # Initialize the status message
                    st.session_state.update_status_message = "Preparing to download update..."
                    
                    # Start the installation in a separate thread
                    threading.Thread(
                        target=install_update_worker,
                        args=("update_status_message",),
                        daemon=True
                    ).start()
                    
                    # Rerun to show the progress UI
                    st.rerun()
        else:
            # Check button is disabled during checks
            check_disabled = st.session_state.update_check_in_progress
            
            # Show last check time if available
            if st.session_state.last_update_check > 0:
                last_check = time.strftime(
                    "%Y-%m-%d %H:%M:%S", 
                    time.localtime(st.session_state.last_update_check)
                )
                st.write(f"Last checked: {last_check}")
            
            # Use a timestamp-based unique key to prevent conflicts
            button_key = f"updates_manager_check_btn_{int(time.time())}"
            if st.button("Check for Updates", disabled=check_disabled, key=button_key):
                # Start the check in a separate thread
                threading.Thread(target=check_for_update_worker, daemon=True).start()
                
                # Show a temporary message
                st.info("Checking for updates...")
                
                # Rerun after a short delay to show the progress
                time.sleep(0.5)
                st.rerun()

        if not UPDATER_AVAILABLE:
            st.warning("Auto-updater not available in this environment.")
            st.write("Please download updates manually from our GitHub repository.")

def update_manager():
    """Main update manager component"""
    # Initialize state
    initialize_update_state()
    
    # Render the update UI in the sidebar
    render_update_ui()
    
    # Check for updates on initial load (if we haven't checked recently)
    if (UPDATER_AVAILABLE and 
        not st.session_state.update_check_in_progress and 
        time.time() - st.session_state.last_update_check > 86400):  # 24 hours
        # Start the check in a separate thread
        threading.Thread(target=check_for_update_worker, daemon=True).start()

# If this file is run directly
if __name__ == "__main__":
    st.title("Update Manager Test")
    update_manager()