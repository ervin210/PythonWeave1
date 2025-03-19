"""
Satellite Network Component for managing WiFi, hotspot, and network connections across all devices
"""

import os
import sys
import platform
import time
import json
import traceback
import streamlit as st
from utils.network_connector import (
    get_available_networks, connect_to_network, create_hotspot, stop_hotspot,
    get_connection_info, diagnose_network, get_network_interfaces
)
from utils.key_generator import generate_unique_key

def satellite_network():
    """
    Satellite Network component for managing WiFi connections and hotspots
    """
    st.title("Network Connectivity")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Connection Status", 
        "WiFi Networks", 
        "Hotspot", 
        "Diagnostics"
    ])
    
    with tab1:
        render_connection_status()
        
    with tab2:
        render_wifi_networks()
        
    with tab3:
        render_hotspot_management()
        
    with tab4:
        render_network_diagnostics()

def render_connection_status():
    """Render the connection status section"""
    st.subheader("Connection Status")
    
    # Add a refresh button
    if st.button("Refresh Connection Status", key=generate_unique_key("refresh_connection")):
        st.rerun()
    
    # Show current connection information
    with st.spinner("Getting connection info..."):
        info = get_connection_info()
        
        if info["connected"]:
            st.success(f"Connected to: {info['ssid']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Signal Strength", info["signal_strength"])
            with col2:
                st.metric("IP Address", info["ip_address"])
                
            if info["hotspot_active"]:
                st.info("Hotspot is active and sharing this connection")
        else:
            st.warning("Not connected to any wireless network")
    
    # Show available interfaces
    st.subheader("Network Interfaces")
    with st.spinner("Getting network interfaces..."):
        interfaces = get_network_interfaces()
        
        if interfaces:
            # Create a DataFrame for better display
            import pandas as pd
            
            # Extract relevant fields
            interface_data = []
            for interface in interfaces:
                interface_data.append({
                    "Name": interface.get("name", "Unknown"),
                    "Type": interface.get("type", "Unknown"),
                    "IP Address": interface.get("ip", "Not assigned"),
                    "MAC Address": interface.get("mac", "Unknown")
                })
                
            # Display as a table
            st.dataframe(pd.DataFrame(interface_data))
        else:
            st.info("No network interfaces detected")

def render_wifi_networks():
    """Render the WiFi networks section"""
    st.subheader("Available WiFi Networks")
    
    # Add a refresh button
    if st.button("Scan for Networks", key=generate_unique_key("scan_networks")):
        st.session_state.networks = get_available_networks()
        
    # Initialize networks in session state if not already present
    if "networks" not in st.session_state:
        with st.spinner("Scanning for networks..."):
            st.session_state.networks = get_available_networks()
    
    # Display available networks
    if st.session_state.networks:
        # Create a DataFrame for better display
        import pandas as pd
        
        # Extract relevant fields
        network_data = []
        for network in st.session_state.networks:
            network_data.append({
                "SSID": network.get("ssid", "Unknown"),
                "Signal": network.get("signal", "Unknown"),
                "Security": network.get("security", "Unknown")
            })
            
        # Display as a table
        st.dataframe(pd.DataFrame(network_data))
        
        # Connect to a network
        with st.form("connect_form", clear_on_submit=True):
            st.subheader("Connect to Network")
            
            # Get list of SSIDs
            ssids = [network.get("ssid", "") for network in st.session_state.networks]
            ssids = [ssid for ssid in ssids if ssid]  # Filter out empty SSIDs
            
            selected_ssid = st.selectbox(
                "Select Network", 
                options=ssids,
                key=generate_unique_key("network_select")
            )
            
            password = st.text_input(
                "Password (leave empty for open networks)", 
                type="password",
                key=generate_unique_key("network_password")
            )
            
            submit_button = st.form_submit_button("Connect")
            
            if submit_button:
                with st.spinner(f"Connecting to {selected_ssid}..."):
                    success = connect_to_network(selected_ssid, password)
                    
                    if success:
                        st.success(f"Successfully connected to {selected_ssid}")
                        # Update connection info
                        st.session_state.connection_info = get_connection_info()
                    else:
                        st.error(f"Failed to connect to {selected_ssid}. Please check password and try again.")
    else:
        st.info("No wireless networks detected")

def render_hotspot_management():
    """Render the hotspot management section"""
    st.subheader("WiFi Hotspot")
    
    # Check current hotspot status
    info = get_connection_info()
    hotspot_active = info.get("hotspot_active", False)
    
    if hotspot_active:
        st.success("Hotspot is currently active")
        
        if st.button("Stop Hotspot", key=generate_unique_key("stop_hotspot")):
            with st.spinner("Stopping hotspot..."):
                success = stop_hotspot()
                
                if success:
                    st.success("Hotspot stopped successfully")
                    st.rerun()
                else:
                    st.error("Failed to stop hotspot")
    else:
        st.info("Create a WiFi hotspot to share your connection with other devices")
        
        with st.form("hotspot_form", clear_on_submit=False):
            st.subheader("Create Hotspot")
            
            ssid = st.text_input(
                "Hotspot Name (SSID)",
                value="QuantumAssistant",
                key=generate_unique_key("hotspot_ssid")
            )
            
            password = st.text_input(
                "Password (minimum 8 characters for secure networks)",
                value="quantum2023",
                key=generate_unique_key("hotspot_password")
            )
            
            # Security warning for short passwords
            if password and len(password) < 8:
                st.warning("Password should be at least 8 characters for secure networks")
            
            submit_button = st.form_submit_button("Create Hotspot")
            
            if submit_button:
                with st.spinner(f"Creating hotspot {ssid}..."):
                    success = create_hotspot(ssid, password)
                    
                    if success:
                        st.success(f"Hotspot {ssid} created successfully")
                        st.info(f"Other devices can now connect to '{ssid}' with password '{password}'")
                        st.experimental_rerun()
                    else:
                        system = platform.system().lower()
                        if system == "darwin":  # macOS
                            st.error("Hotspot creation on macOS requires using System Preferences")
                            st.info("Please enable Internet Sharing in System Preferences")
                        elif system == "windows":
                            st.error("Hotspot creation might require administrator privileges")
                            st.info("Try running the application as administrator")
                        else:
                            st.error("Failed to create hotspot")
                            st.info("Your system might not support hotspot creation or might require special permissions")

def render_network_diagnostics():
    """Render network diagnostics section"""
    st.subheader("Network Diagnostics")
    
    if st.button("Run Network Diagnostics", key=generate_unique_key("run_diagnostics")):
        with st.spinner("Running diagnostics..."):
            results = diagnose_network()
            
            # Create expandable sections for each diagnostic category
            with st.expander("Connectivity Status", expanded=True):
                if results["connected"]:
                    st.success("Connected to a network")
                else:
                    st.error("Not connected to a network")
                    
                if results["internet_access"]:
                    st.success("Internet access available")
                else:
                    st.error("No Internet access")
                    
                if results["dns_working"]:
                    st.success("DNS resolution working")
                else:
                    st.error("DNS resolution failed")
            
            with st.expander("Performance Metrics", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Latency", f"{results['latency_ms']} ms")
                    
                with col2:
                    st.metric("Download Speed", results["download_speed"])
                    
                with col3:
                    st.metric("Upload Speed", results["upload_speed"])
            
            if results["errors"]:
                with st.expander("Diagnostic Errors", expanded=True):
                    for error in results["errors"]:
                        st.error(error)
            
            # Add recommendations based on diagnostics
            with st.expander("Recommendations", expanded=True):
                if not results["connected"]:
                    st.info("Please connect to a WiFi network to access the internet")
                    
                elif not results["internet_access"]:
                    st.info("Your device is connected to a network but cannot access the internet.")
                    st.info("Please check your router or access point's internet connection")
                    
                elif not results["dns_working"]:
                    st.info("DNS resolution is not working. Try using alternative DNS servers")
                    st.info("Suggested DNS servers: 8.8.8.8 (Google) or 1.1.1.1 (Cloudflare)")
                    
                elif results["latency_ms"] > 100:
                    st.info("Your network latency is high. This may affect real-time applications")
                    st.info("Try moving closer to your WiFi router or using a wired connection")
                    
                else:
                    st.success("Your network appears to be functioning properly!")
                    
def main():
    """Main function to run when module is executed directly"""
    satellite_network()
    
if __name__ == "__main__":
    main()