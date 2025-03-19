"""
Network Connector Utility

This module provides functionality to connect to WiFi networks, create hotspots,
and manage network connections across all supported platforms.
"""

import os
import sys
import platform
import subprocess
import time
import json
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Union

class NetworkConnector:
    """
    Cross-platform network connection manager supporting:
    - WiFi connections
    - Hotspot creation
    - Connection sharing
    - Network diagnostics
    """
    
    def __init__(self):
        """Initialize the network connector"""
        self.system = platform.system().lower()
        self.is_mobile = self._detect_mobile()
        self.current_connection = None
        self.hotspot_active = False
        self.hotspot_info = {}
        
    def _detect_mobile(self) -> bool:
        """Detect if running on a mobile device"""
        # Android detection
        if os.environ.get("ANDROID_ROOT") or os.environ.get("ANDROID_DATA"):
            return True
            
        # iOS detection (approximate)
        if self.system == "darwin" and (
            os.environ.get("SIMULATOR_DEVICE_NAME") or 
            os.environ.get("IPHONE_SIMULATOR_ROOT")
        ):
            return True
            
        # Check for ARM-based hardware which might be mobile/IoT
        machine = platform.machine().lower()
        if "arm" in machine and not "darwin" in self.system:  # Exclude M1/M2 Macs
            # Check for typical mobile hardware files
            mobile_indicators = [
                "/sys/class/power_supply/battery",  # Battery info
                "/dev/input/event*"                 # Touch inputs
            ]
            
            for indicator in mobile_indicators:
                if os.path.exists(indicator):
                    return True
                    
        return False
        
    def get_available_networks(self) -> List[Dict[str, str]]:
        """
        Get list of available WiFi networks
        
        Returns:
            List of dictionaries with network information
        """
        networks = []
        
        if self.system == "windows":
            try:
                # Use netsh to get networks on Windows
                output = subprocess.check_output(
                    ["netsh", "wlan", "show", "networks"], 
                    universal_newlines=True
                )
                
                current_network = {}
                for line in output.split('\n'):
                    line = line.strip()
                    if "SSID" in line and ":" in line:
                        if current_network:
                            networks.append(current_network)
                        current_network = {"ssid": line.split(":", 1)[1].strip()}
                    elif "Signal" in line and ":" in line:
                        current_network["signal"] = line.split(":", 1)[1].strip()
                    elif "Authentication" in line and ":" in line:
                        current_network["security"] = line.split(":", 1)[1].strip()
                
                if current_network:
                    networks.append(current_network)
                    
            except Exception as e:
                print(f"Error getting Windows networks: {e}")
                
        elif self.system == "darwin":  # macOS
            try:
                # Use airport command to get networks on macOS
                airport_path = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
                output = subprocess.check_output(
                    [airport_path, "-s"],
                    universal_newlines=True
                )
                
                lines = output.strip().split('\n')
                if len(lines) > 1:  # Skip header line
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) >= 3:
                            networks.append({
                                "ssid": parts[0],
                                "signal": parts[2],
                                "security": parts[6] if len(parts) > 6 else "Unknown"
                            })
                            
            except Exception as e:
                print(f"Error getting macOS networks: {e}")
                
        elif self.system == "linux":
            try:
                # Try nmcli first (NetworkManager)
                try:
                    output = subprocess.check_output(
                        ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"],
                        universal_newlines=True
                    )
                    
                    for line in output.strip().split('\n'):
                        parts = line.split(':')
                        if len(parts) >= 3:
                            networks.append({
                                "ssid": parts[0],
                                "signal": parts[1] + "%",
                                "security": parts[2] if parts[2] else "Open"
                            })
                except:
                    # Fallback to iwlist
                    output = subprocess.check_output(
                        ["iwlist", "scan"],
                        universal_newlines=True
                    )
                    
                    current_network = {}
                    for line in output.split('\n'):
                        line = line.strip()
                        if "ESSID" in line:
                            if current_network:
                                networks.append(current_network)
                            ssid = line.split(":", 1)[1].strip('\"')
                            current_network = {"ssid": ssid}
                        elif "Quality" in line:
                            parts = line.split("=")
                            if len(parts) > 1:
                                quality = parts[1].split()[0]
                                current_network["signal"] = quality
                        elif "key:" in line:
                            security = "WEP" if "on" in line else "Open"
                            current_network["security"] = security
                    
                    if current_network:
                        networks.append(current_network)
                        
            except Exception as e:
                print(f"Error getting Linux networks: {e}")
                
        elif self.is_mobile:
            # For mobile devices, this would require platform-specific APIs
            # Android would use WifiManager
            # iOS would use NEHotspotNetwork
            networks.append({"ssid": "Mobile APIs not available", "signal": "N/A", "security": "N/A"})
            
        return networks
    
    def connect_to_network(self, ssid: str, password: Optional[str] = None) -> bool:
        """
        Connect to a WiFi network
        
        Args:
            ssid: Network name
            password: Network password (None for open networks)
            
        Returns:
            bool: True if connected successfully, False otherwise
        """
        if self.system == "windows":
            try:
                # Create a temporary profile XML file
                profile_path = os.path.join(tempfile.gettempdir(), "wifi_profile.xml")
                
                # Generate the profile XML
                profile_content = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{ssid}</name>
    <SSIDConfig>
        <SSID>
            <name>{ssid}</name>
        </SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>{'WPA2PSK' if password else 'open'}</authentication>
                <encryption>{'AES' if password else 'none'}</encryption>
                <useOneX>false</useOneX>
            </authEncryption>
            {f'<sharedKey><keyType>passPhrase</keyType><protected>false</protected><keyMaterial>{password}</keyMaterial></sharedKey>' if password else ''}
        </security>
    </MSM>
</WLANProfile>"""

                with open(profile_path, 'w') as f:
                    f.write(profile_content)
                
                # Add the profile
                subprocess.run(["netsh", "wlan", "add", "profile", f"filename={profile_path}"], check=True)
                os.remove(profile_path)  # Clean up
                
                # Connect using the profile
                result = subprocess.run(
                    ["netsh", "wlan", "connect", f"name={ssid}"],
                    capture_output=True,
                    text=True
                )
                
                self.current_connection = ssid if "successfully" in result.stdout.lower() else None
                return self.current_connection is not None
                
            except Exception as e:
                print(f"Error connecting to Windows network: {e}")
                return False
                
        elif self.system == "darwin":  # macOS
            try:
                command = [
                    "networksetup", 
                    "-setairportnetwork", 
                    "en0",  # Usually the WiFi interface
                    ssid
                ]
                
                if password:
                    command.append(password)
                    
                result = subprocess.run(command, capture_output=True, text=True)
                
                self.current_connection = ssid if result.returncode == 0 else None
                return self.current_connection is not None
                
            except Exception as e:
                print(f"Error connecting to macOS network: {e}")
                return False
                
        elif self.system == "linux":
            try:
                # Try nmcli first (NetworkManager)
                try:
                    command = ["nmcli", "device", "wifi", "connect", ssid]
                    
                    if password:
                        command.extend(["password", password])
                        
                    result = subprocess.run(command, capture_output=True, text=True)
                    
                    if "successfully" in result.stdout.lower():
                        self.current_connection = ssid
                        return True
                except:
                    # Fallback to wpa_supplicant
                    # Generate wpa_supplicant.conf
                    conf_path = "/tmp/wpa_supplicant.conf"
                    with open(conf_path, 'w') as f:
                        f.write(f"network={{\n\tssid=\"{ssid}\"\n")
                        if password:
                            f.write(f"\tpsk=\"{password}\"\n")
                        else:
                            f.write("\tkey_mgmt=NONE\n")
                        f.write("}\n")
                    
                    # Restart wpa_supplicant (requires sudo)
                    # Note: This requires appropriate sudo permissions
                    interface = "wlan0"  # This might need to be determined dynamically
                    result = subprocess.run(
                        ["sudo", "wpa_supplicant", "-i", interface, "-c", conf_path, "-B"],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        self.current_connection = ssid
                        return True
                        
                return False
                
            except Exception as e:
                print(f"Error connecting to Linux network: {e}")
                return False
                
        elif self.is_mobile:
            # For mobile devices, this would require platform-specific APIs
            print("Network connection on mobile devices requires system APIs")
            return False
            
        return False
    
    def create_hotspot(self, ssid: str, password: Optional[str] = None) -> bool:
        """
        Create a WiFi hotspot
        
        Args:
            ssid: Hotspot name
            password: Hotspot password (None for open hotspot)
            
        Returns:
            bool: True if hotspot was created successfully, False otherwise
        """
        if self.system == "windows":
            try:
                # Stop any existing hosted network
                subprocess.run(["netsh", "wlan", "stop", "hostednetwork"], capture_output=True)
                
                # Set up the hosted network
                subprocess.run([
                    "netsh", "wlan", "set", "hostednetwork", 
                    f"mode=allow", f"ssid={ssid}", 
                    f"key={password if password else ''}"
                ], check=True)
                
                # Start the hosted network
                result = subprocess.run(
                    ["netsh", "wlan", "start", "hostednetwork"],
                    capture_output=True,
                    text=True
                )
                
                self.hotspot_active = "started" in result.stdout.lower()
                if self.hotspot_active:
                    self.hotspot_info = {"ssid": ssid, "password": password}
                
                return self.hotspot_active
                
            except Exception as e:
                print(f"Error creating Windows hotspot: {e}")
                return False
                
        elif self.system == "darwin":  # macOS
            try:
                # macOS Internet Sharing needs to be enabled in System Preferences
                # This command just checks if it's possible
                output = subprocess.check_output(
                    ["networksetup", "-listallnetworkservices"],
                    universal_newlines=True
                )
                
                print("On macOS, enable Internet Sharing in System Preferences")
                print("Available network services:")
                print(output)
                
                self.hotspot_active = False
                return False  # Cannot be fully automated on macOS
                
            except Exception as e:
                print(f"Error with macOS hotspot: {e}")
                return False
                
        elif self.system == "linux":
            try:
                # Try using nmcli (NetworkManager)
                try:
                    # Create a new hotspot connection
                    command = [
                        "nmcli", "device", "wifi", "hotspot", 
                        "ifname", "wlan0",  # Interface name (might need to be determined)
                        "con-name", f"Hotspot-{ssid}",
                        "ssid", ssid
                    ]
                    
                    if password:
                        command.extend(["password", password])
                        
                    result = subprocess.run(command, capture_output=True, text=True)
                    
                    self.hotspot_active = result.returncode == 0
                    if self.hotspot_active:
                        self.hotspot_info = {"ssid": ssid, "password": password}
                    
                    return self.hotspot_active
                    
                except:
                    # Fallback to create_ap (if installed)
                    try:
                        # Check if create_ap is installed
                        subprocess.check_output(["which", "create_ap"], universal_newlines=True)
                        
                        # Determine network interfaces
                        wifi_interface = "wlan0"  # Might need to be determined
                        internet_interface = "eth0"  # Might need to be determined
                        
                        command = [
                            "sudo", "create_ap", 
                            wifi_interface, internet_interface, 
                            ssid
                        ]
                        
                        if password and len(password) >= 8:
                            command.append(password)
                            
                        result = subprocess.run(command, capture_output=True, text=True)
                        
                        self.hotspot_active = result.returncode == 0
                        if self.hotspot_active:
                            self.hotspot_info = {"ssid": ssid, "password": password}
                        
                        return self.hotspot_active
                        
                    except:
                        print("create_ap not installed - hotspot creation not available")
                        return False
                        
            except Exception as e:
                print(f"Error creating Linux hotspot: {e}")
                return False
                
        elif self.is_mobile:
            # For mobile devices, this would require platform-specific APIs
            print("Hotspot creation on mobile devices requires system APIs")
            return False
            
        return False
    
    def stop_hotspot(self) -> bool:
        """
        Stop a running WiFi hotspot
        
        Returns:
            bool: True if hotspot was stopped successfully, False otherwise
        """
        if not self.hotspot_active:
            return True  # Already stopped
            
        if self.system == "windows":
            try:
                result = subprocess.run(
                    ["netsh", "wlan", "stop", "hostednetwork"],
                    capture_output=True,
                    text=True
                )
                
                self.hotspot_active = not ("stopped" in result.stdout.lower())
                return not self.hotspot_active
                
            except Exception as e:
                print(f"Error stopping Windows hotspot: {e}")
                return False
                
        elif self.system == "linux":
            try:
                # Try using nmcli (NetworkManager)
                try:
                    if "ssid" in self.hotspot_info:
                        result = subprocess.run([
                            "nmcli", "connection", "down", f"Hotspot-{self.hotspot_info['ssid']}"
                        ], capture_output=True, text=True)
                        
                        self.hotspot_active = not (result.returncode == 0)
                        return not self.hotspot_active
                except:
                    # Fallback to create_ap
                    try:
                        result = subprocess.run(
                            ["sudo", "create_ap", "--stop", "wlan0"],
                            capture_output=True,
                            text=True
                        )
                        
                        self.hotspot_active = not (result.returncode == 0)
                        return not self.hotspot_active
                    except:
                        pass
                        
                return False
                
            except Exception as e:
                print(f"Error stopping Linux hotspot: {e}")
                return False
                
        # Other platforms - macOS and mobile
        print("Hotspot management not fully supported on this platform")
        return False
    
    def get_connection_info(self) -> Dict[str, str]:
        """
        Get information about the current network connection
        
        Returns:
            Dictionary with connection details
        """
        info = {
            "connected": False,
            "ssid": "",
            "ip_address": "",
            "signal_strength": "",
            "hotspot_active": self.hotspot_active
        }
        
        if self.system == "windows":
            try:
                # Get current connection
                output = subprocess.check_output(
                    ["netsh", "wlan", "show", "interfaces"], 
                    universal_newlines=True
                )
                
                for line in output.split('\n'):
                    line = line.strip()
                    if "SSID" in line and ":" in line and not "BSSID" in line:
                        info["ssid"] = line.split(":", 1)[1].strip()
                        info["connected"] = True
                    elif "Signal" in line and ":" in line:
                        info["signal_strength"] = line.split(":", 1)[1].strip()
                
                # Get IP address
                if info["connected"]:
                    output = subprocess.check_output(
                        ["ipconfig"], 
                        universal_newlines=True
                    )
                    
                    wireless_section = False
                    for line in output.split('\n'):
                        line = line.strip()
                        if "Wireless" in line:
                            wireless_section = True
                        elif wireless_section and "IPv4 Address" in line and ":" in line:
                            info["ip_address"] = line.split(":", 1)[1].strip()
                            break
                            
            except Exception as e:
                print(f"Error getting Windows connection info: {e}")
                
        elif self.system == "darwin":  # macOS
            try:
                # Get current WiFi info
                airport_path = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
                output = subprocess.check_output([airport_path, "-I"], universal_newlines=True)
                
                for line in output.split('\n'):
                    line = line.strip()
                    if "SSID:" in line:
                        info["ssid"] = line.split(":", 1)[1].strip()
                        info["connected"] = True
                    elif "agrCtlRSSI:" in line:
                        # Convert dBm to percentage
                        rssi = int(line.split(":", 1)[1].strip())
                        percentage = min(100, max(0, 2 * (rssi + 100)))
                        info["signal_strength"] = f"{percentage}%"
                
                # Get IP address
                if info["connected"]:
                    output = subprocess.check_output(
                        ["ipconfig", "getifaddr", "en0"],
                        universal_newlines=True
                    )
                    
                    info["ip_address"] = output.strip()
                    
            except Exception as e:
                print(f"Error getting macOS connection info: {e}")
                
        elif self.system == "linux":
            try:
                # Try nmcli first (NetworkManager)
                try:
                    output = subprocess.check_output(
                        ["nmcli", "-t", "-f", "GENERAL.CONNECTION", "device", "show", "wlan0"],
                        universal_newlines=True
                    )
                    
                    connection = output.strip().split(":", 1)[1] if ":" in output else ""
                    if connection and connection != "--":
                        info["connected"] = True
                        info["ssid"] = connection
                        
                        # Get signal strength
                        output = subprocess.check_output(
                            ["nmcli", "-t", "-f", "GENERAL.DEVICE,WIFI-PROPERTIES.SIGNAL", "device", "show", "wlan0"],
                            universal_newlines=True
                        )
                        
                        for line in output.strip().split('\n'):
                            if "SIGNAL" in line and ":" in line:
                                signal = line.split(":", 1)[1]
                                info["signal_strength"] = f"{signal}%"
                                
                        # Get IP address
                        output = subprocess.check_output(
                            ["ip", "-o", "-4", "addr", "show", "wlan0"],
                            universal_newlines=True
                        )
                        
                        if output:
                            ip_parts = output.strip().split()
                            for i, part in enumerate(ip_parts):
                                if part == "inet":
                                    info["ip_address"] = ip_parts[i+1].split("/")[0]
                except:
                    # Fallback to iwconfig
                    output = subprocess.check_output(["iwconfig", "wlan0"], universal_newlines=True)
                    
                    for line in output.split('\n'):
                        line = line.strip()
                        if 'ESSID:' in line:
                            ssid = line.split('ESSID:', 1)[1].strip('"')
                            if ssid and ssid != "off/any":
                                info["connected"] = True
                                info["ssid"] = ssid
                        elif "Quality" in line and "Signal level" in line:
                            signal_parts = line.split("Signal level=", 1)
                            if len(signal_parts) > 1:
                                signal = signal_parts[1].split()[0]
                                # Convert to percentage based on format
                                if "dBm" in signal:
                                    dbm = int(signal.replace("dBm", ""))
                                    percentage = min(100, max(0, 2 * (dbm + 100)))
                                    info["signal_strength"] = f"{percentage}%"
                                else:
                                    info["signal_strength"] = signal
                    
                    # Get IP address
                    if info["connected"]:
                        output = subprocess.check_output(
                            ["ip", "addr", "show", "wlan0"],
                            universal_newlines=True
                        )
                        
                        for line in output.split('\n'):
                            line = line.strip()
                            if "inet " in line:
                                info["ip_address"] = line.split()[1].split('/')[0]
                                break
                        
            except Exception as e:
                print(f"Error getting Linux connection info: {e}")
                
        elif self.is_mobile:
            # For mobile devices, this would require platform-specific APIs
            info["connected"] = True  # Assume connected
            info["ssid"] = "Mobile Network"
            info["signal_strength"] = "N/A"
            info["ip_address"] = "Mobile APIs not available"
            
        return info
    
    def diagnose_network(self) -> Dict[str, any]:
        """
        Run network diagnostics
        
        Returns:
            Dictionary with diagnostic results
        """
        results = {
            "connected": False,
            "internet_access": False,
            "dns_working": False,
            "latency_ms": 0,
            "download_speed": "0 Mbps",
            "upload_speed": "0 Mbps",
            "errors": []
        }
        
        # Check connection
        connection_info = self.get_connection_info()
        results["connected"] = connection_info["connected"]
        
        if not results["connected"]:
            results["errors"].append("Not connected to any network")
            return results
            
        # Check Internet access by pinging a reliable server
        try:
            ping_target = "8.8.8.8"  # Google DNS
            ping_count = "4" if self.system != "windows" else "4"
            ping_cmd = ["ping", "-c", ping_count, ping_target] if self.system != "windows" else ["ping", "-n", ping_count, ping_target]
            
            output = subprocess.check_output(ping_cmd, universal_newlines=True)
            
            results["internet_access"] = "received" in output.lower() and "0%" not in output
            
            # Extract latency if available
            latency_marker = "time=" if self.system != "windows" else "Average ="
            if latency_marker in output:
                latency_line = [line for line in output.split('\n') if latency_marker in line][0]
                if self.system == "windows":
                    # Windows ping shows "Average = Xms"
                    latency = latency_line.split('=')[1].strip()
                    if "ms" in latency:
                        results["latency_ms"] = int(float(latency.replace("ms", "").strip()))
                else:
                    # Unix ping shows multiple lines with "time=X.Y ms"
                    # We'll use the average
                    latency_values = []
                    for line in output.split('\n'):
                        if "time=" in line:
                            time_part = line.split("time=")[1].split()[0]
                            try:
                                latency_values.append(float(time_part.replace("ms", "")))
                            except:
                                pass
                    
                    if latency_values:
                        results["latency_ms"] = int(sum(latency_values) / len(latency_values))
                
        except Exception as e:
            results["errors"].append(f"Internet connectivity error: {e}")
            
        # Check DNS resolution
        try:
            subprocess.check_output(["nslookup", "google.com"], universal_newlines=True)
            results["dns_working"] = True
        except:
            results["errors"].append("DNS resolution failed")
            
        # Try a simple speed test (very basic)
        # For a real solution, use a speed test library or API
        try:
            # Simple download test using a small file
            start_time = time.time()
            test_url = "https://www.google.com/favicon.ico"  # Small test file
            
            import urllib.request
            with urllib.request.urlopen(test_url, timeout=5) as response:
                data = response.read()
                
            download_time = time.time() - start_time
            download_size = len(data) * 8 / 1000000  # Convert bytes to megabits
            download_speed = download_size / download_time if download_time > 0 else 0
            
            results["download_speed"] = f"{download_speed:.2f} Mbps"
            # Upload speed would require a server to accept uploads
            results["upload_speed"] = "Not tested"
            
        except Exception as e:
            results["errors"].append(f"Speed test error: {e}")
            
        return results
    
    def enable_internet_sharing(self, from_interface: str, to_interface: str) -> bool:
        """
        Enable Internet connection sharing between interfaces
        
        Args:
            from_interface: Source interface with Internet connection
            to_interface: Target interface to share connection to
            
        Returns:
            bool: True if sharing was enabled successfully, False otherwise
        """
        if self.system == "windows":
            try:
                # On Windows, this requires registry changes
                # This is a simplified approach and might need admin rights
                interface_guid = None
                
                # Get interface GUID
                output = subprocess.check_output(
                    ["netsh", "interface", "show", "interface", from_interface],
                    universal_newlines=True
                )
                
                # Search for the GUID in the registry
                # This is a placeholder - actual implementation would require more work
                # and would need to use the Windows Registry API
                
                print(f"Internet sharing on Windows requires manual configuration or specialized APIs")
                return False
                
            except Exception as e:
                print(f"Error enabling Windows Internet sharing: {e}")
                return False
                
        elif self.system == "darwin":  # macOS
            try:
                # macOS Internet Sharing must be enabled in System Preferences
                print("On macOS, enable Internet Sharing in System Preferences")
                return False  # Cannot be fully automated on macOS
                
            except Exception as e:
                print(f"Error with macOS Internet sharing: {e}")
                return False
                
        elif self.system == "linux":
            try:
                # Enable IP forwarding
                with open("/proc/sys/net/ipv4/ip_forward", "w") as f:
                    f.write("1")
                
                # Set up NAT using iptables
                subprocess.run(["sudo", "iptables", "-t", "nat", "-A", "POSTROUTING", "-o", from_interface, "-j", "MASQUERADE"], check=True)
                subprocess.run(["sudo", "iptables", "-A", "FORWARD", "-i", from_interface, "-o", to_interface, "-m", "state", "--state", "RELATED,ESTABLISHED", "-j", "ACCEPT"], check=True)
                subprocess.run(["sudo", "iptables", "-A", "FORWARD", "-i", to_interface, "-o", from_interface, "-j", "ACCEPT"], check=True)
                
                print(f"Internet sharing enabled from {from_interface} to {to_interface}")
                return True
                
            except Exception as e:
                print(f"Error enabling Linux Internet sharing: {e}")
                return False
                
        return False
        
    def get_network_interfaces(self) -> List[Dict[str, str]]:
        """
        Get list of available network interfaces
        
        Returns:
            List of dictionaries with interface information
        """
        interfaces = []
        
        if self.system == "windows":
            try:
                output = subprocess.check_output(["ipconfig", "/all"], universal_newlines=True)
                
                adapter_block = ""
                current_adapter = {}
                
                for line in output.split('\n'):
                    line = line.strip()
                    
                    if not line:
                        if current_adapter and "name" in current_adapter:
                            interfaces.append(current_adapter)
                            current_adapter = {}
                        continue
                        
                    if "adapter" in line.lower() and ":" in line:
                        adapter_name = line.split(":", 1)[0].strip()
                        current_adapter = {"name": adapter_name}
                    elif "physical address" in line.lower() and ":" in line:
                        current_adapter["mac"] = line.split(":", 1)[1].strip()
                    elif "ipv4 address" in line.lower() and ":" in line:
                        current_adapter["ip"] = line.split(":", 1)[1].strip().split("(")[0].strip()
                    elif "default gateway" in line.lower() and ":" in line:
                        gateway = line.split(":", 1)[1].strip()
                        if gateway:
                            current_adapter["gateway"] = gateway
                
                if current_adapter and "name" in current_adapter:
                    interfaces.append(current_adapter)
                    
            except Exception as e:
                print(f"Error getting Windows interfaces: {e}")
                
        elif self.system == "darwin":  # macOS
            try:
                # Get list of interfaces
                output = subprocess.check_output(["networksetup", "-listallhardwareports"], universal_newlines=True)
                
                current_interface = {}
                for line in output.split('\n'):
                    line = line.strip()
                    
                    if not line:
                        if current_interface and "name" in current_interface:
                            interfaces.append(current_interface)
                            current_interface = {}
                        continue
                        
                    if "Hardware Port:" in line:
                        current_interface = {"name": line.split(":", 1)[1].strip()}
                    elif "Device:" in line:
                        current_interface["device"] = line.split(":", 1)[1].strip()
                        
                        # Get IP address for this device
                        try:
                            ip_output = subprocess.check_output(
                                ["ipconfig", "getifaddr", current_interface["device"]],
                                universal_newlines=True
                            )
                            current_interface["ip"] = ip_output.strip()
                        except:
                            pass
                            
                    elif "Ethernet Address:" in line:
                        current_interface["mac"] = line.split(":", 1)[1].strip()
                
                if current_interface and "name" in current_interface:
                    interfaces.append(current_interface)
                    
            except Exception as e:
                print(f"Error getting macOS interfaces: {e}")
                
        elif self.system == "linux":
            try:
                # Try ip command
                output = subprocess.check_output(["ip", "-o", "addr", "show"], universal_newlines=True)
                
                for line in output.split('\n'):
                    if not line:
                        continue
                        
                    parts = line.strip().split()
                    if len(parts) > 3:
                        interface = {
                            "name": parts[1],
                            "type": "loopback" if parts[1] == "lo" else "unknown"
                        }
                        
                        # Determine interface type
                        if interface["name"].startswith("wl"):
                            interface["type"] = "wireless"
                        elif interface["name"].startswith("en") or interface["name"].startswith("eth"):
                            interface["type"] = "ethernet"
                        elif interface["name"].startswith("ww"):
                            interface["type"] = "wwan"
                        
                        # Get IP address
                        for i, part in enumerate(parts):
                            if part == "inet":
                                interface["ip"] = parts[i+1].split("/")[0]
                                break
                        
                        # Get MAC address
                        try:
                            mac_output = subprocess.check_output(
                                ["ip", "link", "show", interface["name"]],
                                universal_newlines=True
                            )
                            
                            for mac_line in mac_output.split('\n'):
                                if "link/ether" in mac_line:
                                    interface["mac"] = mac_line.split()[1]
                        except:
                            pass
                            
                        interfaces.append(interface)
                    
            except Exception as e:
                print(f"Error getting Linux interfaces: {e}")
                
        return interfaces

# Initialize the network connector
network_connector = NetworkConnector()

def get_available_networks():
    """Convenience function to get available networks"""
    return network_connector.get_available_networks()

def connect_to_network(ssid, password=None):
    """Convenience function to connect to a network"""
    return network_connector.connect_to_network(ssid, password)

def create_hotspot(ssid, password=None):
    """Convenience function to create a hotspot"""
    return network_connector.create_hotspot(ssid, password)

def stop_hotspot():
    """Convenience function to stop a running hotspot"""
    return network_connector.stop_hotspot()

def get_connection_info():
    """Convenience function to get connection information"""
    return network_connector.get_connection_info()

def diagnose_network():
    """Convenience function to run network diagnostics"""
    return network_connector.diagnose_network()

def get_network_interfaces():
    """Convenience function to get network interfaces"""
    return network_connector.get_network_interfaces()