import streamlit as st
import os
import json
import uuid
import hashlib
import time
from authlib.integrations.requests_client import OAuth2Session
import requests
from urllib.parse import urlencode

class SocialAuth:
    """Handle social authentication with various providers"""
    
    def __init__(self):
        """Initialize social auth component"""
        # Initialize session state variables for social auth
        if "social_auth_state" not in st.session_state:
            st.session_state.social_auth_state = str(uuid.uuid4())
        
        if "social_auth_flow" not in st.session_state:
            st.session_state.social_auth_flow = None
        
        # Load OAuth credentials from secure storage
        self.credentials = self._load_oauth_credentials()
        
        # Define redirect URI (set to localhost for development)
        self.redirect_uri = "http://localhost:5000/callback"
    
    def _load_oauth_credentials(self):
        """Load OAuth client credentials from secure storage"""
        # Create secure directory if it doesn't exist
        os.makedirs("secure_assets", exist_ok=True)
        
        # Path to credentials file
        creds_path = "secure_assets/oauth_credentials.json"
        
        # Check if credentials file exists
        if os.path.exists(creds_path):
            try:
                with open(creds_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading OAuth credentials: {str(e)}")
                return {}
        else:
            # Create a template file if it doesn't exist
            template = {
                "google": {
                    "client_id": "",
                    "client_secret": "",
                    "configured": False
                },
                "microsoft": {
                    "client_id": "",
                    "client_secret": "",
                    "configured": False
                },
                "facebook": {
                    "client_id": "",
                    "client_secret": "",
                    "configured": False
                },
                "apple": {
                    "client_id": "",
                    "client_secret": "",
                    "team_id": "",
                    "key_id": "",
                    "private_key": "",
                    "configured": False
                }
            }
            
            with open(creds_path, "w") as f:
                json.dump(template, f, indent=4)
            
            return template
    
    def save_oauth_credentials(self, provider, credentials):
        """Save OAuth credentials for a provider"""
        # Load current credentials
        current_creds = self._load_oauth_credentials()
        
        # Update with new credentials
        current_creds[provider] = credentials
        current_creds[provider]["configured"] = True
        
        # Save to file
        creds_path = "secure_assets/oauth_credentials.json"
        with open(creds_path, "w") as f:
            json.dump(current_creds, f, indent=4)
        
        # Reload credentials
        self.credentials = self._load_oauth_credentials()
        
        return True
    
    def get_login_url(self, provider):
        """Get the authorization URL for the specified provider"""
        # Check if provider is configured
        if provider not in self.credentials or not self.credentials[provider].get("configured", False):
            return None
        
        # Google OAuth2 config
        if provider == "google":
            client_id = self.credentials["google"]["client_id"]
            
            auth_url = "https://accounts.google.com/o/oauth2/auth"
            params = {
                "client_id": client_id,
                "redirect_uri": self.redirect_uri,
                "response_type": "code",
                "scope": "openid email profile",
                "state": st.session_state.social_auth_state,
                "access_type": "offline",
                "prompt": "consent"
            }
            
            return f"{auth_url}?{urlencode(params)}"
        
        # Microsoft OAuth2 config
        elif provider == "microsoft":
            client_id = self.credentials["microsoft"]["client_id"]
            
            auth_url = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
            params = {
                "client_id": client_id,
                "redirect_uri": self.redirect_uri,
                "response_type": "code",
                "scope": "openid email profile User.Read",
                "state": st.session_state.social_auth_state,
                "response_mode": "query"
            }
            
            return f"{auth_url}?{urlencode(params)}"
        
        # Facebook OAuth2 config
        elif provider == "facebook":
            client_id = self.credentials["facebook"]["client_id"]
            
            auth_url = "https://www.facebook.com/v16.0/dialog/oauth"
            params = {
                "client_id": client_id,
                "redirect_uri": self.redirect_uri,
                "state": st.session_state.social_auth_state,
                "scope": "email,public_profile"
            }
            
            return f"{auth_url}?{urlencode(params)}"
        
        # Apple OAuth2 config
        elif provider == "apple":
            client_id = self.credentials["apple"]["client_id"]
            
            auth_url = "https://appleid.apple.com/auth/authorize"
            params = {
                "client_id": client_id,
                "redirect_uri": self.redirect_uri,
                "response_type": "code",
                "scope": "name email",
                "state": st.session_state.social_auth_state,
                "response_mode": "form_post"
            }
            
            return f"{auth_url}?{urlencode(params)}"
        
        return None
    
    def exchange_code(self, provider, code):
        """Exchange authorization code for access token"""
        if provider not in self.credentials or not self.credentials[provider].get("configured", False):
            return None
        
        client_id = self.credentials[provider]["client_id"]
        client_secret = self.credentials[provider]["client_secret"]
        
        # Google token endpoint
        if provider == "google":
            token_url = "https://oauth2.googleapis.com/token"
            
            data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code"
            }
            
            response = requests.post(token_url, data=data)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error exchanging Google code: {response.text}")
                return None
        
        # Microsoft token endpoint
        elif provider == "microsoft":
            token_url = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
            
            data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code"
            }
            
            response = requests.post(token_url, data=data)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error exchanging Microsoft code: {response.text}")
                return None
        
        # Facebook token endpoint
        elif provider == "facebook":
            token_url = "https://graph.facebook.com/v16.0/oauth/access_token"
            
            params = {
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": self.redirect_uri
            }
            
            response = requests.get(token_url, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error exchanging Facebook code: {response.text}")
                return None
        
        # Apple token endpoint
        elif provider == "apple":
            token_url = "https://appleid.apple.com/auth/token"
            
            # For Apple, we need to generate a client secret dynamically
            # This is a simplified version; in production, you'd need to create a proper JWT
            client_secret = self._generate_apple_client_secret()
            
            data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code"
            }
            
            response = requests.post(token_url, data=data)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error exchanging Apple code: {response.text}")
                return None
        
        return None
    
    def _generate_apple_client_secret(self):
        """Generate a client secret for Apple Sign In"""
        # In a real implementation, this would create a JWT token signed with your private key
        # This is a placeholder for the actual implementation
        if "apple" not in self.credentials or not self.credentials["apple"].get("configured", False):
            return None
        
        try:
            from authlib.jose import jwt
            import time
            
            # Load credentials
            key_id = self.credentials["apple"]["key_id"]
            team_id = self.credentials["apple"]["team_id"]
            client_id = self.credentials["apple"]["client_id"]
            private_key = self.credentials["apple"]["private_key"]
            
            # Create header
            header = {
                "alg": "ES256",
                "kid": key_id
            }
            
            # Create payload
            payload = {
                "iss": team_id,
                "iat": int(time.time()),
                "exp": int(time.time()) + 86400 * 180,  # 180 days
                "aud": "https://appleid.apple.com",
                "sub": client_id
            }
            
            # Create JWT
            client_secret = jwt.encode(header, payload, private_key)
            return client_secret.decode('utf-8')
            
        except Exception as e:
            st.error(f"Error generating Apple client secret: {str(e)}")
            return None
    
    def get_user_info(self, provider, token_response):
        """Get user information from the provider using the access token"""
        if not token_response or "access_token" not in token_response:
            return None
        
        access_token = token_response["access_token"]
        
        # Google user info endpoint
        if provider == "google":
            userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
            headers = {"Authorization": f"Bearer {access_token}"}
            
            response = requests.get(userinfo_url, headers=headers)
            
            if response.status_code == 200:
                user_data = response.json()
                return {
                    "provider": "google",
                    "id": user_data.get("sub"),
                    "email": user_data.get("email"),
                    "name": user_data.get("name"),
                    "picture": user_data.get("picture")
                }
            else:
                st.error(f"Error fetching Google user info: {response.text}")
                return None
        
        # Microsoft user info endpoint
        elif provider == "microsoft":
            userinfo_url = "https://graph.microsoft.com/v1.0/me"
            headers = {"Authorization": f"Bearer {access_token}"}
            
            response = requests.get(userinfo_url, headers=headers)
            
            if response.status_code == 200:
                user_data = response.json()
                return {
                    "provider": "microsoft",
                    "id": user_data.get("id"),
                    "email": user_data.get("userPrincipalName"),
                    "name": user_data.get("displayName"),
                    "picture": None  # Microsoft Graph doesn't provide a picture URL directly
                }
            else:
                st.error(f"Error fetching Microsoft user info: {response.text}")
                return None
        
        # Facebook user info endpoint
        elif provider == "facebook":
            userinfo_url = "https://graph.facebook.com/me"
            params = {
                "fields": "id,name,email,picture",
                "access_token": access_token
            }
            
            response = requests.get(userinfo_url, params=params)
            
            if response.status_code == 200:
                user_data = response.json()
                picture_url = user_data.get("picture", {}).get("data", {}).get("url") if "picture" in user_data else None
                
                return {
                    "provider": "facebook",
                    "id": user_data.get("id"),
                    "email": user_data.get("email"),
                    "name": user_data.get("name"),
                    "picture": picture_url
                }
            else:
                st.error(f"Error fetching Facebook user info: {response.text}")
                return None
        
        # Apple doesn't have a userinfo endpoint; user info comes in the ID token
        elif provider == "apple":
            # For Apple, the user info is included in the ID token
            # We'd need to decode the JWT token here
            if "id_token" in token_response:
                try:
                    from authlib.jose import jwt
                    
                    # Decode without verification for simplicity
                    # In production, you should verify the token properly
                    id_token = token_response["id_token"]
                    claims = jwt.decode(id_token, None, verify=False)
                    
                    return {
                        "provider": "apple",
                        "id": claims.get("sub"),
                        "email": claims.get("email"),
                        "name": None,  # Apple doesn't always provide name in the ID token
                        "picture": None
                    }
                except Exception as e:
                    st.error(f"Error decoding Apple ID token: {str(e)}")
                    return None
            else:
                st.error("No ID token found in Apple token response")
                return None
        
        return None
    
    def render_oauth_config(self):
        """Render the OAuth configuration interface for admins"""
        st.header("Social Login Configuration")
        st.markdown("""
        Configure social login providers for your application. You'll need to create OAuth applications 
        with each provider and enter the client credentials here.
        """)
        
        provider_tab_names = ["Google", "Microsoft", "Facebook", "Apple"]
        tabs = st.tabs(provider_tab_names)
        
        # Google configuration
        with tabs[0]:
            st.subheader("Google OAuth Configuration")
            
            google_creds = self.credentials.get("google", {})
            google_configured = google_creds.get("configured", False)
            
            if google_configured:
                st.success("Google OAuth is configured")
                
                # Show partial client ID
                client_id = google_creds.get("client_id", "")
                if client_id:
                    masked_id = f"{client_id[:5]}...{client_id[-5:]}" if len(client_id) > 10 else "Configured"
                    st.info(f"Client ID: {masked_id}")
                
                # Option to reconfigure
                if st.button("Reconfigure Google OAuth"):
                    st.session_state.reconfigure_google = True
            
            # Show configuration form if not configured or reconfiguring
            if not google_configured or st.session_state.get("reconfigure_google", False):
                with st.form("google_oauth_form"):
                    client_id = st.text_input(
                        "Client ID", 
                        value=google_creds.get("client_id", ""),
                        type="password"
                    )
                    client_secret = st.text_input(
                        "Client Secret", 
                        value=google_creds.get("client_secret", ""),
                        type="password"
                    )
                    
                    st.markdown("""
                    **How to set up Google OAuth:**
                    1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
                    2. Create a new project or select an existing one
                    3. Navigate to "APIs & Services" > "Credentials"
                    4. Click "Create Credentials" > "OAuth client ID"
                    5. Set up the OAuth consent screen if required
                    6. For Application type, select "Web application"
                    7. Add "http://localhost:5000/callback" as an authorized redirect URI
                    8. Copy the Client ID and Client Secret
                    """)
                    
                    submit_button = st.form_submit_button("Save Google Configuration")
                    
                    if submit_button:
                        if client_id and client_secret:
                            # Save the configuration
                            new_config = {
                                "client_id": client_id,
                                "client_secret": client_secret,
                                "configured": True
                            }
                            
                            if self.save_oauth_credentials("google", new_config):
                                st.success("Google OAuth configuration saved successfully!")
                                if "reconfigure_google" in st.session_state:
                                    del st.session_state.reconfigure_google
                                st.rerun()
                        else:
                            st.error("Please provide both Client ID and Client Secret")
        
        # Microsoft configuration
        with tabs[1]:
            st.subheader("Microsoft OAuth Configuration")
            
            ms_creds = self.credentials.get("microsoft", {})
            ms_configured = ms_creds.get("configured", False)
            
            if ms_configured:
                st.success("Microsoft OAuth is configured")
                
                # Show partial client ID
                client_id = ms_creds.get("client_id", "")
                if client_id:
                    masked_id = f"{client_id[:5]}...{client_id[-5:]}" if len(client_id) > 10 else "Configured"
                    st.info(f"Client ID: {masked_id}")
                
                # Option to reconfigure
                if st.button("Reconfigure Microsoft OAuth"):
                    st.session_state.reconfigure_microsoft = True
            
            # Show configuration form if not configured or reconfiguring
            if not ms_configured or st.session_state.get("reconfigure_microsoft", False):
                with st.form("microsoft_oauth_form"):
                    client_id = st.text_input(
                        "Client ID", 
                        value=ms_creds.get("client_id", ""),
                        type="password"
                    )
                    client_secret = st.text_input(
                        "Client Secret", 
                        value=ms_creds.get("client_secret", ""),
                        type="password"
                    )
                    
                    st.markdown("""
                    **How to set up Microsoft OAuth:**
                    1. Go to the [Azure Portal](https://portal.azure.com/)
                    2. Navigate to "Azure Active Directory" > "App registrations"
                    3. Click "New registration"
                    4. Enter a name for your application
                    5. For Redirect URI, select "Web" and enter "http://localhost:5000/callback"
                    6. After registration, go to "Certificates & secrets"
                    7. Create a new client secret
                    8. Copy the Application (client) ID and Client Secret
                    """)
                    
                    submit_button = st.form_submit_button("Save Microsoft Configuration")
                    
                    if submit_button:
                        if client_id and client_secret:
                            # Save the configuration
                            new_config = {
                                "client_id": client_id,
                                "client_secret": client_secret,
                                "configured": True
                            }
                            
                            if self.save_oauth_credentials("microsoft", new_config):
                                st.success("Microsoft OAuth configuration saved successfully!")
                                if "reconfigure_microsoft" in st.session_state:
                                    del st.session_state.reconfigure_microsoft
                                st.rerun()
                        else:
                            st.error("Please provide both Client ID and Client Secret")
        
        # Facebook configuration
        with tabs[2]:
            st.subheader("Facebook OAuth Configuration")
            
            fb_creds = self.credentials.get("facebook", {})
            fb_configured = fb_creds.get("configured", False)
            
            if fb_configured:
                st.success("Facebook OAuth is configured")
                
                # Show partial client ID
                client_id = fb_creds.get("client_id", "")
                if client_id:
                    masked_id = f"{client_id[:5]}...{client_id[-5:]}" if len(client_id) > 10 else "Configured"
                    st.info(f"Client ID: {masked_id}")
                
                # Option to reconfigure
                if st.button("Reconfigure Facebook OAuth"):
                    st.session_state.reconfigure_facebook = True
            
            # Show configuration form if not configured or reconfiguring
            if not fb_configured or st.session_state.get("reconfigure_facebook", False):
                with st.form("facebook_oauth_form"):
                    client_id = st.text_input(
                        "App ID", 
                        value=fb_creds.get("client_id", ""),
                        type="password"
                    )
                    client_secret = st.text_input(
                        "App Secret", 
                        value=fb_creds.get("client_secret", ""),
                        type="password"
                    )
                    
                    st.markdown("""
                    **How to set up Facebook OAuth:**
                    1. Go to the [Facebook Developers](https://developers.facebook.com/)
                    2. Create a new app or select an existing one
                    3. Add the "Facebook Login" product to your app
                    4. In the Facebook Login settings, add "http://localhost:5000/callback" as a valid OAuth redirect URI
                    5. Copy the App ID and App Secret from the app settings page
                    """)
                    
                    submit_button = st.form_submit_button("Save Facebook Configuration")
                    
                    if submit_button:
                        if client_id and client_secret:
                            # Save the configuration
                            new_config = {
                                "client_id": client_id,
                                "client_secret": client_secret,
                                "configured": True
                            }
                            
                            if self.save_oauth_credentials("facebook", new_config):
                                st.success("Facebook OAuth configuration saved successfully!")
                                if "reconfigure_facebook" in st.session_state:
                                    del st.session_state.reconfigure_facebook
                                st.rerun()
                        else:
                            st.error("Please provide both App ID and App Secret")
        
        # Apple configuration
        with tabs[3]:
            st.subheader("Apple Sign In Configuration")
            
            apple_creds = self.credentials.get("apple", {})
            apple_configured = apple_creds.get("configured", False)
            
            if apple_configured:
                st.success("Apple Sign In is configured")
                
                # Show partial client ID
                client_id = apple_creds.get("client_id", "")
                if client_id:
                    masked_id = f"{client_id[:5]}...{client_id[-5:]}" if len(client_id) > 10 else "Configured"
                    st.info(f"Client ID: {masked_id}")
                
                # Option to reconfigure
                if st.button("Reconfigure Apple Sign In"):
                    st.session_state.reconfigure_apple = True
            
            # Show configuration form if not configured or reconfiguring
            if not apple_configured or st.session_state.get("reconfigure_apple", False):
                with st.form("apple_oauth_form"):
                    client_id = st.text_input(
                        "Service ID", 
                        value=apple_creds.get("client_id", ""),
                        type="password"
                    )
                    team_id = st.text_input(
                        "Team ID", 
                        value=apple_creds.get("team_id", ""),
                        type="password"
                    )
                    key_id = st.text_input(
                        "Key ID", 
                        value=apple_creds.get("key_id", ""),
                        type="password"
                    )
                    private_key = st.text_area(
                        "Private Key (contents of the .p8 file)", 
                        value=apple_creds.get("private_key", ""),
                        height=150
                    )
                    
                    st.markdown("""
                    **How to set up Apple Sign In:**
                    1. Go to the [Apple Developer Portal](https://developer.apple.com/)
                    2. Navigate to "Certificates, Identifiers & Profiles"
                    3. Create a new Services ID and enable "Sign In with Apple"
                    4. Add "http://localhost:5000/callback" as a Return URL
                    5. Create a new private key for "Sign In with Apple"
                    6. Take note of your Team ID, Key ID, and download the private key (.p8) file
                    7. Enter the Service ID, Team ID, Key ID and the contents of the .p8 file
                    """)
                    
                    submit_button = st.form_submit_button("Save Apple Configuration")
                    
                    if submit_button:
                        if client_id and team_id and key_id and private_key:
                            # Save the configuration
                            new_config = {
                                "client_id": client_id,
                                "client_secret": "",  # Apple uses dynamically generated client secrets
                                "team_id": team_id,
                                "key_id": key_id,
                                "private_key": private_key,
                                "configured": True
                            }
                            
                            if self.save_oauth_credentials("apple", new_config):
                                st.success("Apple Sign In configuration saved successfully!")
                                if "reconfigure_apple" in st.session_state:
                                    del st.session_state.reconfigure_apple
                                st.rerun()
                        else:
                            st.error("Please provide all required Apple Sign In credentials")
    
    def render_social_login_buttons(self):
        """Render social login buttons"""
        st.subheader("Sign in with")
        
        # Create columns for login buttons
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        # Check which providers are configured
        google_configured = self.credentials.get("google", {}).get("configured", False)
        microsoft_configured = self.credentials.get("microsoft", {}).get("configured", False)
        facebook_configured = self.credentials.get("facebook", {}).get("configured", False)
        apple_configured = self.credentials.get("apple", {}).get("configured", False)
        
        with col1:
            if google_configured:
                google_url = self.get_login_url("google")
                if google_url:
                    # Use HTML for a nicer button
                    st.markdown(
                        f'<a href="{google_url}" target="_self" style="text-decoration:none;">'
                        f'<div style="background-color:#4285F4;color:white;padding:8px 12px;'
                        f'border-radius:4px;cursor:pointer;text-align:center;margin:4px 0;">'
                        f'<img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg" '
                        f'style="height:20px;vertical-align:middle;margin-right:10px;background-color:white;padding:2px;border-radius:2px;">'
                        f'Google</div></a>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    '<div style="background-color:#f0f0f0;color:#888;padding:8px 12px;'
                    'border-radius:4px;text-align:center;margin:4px 0;cursor:not-allowed;">'
                    'Google (Not Configured)</div>',
                    unsafe_allow_html=True
                )
        
        with col2:
            if microsoft_configured:
                microsoft_url = self.get_login_url("microsoft")
                if microsoft_url:
                    # Use HTML for a nicer button
                    st.markdown(
                        f'<a href="{microsoft_url}" target="_self" style="text-decoration:none;">'
                        f'<div style="background-color:#2F2F2F;color:white;padding:8px 12px;'
                        f'border-radius:4px;cursor:pointer;text-align:center;margin:4px 0;">'
                        f'<img src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" '
                        f'style="height:20px;vertical-align:middle;margin-right:10px;background-color:white;padding:2px;border-radius:2px;">'
                        f'Microsoft</div></a>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    '<div style="background-color:#f0f0f0;color:#888;padding:8px 12px;'
                    'border-radius:4px;text-align:center;margin:4px 0;cursor:not-allowed;">'
                    'Microsoft (Not Configured)</div>',
                    unsafe_allow_html=True
                )
        
        with col3:
            if facebook_configured:
                facebook_url = self.get_login_url("facebook")
                if facebook_url:
                    # Use HTML for a nicer button
                    st.markdown(
                        f'<a href="{facebook_url}" target="_self" style="text-decoration:none;">'
                        f'<div style="background-color:#1877F2;color:white;padding:8px 12px;'
                        f'border-radius:4px;cursor:pointer;text-align:center;margin:4px 0;">'
                        f'<img src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" '
                        f'style="height:20px;vertical-align:middle;margin-right:10px;background-color:white;padding:2px;border-radius:2px;">'
                        f'Facebook</div></a>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    '<div style="background-color:#f0f0f0;color:#888;padding:8px 12px;'
                    'border-radius:4px;text-align:center;margin:4px 0;cursor:not-allowed;">'
                    'Facebook (Not Configured)</div>',
                    unsafe_allow_html=True
                )
        
        with col4:
            if apple_configured:
                apple_url = self.get_login_url("apple")
                if apple_url:
                    # Use HTML for a nicer button
                    st.markdown(
                        f'<a href="{apple_url}" target="_self" style="text-decoration:none;">'
                        f'<div style="background-color:#000000;color:white;padding:8px 12px;'
                        f'border-radius:4px;cursor:pointer;text-align:center;margin:4px 0;">'
                        f'<img src="https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg" '
                        f'style="height:20px;vertical-align:middle;margin-right:10px;background-color:white;padding:2px;border-radius:2px;">'
                        f'Apple</div></a>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    '<div style="background-color:#f0f0f0;color:#888;padding:8px 12px;'
                    'border-radius:4px;text-align:center;margin:4px 0;cursor:not-allowed;">'
                    'Apple (Not Configured)</div>',
                    unsafe_allow_html=True
                )
        
        # Note about configuration
        if not any([google_configured, microsoft_configured, facebook_configured, apple_configured]):
            st.info("Social login providers need to be configured by an administrator.")

def social_login_page():
    """Main function to render the social login page"""
    st.title("Social Login")
    
    # Initialize social auth handler
    social_auth = SocialAuth()
    
    # Check if we're in the auth callback
    query_params = st.experimental_get_query_params()
    
    # Handle callback from OAuth providers
    if "code" in query_params and "state" in query_params:
        code = query_params["code"][0]
        state = query_params["state"][0]
        
        # Verify state to prevent CSRF
        if state == st.session_state.social_auth_state:
            # Determine provider from state or other means
            provider = st.session_state.get("oauth_provider", "google")  # Default to Google
            
            # Exchange code for token
            token_response = social_auth.exchange_code(provider, code)
            
            if token_response:
                # Get user info
                user_info = social_auth.get_user_info(provider, token_response)
                
                if user_info and user_info.get("email"):
                    # Store user info in session
                    st.session_state.social_user = user_info
                    
                    # Create or update user in the database
                    from components.user_management import create_user
                    
                    # Check if user exists, if not create a new user
                    email = user_info["email"]
                    name = user_info.get("name", email.split("@")[0])
                    
                    # Use default role for social login users
                    role = "user"
                    
                    # Create user (this function should handle existing users gracefully)
                    create_user(email, name, role)
                    
                    # Set authenticated state
                    st.session_state.user_authenticated = True
                    st.session_state.current_user = email
                    
                    # Also set W&B authenticated for backward compatibility
                    st.session_state.authenticated = True
                    
                    # Redirect to projects page
                    st.session_state.current_page = "projects"
                    st.rerun()
                else:
                    st.error("Could not retrieve user information from the provider. Please try again or use another login method.")
            else:
                st.error("Failed to authenticate with the provider. Please try again or use another login method.")
        else:
            st.error("Invalid state parameter. This could be a cross-site request forgery attempt.")
    
    # For admin users, show OAuth configuration
    if st.session_state.get("user_authenticated") and st.session_state.get("user_role") == "admin":
        social_auth.render_oauth_config()
        st.divider()
    
    # Render social login buttons
    social_auth.render_social_login_buttons()
    
    # Also provide traditional login option
    st.divider()
    st.markdown("Or use traditional login:")
    
    # Add link to traditional login
    if st.button("Email & Password Login"):
        st.session_state.current_page = "authentication"
        st.rerun()

def social_auth_callback_handler():
    """Handle OAuth callback in the main app.py"""
    # Get query parameters
    query_params = st.experimental_get_query_params()
    
    # Check if this is an OAuth callback
    if "code" in query_params and "state" in query_params:
        # Set the current page to social_login to handle the callback
        st.session_state.current_page = "social_login"
        
        # If we can determine the provider from the parameters, set it
        # This is a simplified version; in production you might need more robust detection
        if "microsoft" in query_params.get("session_state", [""]):
            st.session_state.oauth_provider = "microsoft"
        elif "appleid" in query_params.get("id_token", [""]):
            st.session_state.oauth_provider = "apple"
        # Add more provider detection logic as needed