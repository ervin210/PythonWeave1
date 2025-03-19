import streamlit as st
import requests
import json
import os
import uuid
import time
from authlib.integrations.requests_client import OAuth2Session
import hashlib
import base64
from datetime import datetime, timedelta
from utils.key_generator import generate_unique_key, generate_button_key, generate_widget_key

class SocialAuth:
    """Handle social authentication with various providers"""
    
    def __init__(self):
        """Initialize social auth component"""
        self.providers = {
            "google": {
                "name": "Google",
                "icon": "https://img.icons8.com/color/48/000000/google-logo.png",
                "auth_url": "https://accounts.google.com/o/oauth2/auth",
                "token_url": "https://oauth2.googleapis.com/token",
                "userinfo_url": "https://www.googleapis.com/oauth2/v3/userinfo",
                "scope": ["openid", "email", "profile"],
                "color": "#4285F4"
            },
            "microsoft": {
                "name": "Microsoft",
                "icon": "https://img.icons8.com/color/48/000000/microsoft.png",
                "auth_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                "userinfo_url": "https://graph.microsoft.com/v1.0/me",
                "scope": ["User.Read", "email", "profile", "openid"],
                "color": "#00A4EF"
            },
            "apple": {
                "name": "Apple",
                "icon": "https://img.icons8.com/ios-filled/50/000000/mac-os.png",
                "auth_url": "https://appleid.apple.com/auth/authorize",
                "token_url": "https://appleid.apple.com/auth/token",
                "scope": ["name", "email"],
                "color": "#000000"
            },
            "facebook": {
                "name": "Facebook",
                "icon": "https://img.icons8.com/color/48/000000/facebook-new.png",
                "auth_url": "https://www.facebook.com/v12.0/dialog/oauth",
                "token_url": "https://graph.facebook.com/v12.0/oauth/access_token",
                "userinfo_url": "https://graph.facebook.com/me?fields=name,email",
                "scope": ["email", "public_profile"],
                "color": "#1877F2"
            }
        }
        
        # Load credentials for providers if available
        self._load_oauth_credentials()
        
    def _load_oauth_credentials(self):
        """Load OAuth client credentials from secure storage"""
        # Initialize session state variables for oauth credentials if not exist
        if 'oauth_credentials' not in st.session_state:
            st.session_state.oauth_credentials = {}
            
        # Initialize with environment variables if available
        for provider in self.providers:
            provider_upper = provider.upper()
            client_id_env = f"{provider_upper}_CLIENT_ID"
            client_secret_env = f"{provider_upper}_CLIENT_SECRET"
            
            if client_id_env in os.environ and client_secret_env in os.environ:
                if provider not in st.session_state.oauth_credentials:
                    st.session_state.oauth_credentials[provider] = {
                        "client_id": os.environ[client_id_env],
                        "client_secret": os.environ[client_secret_env]
                    }
                    
        # Add demo credentials for testing if no providers are configured
        if not st.session_state.oauth_credentials and not os.path.exists(".oauth_configured"):
            # Add a demo Google configuration (these are placeholder values)
            st.session_state.oauth_credentials["google"] = {
                "client_id": "demo-client-id",
                "client_secret": "demo-client-secret"
            }
        
    def save_oauth_credentials(self, provider, credentials):
        """Save OAuth credentials for a provider"""
        if 'oauth_credentials' not in st.session_state:
            st.session_state.oauth_credentials = {}
            
        st.session_state.oauth_credentials[provider] = credentials
        
    def get_login_url(self, provider):
        """Get the authorization URL for the specified provider"""
        if provider not in self.providers:
            return None, "Provider not supported"
            
        if provider not in st.session_state.oauth_credentials:
            return None, "Provider credentials not configured"
            
        # Generate state for CSRF protection
        state = st.session_state.social_auth_state
        
        provider_config = self.providers[provider]
        client_id = st.session_state.oauth_credentials[provider]["client_id"]
        
        # Create OAuth session
        session = OAuth2Session(
            client_id, 
            scope=provider_config.get("scope", []),
            redirect_uri=self._get_redirect_uri()
        )
        
        # Get authorization URL
        auth_url, _ = session.create_authorization_url(
            provider_config["auth_url"],
            state=state
        )
        
        return auth_url, None
        
    def exchange_code(self, provider, code):
        """Exchange authorization code for access token"""
        if provider not in self.providers or provider not in st.session_state.oauth_credentials:
            return None, "Provider not configured"
        
        provider_config = self.providers[provider]
        credentials = st.session_state.oauth_credentials[provider]
        
        # For Apple, generate a client secret
        client_secret = credentials["client_secret"]
        if provider == "apple":
            client_secret = self._generate_apple_client_secret()
        
        # Create OAuth session
        session = OAuth2Session(
            credentials["client_id"],
            redirect_uri=self._get_redirect_uri()
        )
        
        try:
            # Exchange code for token
            token = session.fetch_token(
                provider_config["token_url"],
                code=code,
                client_secret=client_secret,
                include_client_id=True
            )
            return token, None
        except Exception as e:
            return None, f"Error exchanging code: {str(e)}"
    
    def _generate_apple_client_secret(self):
        """Generate a client secret for Apple Sign In"""
        if 'apple' not in st.session_state.oauth_credentials:
            return None
            
        credentials = st.session_state.oauth_credentials['apple']
        
        if 'team_id' not in credentials or 'key_id' not in credentials or 'private_key' not in credentials:
            # If we don't have the required fields, just return the stored client secret
            return credentials.get("client_secret", "")
            
        # Import JWT libraries only when needed
        from authlib.jose import jwt
        
        # Get configuration
        team_id = credentials['team_id']
        key_id = credentials['key_id']
        private_key = credentials['private_key']
        client_id = credentials['client_id']
        
        # Headers
        headers = {
            'kid': key_id
        }
        
        # Create payload
        time_now = int(time.time())
        payload = {
            'iss': team_id,
            'iat': time_now,
            'exp': time_now + 86400*180,  # 180 days
            'aud': 'https://appleid.apple.com',
            'sub': client_id
        }
        
        # Generate JWT
        try:
            client_secret = jwt.encode(headers, payload, private_key).decode('utf-8')
            return client_secret
        except Exception as e:
            st.error(f"Error generating Apple client secret: {str(e)}")
            return credentials.get("client_secret", "")
    
    def get_user_info(self, provider, token_response):
        """Get user information from the provider using the access token"""
        if provider not in self.providers:
            return None, "Provider not supported"
            
        provider_config = self.providers[provider]
        
        # Handle special case for Apple
        if provider == "apple" and "id_token" in token_response:
            try:
                from authlib.jose import jwt
                # Decode the id_token to get user information
                id_token = token_response["id_token"]
                claims = jwt.decode(id_token, None, verify=False)
                
                # Extract user info from claims
                user_info = {
                    "sub": claims.get("sub"),
                    "email": claims.get("email"),
                    "name": claims.get("name", "Apple User"),
                    "provider": "apple"
                }
                return user_info, None
            except Exception as e:
                return None, f"Error decoding Apple ID token: {str(e)}"
        
        # For other providers, make API call to get user info
        if "userinfo_url" not in provider_config:
            return None, "Provider doesn't support user info endpoint"
            
        access_token = token_response.get("access_token")
        if not access_token:
            return None, "No access token available"
            
        headers = {"Authorization": f"Bearer {access_token}"}
        
        try:
            response = requests.get(provider_config["userinfo_url"], headers=headers)
            response.raise_for_status()
            user_info = response.json()
            
            # Add provider to user info
            user_info["provider"] = provider
            
            return user_info, None
        except Exception as e:
            return None, f"Error fetching user info: {str(e)}"
    
    def render_oauth_config(self):
        """Render the OAuth configuration interface for admins"""
        st.subheader("OAuth Provider Configuration")
        
        selected_provider = st.selectbox(
            "Select Provider",
            list(self.providers.keys()),
            format_func=lambda x: self.providers[x]["name"],
            key=generate_widget_key("selectbox", "oauth_provider")
        )
        
        with st.form(f"oauth_config_{selected_provider}"):
            provider_name = self.providers[selected_provider]["name"]
            st.markdown(f"### {provider_name} Configuration")
            
            # Get current config if available
            current_config = st.session_state.oauth_credentials.get(selected_provider, {})
            
            client_id = st.text_input(
                f"{provider_name} Client ID",
                value=current_config.get("client_id", ""),
                type="default",
                key=generate_widget_key("text_input", f"{selected_provider}_client_id")
            )
            
            client_secret = st.text_input(
                f"{provider_name} Client Secret",
                value=current_config.get("client_secret", ""),
                type="password",
                key=generate_widget_key("text_input", f"{selected_provider}_client_secret")
            )
            
            # Additional fields for Apple
            additional_fields = {}
            if selected_provider == "apple":
                additional_fields["team_id"] = st.text_input(
                    "Team ID",
                    value=current_config.get("team_id", ""),
                    help="Your Apple Developer Team ID",
                    key=generate_widget_key("text_input", "apple_team_id")
                )
                
                additional_fields["key_id"] = st.text_input(
                    "Key ID",
                    value=current_config.get("key_id", ""),
                    help="Your Sign in with Apple Key ID",
                    key=generate_widget_key("text_input", "apple_key_id")
                )
                
                private_key = st.text_area(
                    "Private Key",
                    value=current_config.get("private_key", ""),
                    help="Your Sign in with Apple private key (p8 file content)",
                    key=generate_widget_key("text_area", "apple_private_key")
                )
                if private_key:
                    additional_fields["private_key"] = private_key
            
            submitted = st.form_submit_button(
                "Save Configuration", 
                key=generate_widget_key("form_submit", f"{selected_provider}_config")
            )
            
            if submitted and client_id and client_secret:
                credentials = {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    **additional_fields
                }
                
                self.save_oauth_credentials(selected_provider, credentials)
                st.success(f"{provider_name} configuration saved!")
    
    def render_social_login_buttons(self):
        """Render social login buttons"""
        st.markdown("### Sign in with:")
        
        # Check which providers are configured
        configured_providers = []
        for provider, config in self.providers.items():
            if provider in st.session_state.oauth_credentials:
                configured_providers.append(provider)
        
        if not configured_providers:
            st.warning("No OAuth providers are configured. Please contact your administrator.")
            return
        
        # Use columns for buttons
        cols = st.columns(min(len(configured_providers), 4))
        
        for i, provider in enumerate(configured_providers):
            config = self.providers[provider]
            col = cols[i % len(cols)]
            
            # Generate a unique button key using our utility function
            btn_key = generate_button_key(f"social_login_{provider}")
            
            with col:
                if col.button(
                    config["name"],
                    key=btn_key,
                    use_container_width=True,
                ):
                    auth_url, error = self.get_login_url(provider)
                    if error:
                        st.error(error)
                    else:
                        # Store the provider for callback handling
                        st.session_state.oauth_provider = provider
                        # Redirect to auth URL
                        st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', unsafe_allow_html=True)
    
    def _get_redirect_uri(self):
        """Get the redirect URI for OAuth flow"""
        # Use the current URL as the base for the redirect URI
        base_url = os.environ.get("REPLIT_URL", "http://localhost:5000")
        if not base_url.startswith("http"):
            base_url = f"https://{base_url}"
            
        # Ensure the URL doesn't end with a slash
        base_url = base_url.rstrip("/")
        
        # The callback URL is the same as the base URL (we'll handle it in the app)
        return base_url

def social_login_page():
    """Main function to render the social login page"""
    st.header("Sign in with Social Accounts")
    st.markdown("Connect to Quantum AI Assistant using your social accounts.")
    
    auth = SocialAuth()
    auth.render_social_login_buttons()
    
    if st.session_state.get("user_authenticated", False):
        st.success("You're already logged in!")
        # Use the utility function for unique key generation
        btn_key = generate_button_key("continue_dashboard")
        if st.button("Continue to Dashboard", key=btn_key):
            st.rerun()

def social_auth_callback_handler():
    """Handle OAuth callback in the main app.py"""
    # Check if this is a callback from OAuth provider
    query_params = st.query_params
    
    if "code" in query_params and "state" in query_params:
        code = query_params["code"]
        state = query_params["state"]
        
        # Verify state to prevent CSRF attacks
        if state != st.session_state.social_auth_state:
            st.error("Invalid state parameter. Authentication failed.")
            # Clear query parameters
            st.query_params.clear()
            return
        
        # Get the provider from session state
        provider = st.session_state.get("oauth_provider")
        if not provider:
            st.error("Authentication provider not found. Please try again.")
            # Clear query parameters
            st.query_params.clear()
            return
        
        # Process the authentication
        auth = SocialAuth()
        token_response, error = auth.exchange_code(provider, code)
        
        if error:
            st.error(f"Authentication failed: {error}")
        elif token_response:
            # Get user info
            user_info, user_error = auth.get_user_info(provider, token_response)
            
            if user_error:
                st.error(f"Failed to get user information: {user_error}")
            elif user_info:
                # Store user info in session state
                email = user_info.get("email")
                name = user_info.get("name", "User")
                provider_name = auth.providers[provider]["name"]
                
                if email:
                    # Import user management functions
                    from components.user_management import get_current_user, authenticate_user, create_user
                    
                    # Check if user exists, if not create them
                    current_user = get_current_user()
                    
                    if not current_user:
                        # Try to authenticate with the email
                        if not authenticate_user(email, None, oauth=True):
                            # User doesn't exist, create a new account
                            create_user(
                                email=email,
                                name=name,
                                role="user",
                                created_by="social_auth",
                                oauth_provider=provider
                            )
                            # Try to authenticate again
                            authenticate_user(email, None, oauth=True)
                    
                    st.success(f"Successfully signed in with {provider_name}!")
                    
                    # If W&B API key is present, authenticate with W&B
                    if "api_key" in st.session_state and st.session_state.api_key:
                        from app import authenticate_wandb
                        authenticate_wandb(st.session_state.api_key)
                else:
                    st.error("Could not get email from social provider. Email is required.")
        
        # Reset OAuth state
        st.session_state.oauth_provider = None
        
        # Clear query parameters
        st.query_params.clear()
        
        # Rerun the app to remove URL parameters
        st.rerun()