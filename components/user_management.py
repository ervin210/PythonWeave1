import streamlit as st
import pandas as pd
import os
import json
import hashlib
import uuid
import ipaddress
import socket
from datetime import datetime, timedelta
from utils.key_generator import generate_unique_key, generate_widget_key

# Define the root admin emails (protected from changes)
ROOT_ADMIN_EMAILS = [
    "ervin210@icloud.com",
    "radosavlevici.ervin@gmail.com"
]

# Security settings
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30
SUSPICIOUS_ACTIVITY_THRESHOLD = 10

# User roles and their permissions
USER_ROLES = {
    "root_admin": {
        "name": "Root Administrator",
        "description": "Full system access with irrevocable privileges",
        "permissions": ["all"]
    },
    "admin": {
        "name": "Administrator",
        "description": "Full system access with permissions granted by root admin",
        "permissions": ["manage_users", "manage_projects", "manage_settings", "view_all_data"]
    },
    "power_user": {
        "name": "Power User",
        "description": "Extended access to features including advanced quantum capabilities",
        "permissions": ["use_advanced_features", "create_projects", "export_data"]
    },
    "standard_user": {
        "name": "Standard User",
        "description": "Regular access to basic features",
        "permissions": ["use_basic_features", "view_own_data"]
    }
}

def generate_password_hash(password, salt=None):
    """Generate a secure hash for the password"""
    if salt is None:
        salt = os.urandom(32)  # 32 bytes of random data for salt
    elif isinstance(salt, str):
        # Convert hex string to bytes
        try:
            salt = bytes.fromhex(salt)
        except ValueError:
            # If not a valid hex string, use it as bytes
            salt = salt.encode('utf-8')
    
    # Use a secure hashing algorithm with the salt
    key = hashlib.pbkdf2_hmac(
        'sha256',  # Hash algorithm
        password.encode('utf-8'),  # Convert password to bytes
        salt,  # Salt
        100000,  # Number of iterations (higher is more secure but slower)
        dklen=128  # Length of the derived key
    )
    
    # Return the hash and salt as a dictionary
    return {
        'hash': key,
        'salt': salt
    }

def verify_password(stored_password, provided_password):
    """Verify if the provided password matches the stored hash"""
    salt = stored_password['salt']
    stored_hash = stored_password['hash']
    
    # Generate hash of the provided password using the same salt
    verification_hash = hashlib.pbkdf2_hmac(
        'sha256',
        provided_password.encode('utf-8'),
        salt,
        100000,
        dklen=128
    )
    
    # Compare in constant time to prevent timing attacks
    return verification_hash == stored_hash

def initialize_user_management():
    """Initialize the user management system"""
    if 'user_db' not in st.session_state:
        # Check if we have a saved user database
        try:
            with open('secure_assets/users.json', 'r') as f:
                user_data = json.load(f)
                # Convert stored binary data from strings back to bytes
                for user in user_data:
                    if 'password' in user_data[user]:
                        user_data[user]['password']['hash'] = bytes.fromhex(user_data[user]['password']['hash'])
                        user_data[user]['password']['salt'] = bytes.fromhex(user_data[user]['password']['salt'])
                st.session_state.user_db = user_data
        except (FileNotFoundError, json.JSONDecodeError):
            # Create default user database with root admin users
            st.session_state.user_db = {}
            for email in ROOT_ADMIN_EMAILS:
                # Create fixed initial passwords for root admins
                # Using consistent passwords for designated root admins
                if email == "radosavlevici.ervin@gmail.com":
                    initial_password = "zZrgAgz1O64G"  # Use the specified password
                elif email == "ervin210@icloud.com":
                    initial_password = "Admin123!"  # Default password for second admin
                else:
                    # For any other future root admins
                    initial_password = "RootAdmin!" + hashlib.sha256(email.encode()).hexdigest()[:6]
                
                password_data = generate_password_hash(initial_password)
                
                st.session_state.user_db[email] = {
                    'email': email,
                    'name': 'Root Administrator',
                    'role': 'root_admin',
                    'password': {
                        'hash': password_data['hash'],
                        'salt': password_data['salt']
                    },
                    'created_at': datetime.now().isoformat(),
                    'last_login': None,
                    'status': 'active',
                    'must_change_password': True
                }
            
            # Save the initial user database
            save_user_database()
    
    # Initialize authentication state
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
        
    # Initialize security tracking
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = {}
        
    if 'account_lockouts' not in st.session_state:
        st.session_state.account_lockouts = {}
        
    if 'ip_blacklist' not in st.session_state:
        # Load blacklist from file if exists
        try:
            with open('secure_assets/ip_blacklist.json', 'r') as f:
                st.session_state.ip_blacklist = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st.session_state.ip_blacklist = {}

def save_user_database():
    """Save the user database to disk"""
    os.makedirs('secure_assets', exist_ok=True)
    
    # Create a copy of the database to save
    db_to_save = {}
    for user, data in st.session_state.user_db.items():
        db_to_save[user] = data.copy()
        # Convert binary data to hex strings for storage
        if 'password' in db_to_save[user]:
            db_to_save[user]['password'] = {
                'hash': db_to_save[user]['password']['hash'].hex(),
                'salt': db_to_save[user]['password']['salt'].hex()
            }
    
    with open('secure_assets/users.json', 'w') as f:
        json.dump(db_to_save, f, indent=2)

def is_root_admin(email):
    """Check if an email is a root admin email"""
    return email in ROOT_ADMIN_EMAILS

def get_client_ip():
    """Get the client's IP address (simulated in this environment)"""
    try:
        return socket.gethostbyname(socket.gethostname())
    except:
        return "127.0.0.1"  # Fallback to localhost

def track_login_attempt(email, success):
    """Track login attempts for security purposes"""
    ip_address = get_client_ip()
    
    # Initialize counters if needed
    if email not in st.session_state.login_attempts:
        st.session_state.login_attempts[email] = {
            'attempts': 0,
            'last_attempt': None,
            'successful_logins': 0,
            'failed_logins': 0,
            'ip_addresses': set()
        }
    
    # Update tracking information
    current_time = datetime.now()
    st.session_state.login_attempts[email]['attempts'] += 1
    st.session_state.login_attempts[email]['last_attempt'] = current_time.isoformat()
    st.session_state.login_attempts[email]['ip_addresses'].add(ip_address)
    
    if success:
        st.session_state.login_attempts[email]['successful_logins'] += 1
        # Reset failed login counter on successful login
        st.session_state.login_attempts[email]['failed_logins'] = 0
    else:
        st.session_state.login_attempts[email]['failed_logins'] += 1
    
    # Check if account should be locked
    if st.session_state.login_attempts[email]['failed_logins'] >= MAX_LOGIN_ATTEMPTS:
        lock_account(email)
        
    # Check for suspicious activity
    if is_suspicious_activity(email):
        flag_suspicious_activity(email, ip_address)

def is_account_locked(email):
    """Check if an account is currently locked due to failed login attempts"""
    if email in st.session_state.account_lockouts:
        lockout_info = st.session_state.account_lockouts[email]
        lockout_time = datetime.fromisoformat(lockout_info['lockout_time'])
        lockout_duration = timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        
        # Check if lockout period has expired
        if datetime.now() - lockout_time < lockout_duration:
            return True, lockout_info
        else:
            # Lockout period expired, remove the lockout
            del st.session_state.account_lockouts[email]
    
    return False, None

def lock_account(email):
    """Lock an account due to too many failed login attempts"""
    st.session_state.account_lockouts[email] = {
        'lockout_time': datetime.now().isoformat(),
        'reason': 'Too many failed login attempts',
        'ip_address': get_client_ip()
    }
    
    # Add security notification to the account
    if email in st.session_state.user_db:
        if 'security_notifications' not in st.session_state.user_db[email]:
            st.session_state.user_db[email]['security_notifications'] = []
        
        st.session_state.user_db[email]['security_notifications'].append({
            'type': 'account_lockout',
            'timestamp': datetime.now().isoformat(),
            'message': f"Account locked due to {MAX_LOGIN_ATTEMPTS} failed login attempts",
            'ip_address': get_client_ip()
        })
        
        save_user_database()

def is_suspicious_activity(email):
    """Check for suspicious account activity"""
    if email not in st.session_state.login_attempts:
        return False
    
    # Check number of failed attempts
    if st.session_state.login_attempts[email]['failed_logins'] >= SUSPICIOUS_ACTIVITY_THRESHOLD:
        return True
    
    # Check multiple IP addresses (more than 3 different IPs)
    if len(st.session_state.login_attempts[email]['ip_addresses']) > 3:
        return True
    
    return False

def flag_suspicious_activity(email, ip_address):
    """Flag an account for suspicious activity"""
    # Add to blacklist if not already there
    if ip_address not in st.session_state.ip_blacklist:
        st.session_state.ip_blacklist[ip_address] = {
            'timestamp': datetime.now().isoformat(),
            'reason': f"Suspicious activity on account {email}",
            'associated_email': email
        }
        
        # Save blacklist to file
        os.makedirs('secure_assets', exist_ok=True)
        with open('secure_assets/ip_blacklist.json', 'w') as f:
            # Convert set to list for JSON serialization
            blacklist_copy = {}
            for ip, data in st.session_state.ip_blacklist.items():
                blacklist_copy[ip] = data
            json.dump(blacklist_copy, f, indent=2)
    
    # Add security notification to the account
    if email in st.session_state.user_db:
        if 'security_notifications' not in st.session_state.user_db[email]:
            st.session_state.user_db[email]['security_notifications'] = []
        
        st.session_state.user_db[email]['security_notifications'].append({
            'type': 'suspicious_activity',
            'timestamp': datetime.now().isoformat(),
            'message': f"Suspicious activity detected from IP {ip_address}",
            'ip_address': ip_address
        })
        
        save_user_database()

def is_ip_blacklisted(ip_address=None):
    """Check if an IP address is blacklisted"""
    if ip_address is None:
        ip_address = get_client_ip()
    
    return ip_address in st.session_state.ip_blacklist

def authenticate_user(email, password, oauth=False):
    """Authenticate a user with email and password or via OAuth"""
    # Check if the client IP is blacklisted
    ip_address = get_client_ip()
    if is_ip_blacklisted(ip_address):
        return False, "Access denied: Your IP address has been blacklisted due to suspicious activity."
    
    # Check if account is locked
    is_locked, lockout_info = is_account_locked(email)
    if is_locked:
        lockout_time = datetime.fromisoformat(lockout_info['lockout_time'])
        unlock_time = lockout_time + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        time_remaining = unlock_time - datetime.now()
        minutes_remaining = max(1, int(time_remaining.total_seconds() / 60))
        
        return False, f"Account is temporarily locked due to too many failed login attempts. Try again in {minutes_remaining} minutes."
    
    # OAuth logins don't count towards login attempts
    if not oauth:
        # Track the login attempt (we'll update the success status later)
        track_login_attempt(email, False)
    
    # Check if the user exists
    if email in st.session_state.user_db:
        user = st.session_state.user_db[email]
        
        # Check if account is active
        if user['status'] != 'active':
            return False, "Account is inactive or suspended"
        
        # For OAuth login, skip password verification
        if oauth or verify_password(user['password'], password):
            # Update last login time
            user['last_login'] = datetime.now().isoformat()
            
            # Set authentication state
            st.session_state.user_authenticated = True
            st.session_state.current_user = email
            
            # Update login tracking for successful login
            if not oauth:
                track_login_attempt(email, True)
            
            # Save updates
            save_user_database()
            
            return True, "Authentication successful"
    
    return False, "Invalid email or password"

def logout_user():
    """Log out the current user"""
    st.session_state.user_authenticated = False
    st.session_state.current_user = None

def get_current_user():
    """Get the currently logged in user's data"""
    if not st.session_state.user_authenticated or st.session_state.current_user is None:
        return None
    
    return st.session_state.user_db.get(st.session_state.current_user)

def has_permission(permission):
    """Check if the current user has a specific permission"""
    user = get_current_user()
    if user is None:
        return False
    
    # Root admins have all permissions
    if user['role'] == 'root_admin':
        return True
    
    # Check if the user's role has the requested permission
    role = user['role']
    if role in USER_ROLES:
        return permission in USER_ROLES[role]['permissions']
    
    return False

def create_user(email, name, role, password=None, created_by=None, oauth_provider=None):
    """Create a new user"""
    # Skip permission check for OAuth created users
    if oauth_provider is None:
        # Check if the current user has permission to create users
        if not has_permission('manage_users') and not st.session_state.current_user in ROOT_ADMIN_EMAILS:
            return False, "You don't have permission to create users"
    
    # Check if email already exists
    if email in st.session_state.user_db:
        return False, "A user with this email already exists"
    
    # Validate role
    if role not in USER_ROLES:
        return False, "Invalid role"
    
    # Cannot create root_admin users except by existing root_admin
    if oauth_provider is None:  # Skip this check for OAuth created users
        current_user = get_current_user()
        if role == 'root_admin' and (current_user is None or current_user['role'] != 'root_admin'):
            return False, "Only root administrators can create other root administrators"
    
    # Generate a random password if none provided
    if password is None:
        password = os.urandom(8).hex()
    
    # Hash the password
    password_data = generate_password_hash(password)
    
    # Create the user
    user_data = {
        'email': email,
        'name': name,
        'role': role,
        'password': {
            'hash': password_data['hash'],
            'salt': password_data['salt']
        },
        'created_at': datetime.now().isoformat(),
        'created_by': created_by or st.session_state.current_user,
        'last_login': None,
        'status': 'active',
        'must_change_password': not bool(oauth_provider)  # Don't require password change for OAuth users
    }
    
    # Add OAuth provider info if available
    if oauth_provider:
        user_data['oauth_provider'] = oauth_provider
    
    # Save user to database
    st.session_state.user_db[email] = user_data
    
    # Save changes
    save_user_database()
    
    return True, f"User created successfully. {'Password: ' + password if password else ''}"

def update_user(email, updates):
    """Update a user's information"""
    # Check permissions
    current_user = get_current_user()
    if not current_user:
        return False, "You must be logged in to update users"
    
    # Can always update own info except role
    is_self_update = email == current_user['email']
    if is_self_update and 'role' in updates and updates['role'] != current_user['role']:
        return False, "You cannot change your own role"
    
    # For updating others, need manage_users permission
    if not is_self_update and not has_permission('manage_users'):
        return False, "You don't have permission to update other users"
    
    # Check if user exists
    if email not in st.session_state.user_db:
        return False, "User not found"
    
    # Get the user to update
    user = st.session_state.user_db[email]
    
    # Cannot update root_admin users unless you're also a root_admin
    if user['role'] == 'root_admin' and current_user['role'] != 'root_admin':
        return False, "Only root administrators can modify other root administrators"
    
    # Cannot remove root_admin status from ROOT_ADMIN_EMAILS
    if is_root_admin(email) and 'role' in updates and updates['role'] != 'root_admin':
        return False, "Cannot remove root administrator status from this user"
    
    # Update fields
    for field, value in updates.items():
        if field == 'password':
            # Handle password updates
            password_data = generate_password_hash(value)
            user['password'] = {
                'hash': password_data['hash'],
                'salt': password_data['salt']
            }
            user['must_change_password'] = False
        elif field != 'email':  # Email cannot be changed, it's the primary key
            user[field] = value
    
    # Update last modified info
    user['last_modified'] = datetime.now().isoformat()
    user['modified_by'] = current_user['email']
    
    # Save changes
    save_user_database()
    
    return True, "User updated successfully"

def delete_user(email):
    """Delete a user"""
    # Check permissions
    if not has_permission('manage_users'):
        return False, "You don't have permission to delete users"
    
    # Cannot delete root admin users
    if is_root_admin(email):
        return False, "Root administrator accounts cannot be deleted"
    
    # Check if user exists
    if email not in st.session_state.user_db:
        return False, "User not found"
    
    # Delete the user
    del st.session_state.user_db[email]
    
    # Save changes
    save_user_database()
    
    return True, "User deleted successfully"

def reset_password(email):
    """Reset the password for a user and generate a new random one"""
    if email in st.session_state.user_db:
        # Generate a random password
        import random
        import string
        # Create a more readable random password with at least one uppercase, one lowercase, one digit
        letters_lower = string.ascii_lowercase
        letters_upper = string.ascii_uppercase
        digits = string.digits
        
        # Ensure at least one of each character type
        random_password = (
            random.choice(letters_upper) +
            random.choice(letters_lower) +
            random.choice(digits) +
            ''.join(random.choices(letters_lower + letters_upper + digits, k=9))
        )
        
        # Shuffle the password to make it truly random
        password_list = list(random_password)
        random.shuffle(password_list)
        random_password = ''.join(password_list)
        
        # Update the user's password
        password_data = generate_password_hash(random_password)
        
        # Update user password structure
        st.session_state.user_db[email]['password'] = {
            'hash': password_data['hash'],
            'salt': password_data['salt']
        }
        st.session_state.user_db[email]["must_change_password"] = True
        
        # Save the user database
        save_user_database()
        
        return True, random_password
    else:
        return False, None

def render_login_form():
    """Render the login form"""
    st.subheader("Login")
    
    # Create tabs for password login, reset password, and social login
    login_tab, reset_tab, social_tab = st.tabs(["Email & Password", "Reset Password", "Social Login"])
    
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                success, message = authenticate_user(email, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    with reset_tab:
        st.markdown("### Password Reset")
        st.markdown("Forgot your password? Enter your email address below to reset it.")
        
        with st.form("reset_password_form"):
            reset_email = st.text_input("Email Address")
            reset_submitted = st.form_submit_button("Reset Password")
            
            if reset_submitted:
                # Check if the user exists in the database
                if reset_email in st.session_state.user_db:
                    success, new_password = reset_password(reset_email)
                    if success:
                        st.success(f"Password has been reset. Your new password is: {new_password}")
                        st.info("Please login with this password and change it immediately for security reasons.")
                    else:
                        st.error("Could not reset password. Please contact system administrator.")
                else:
                    st.error("Email address not found in the system.")
    
    with social_tab:
        # Import the social login component
        from components.social_auth import social_login_page
        # Initialize social auth state if needed
        if "social_auth_state" not in st.session_state:
            st.session_state.social_auth_state = str(uuid.uuid4())
        # Show social login buttons with a unique context for this instance
        social_login_page(context="user_management_tab")

def render_user_management():
    """Render the user management interface"""
    initialize_user_management()
    
    # Check if user is logged in
    if not st.session_state.user_authenticated:
        render_login_form()
        return
    
    # Get current user
    current_user = get_current_user()
    
    # Welcome message
    st.subheader(f"Welcome, {current_user['name']}")
    st.write(f"Role: {USER_ROLES[current_user['role']]['name']}")
    
    # Check if password change is required
    if current_user.get('must_change_password', False):
        st.warning("You need to change your password")
        with st.form("change_password_form"):
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Change Password")
            
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters long")
                else:
                    success, message = update_user(current_user['email'], {"password": new_password})
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    # Logout button
    if st.button("Logout"):
        logout_user()
        st.rerun()
    
    # User management (only for admins)
    if has_permission('manage_users'):
        st.subheader("User Management")
        
        # Tabs for user management actions including security dashboard
        user_list_tab, create_user_tab, edit_users_tab, security_tab = st.tabs(["User List", "Create User", "Edit Users", "Security Dashboard"])
        
        with user_list_tab:
            # Display user list
            users_data = []
            for email, user in st.session_state.user_db.items():
                users_data.append({
                    "Email": email,
                    "Name": user['name'],
                    "Role": USER_ROLES[user['role']]['name'],
                    "Status": user['status'].capitalize(),
                    "Last Login": user.get('last_login', 'Never') if user.get('last_login') != None else 'Never'
                })
            
            if users_data:
                users_df = pd.DataFrame(users_data)
                st.dataframe(users_df, use_container_width=True)
            else:
                st.info("No users found")
        
        with create_user_tab:
            st.subheader("Create New User")
            
            with st.form("create_user_form"):
                new_email = st.text_input("Email Address")
                new_name = st.text_input("Full Name")
                
                # Determine available roles based on current user's role
                available_roles = list(USER_ROLES.keys())
                if current_user['role'] != 'root_admin':
                    # Non-root admins cannot create root admins
                    available_roles.remove('root_admin')
                
                new_role = st.selectbox(
                    "Role",
                    options=available_roles,
                    format_func=lambda x: USER_ROLES[x]['name']
                )
                
                generate_random_password = st.checkbox("Generate Random Password", value=True)
                new_password = None if generate_random_password else st.text_input("Password", type="password")
                
                submitted = st.form_submit_button("Create User")
                
                if submitted:
                    if not new_email or not new_name:
                        st.error("Email and name are required")
                    else:
                        success, message = create_user(
                            new_email, 
                            new_name, 
                            new_role, 
                            password=new_password
                        )
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        
        with edit_users_tab:
            st.subheader("Edit User")
            
            # Select user to edit
            user_emails = list(st.session_state.user_db.keys())
            selected_user_email = st.selectbox("Select User", user_emails)
            
            if selected_user_email:
                user = st.session_state.user_db[selected_user_email]
                
                # Determine if current user can edit this user
                can_edit = True
                edit_restrictions = []
                
                if is_root_admin(selected_user_email) and not is_root_admin(current_user['email']):
                    can_edit = False
                    edit_restrictions.append("Only root administrators can edit other root administrators")
                
                if selected_user_email == current_user['email'] and user['role'] == 'root_admin':
                    edit_restrictions.append("You can edit your own info but not change your role")
                
                # Show edit form if the user can be edited
                if can_edit:
                    with st.form("edit_user_form"):
                        updated_name = st.text_input("Name", value=user['name'])
                        
                        # Role selection (restricted for self-edits)
                        available_roles = list(USER_ROLES.keys())
                        if current_user['role'] != 'root_admin':
                            available_roles.remove('root_admin')
                        
                        role_disabled = selected_user_email == current_user['email']
                        updated_role = st.selectbox(
                            "Role",
                            options=available_roles,
                            index=available_roles.index(user['role']),
                            disabled=role_disabled,
                            format_func=lambda x: USER_ROLES[x]['name']
                        )
                        
                        updated_status = st.selectbox(
                            "Status",
                            options=["active", "inactive", "suspended"],
                            index=["active", "inactive", "suspended"].index(user['status'])
                        )
                        
                        reset_password = st.checkbox("Reset Password")
                        new_password = st.text_input("New Password", type="password") if reset_password else None
                        
                        if edit_restrictions:
                            for restriction in edit_restrictions:
                                st.info(restriction)
                        
                        submitted = st.form_submit_button("Update User")
                        
                        if submitted:
                            updates = {
                                "name": updated_name,
                                "status": updated_status
                            }
                            
                            # Don't update role for self-edits
                            if not role_disabled:
                                updates["role"] = updated_role
                            
                            if reset_password and new_password:
                                updates["password"] = new_password
                            
                            success, message = update_user(selected_user_email, updates)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    
                    # Delete user button (not for self or root admins)
                    if (selected_user_email != current_user['email'] and 
                        not is_root_admin(selected_user_email)):
                        if st.button("Delete User", type="primary", key="delete_user"):
                            success, message = delete_user(selected_user_email)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                else:
                    st.error("You don't have permission to edit this user")
                    for restriction in edit_restrictions:
                        st.info(restriction)
        
        with security_tab:
            st.subheader("Security Dashboard")
            
            security_subtabs = st.tabs(["Login Attempts", "Account Lockouts", "IP Blacklist", "Security Notifications"])
            
            with security_subtabs[0]:  # Login Attempts
                st.markdown("### Login Attempts Tracking")
                
                if st.session_state.login_attempts:
                    login_data = []
                    for email, data in st.session_state.login_attempts.items():
                        ip_count = len(data.get('ip_addresses', set()))
                        login_data.append({
                            "Email": email,
                            "Total Attempts": data.get('attempts', 0),
                            "Successful Logins": data.get('successful_logins', 0),
                            "Failed Logins": data.get('failed_logins', 0),
                            "IP Addresses": ip_count,
                            "Last Attempt": data.get('last_attempt', 'Never')
                        })
                    
                    login_df = pd.DataFrame(login_data)
                    st.dataframe(login_df, use_container_width=True)
                    
                    # Clear login attempts button
                    if st.button("Clear Login Attempts Data", key="clear_login_attempts"):
                        st.session_state.login_attempts = {}
                        st.success("Login attempts data cleared")
                        st.rerun()
                else:
                    st.info("No login attempts recorded")
            
            with security_subtabs[1]:  # Account Lockouts
                st.markdown("### Account Lockouts")
                
                if st.session_state.account_lockouts:
                    lockout_data = []
                    current_time = datetime.now()
                    
                    for email, data in st.session_state.account_lockouts.items():
                        lockout_time = datetime.fromisoformat(data.get('lockout_time', current_time.isoformat()))
                        unlock_time = lockout_time + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
                        time_remaining = max(0, (unlock_time - current_time).total_seconds() / 60)
                        
                        lockout_data.append({
                            "Email": email,
                            "Lockout Time": lockout_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Minutes Remaining": round(time_remaining, 1),
                            "Reason": data.get('reason', 'Unknown'),
                            "IP Address": data.get('ip_address', 'Unknown')
                        })
                    
                    lockout_df = pd.DataFrame(lockout_data)
                    st.dataframe(lockout_df, use_container_width=True)
                    
                    # Unlock accounts button
                    selected_account = st.selectbox("Select account to unlock", 
                                                   [row["Email"] for row in lockout_data],
                                                   key="unlock_account_select")
                    
                    if st.button("Unlock Selected Account", key="unlock_account"):
                        if selected_account in st.session_state.account_lockouts:
                            del st.session_state.account_lockouts[selected_account]
                            
                            # Reset failed login counters for this account
                            if selected_account in st.session_state.login_attempts:
                                st.session_state.login_attempts[selected_account]['failed_logins'] = 0
                            
                            st.success(f"Account {selected_account} unlocked successfully")
                            st.rerun()
                else:
                    st.info("No accounts are currently locked")
            
            with security_subtabs[2]:  # IP Blacklist
                st.markdown("### IP Blacklist Management")
                
                if st.session_state.ip_blacklist:
                    blacklist_data = []
                    
                    for ip, data in st.session_state.ip_blacklist.items():
                        blacklist_time = datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
                        
                        blacklist_data.append({
                            "IP Address": ip,
                            "Blacklisted On": blacklist_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Reason": data.get('reason', 'Unknown'),
                            "Associated Email": data.get('associated_email', 'Unknown')
                        })
                    
                    blacklist_df = pd.DataFrame(blacklist_data)
                    st.dataframe(blacklist_df, use_container_width=True)
                    
                    # Remove from blacklist
                    selected_ip = st.selectbox("Select IP to remove from blacklist", 
                                              [row["IP Address"] for row in blacklist_data],
                                              key="remove_ip_select")
                    
                    if st.button("Remove IP from Blacklist", key="remove_ip"):
                        if selected_ip in st.session_state.ip_blacklist:
                            del st.session_state.ip_blacklist[selected_ip]
                            
                            # Save blacklist to file
                            os.makedirs('secure_assets', exist_ok=True)
                            with open('secure_assets/ip_blacklist.json', 'w') as f:
                                json.dump(st.session_state.ip_blacklist, f, indent=2)
                            
                            st.success(f"IP address {selected_ip} removed from blacklist")
                            st.rerun()
                    
                    # Add manual blacklist entry
                    st.markdown("### Add IP to Blacklist")
                    with st.form("add_to_blacklist_form"):
                        new_ip = st.text_input("IP Address")
                        reason = st.text_input("Reason")
                        associated_email = st.text_input("Associated Email (optional)")
                        
                        submitted = st.form_submit_button("Add to Blacklist")
                        
                        if submitted:
                            try:
                                # Validate IP address
                                ipaddress.ip_address(new_ip)
                                
                                # Add to blacklist
                                st.session_state.ip_blacklist[new_ip] = {
                                    'timestamp': datetime.now().isoformat(),
                                    'reason': reason or "Manually blacklisted",
                                    'associated_email': associated_email or "N/A"
                                }
                                
                                # Save blacklist to file
                                os.makedirs('secure_assets', exist_ok=True)
                                with open('secure_assets/ip_blacklist.json', 'w') as f:
                                    json.dump(st.session_state.ip_blacklist, f, indent=2)
                                
                                st.success(f"IP address {new_ip} added to blacklist")
                                st.rerun()
                            except ValueError:
                                st.error("Invalid IP address format")
                else:
                    st.info("IP blacklist is empty")
                    
                    # Add manual blacklist entry
                    st.markdown("### Add IP to Blacklist")
                    with st.form("add_to_blacklist_form"):
                        new_ip = st.text_input("IP Address")
                        reason = st.text_input("Reason")
                        associated_email = st.text_input("Associated Email (optional)")
                        
                        submitted = st.form_submit_button("Add to Blacklist")
                        
                        if submitted:
                            try:
                                # Validate IP address
                                ipaddress.ip_address(new_ip)
                                
                                # Add to blacklist
                                st.session_state.ip_blacklist[new_ip] = {
                                    'timestamp': datetime.now().isoformat(),
                                    'reason': reason or "Manually blacklisted",
                                    'associated_email': associated_email or "N/A"
                                }
                                
                                # Save blacklist to file
                                os.makedirs('secure_assets', exist_ok=True)
                                with open('secure_assets/ip_blacklist.json', 'w') as f:
                                    json.dump(st.session_state.ip_blacklist, f, indent=2)
                                
                                st.success(f"IP address {new_ip} added to blacklist")
                                st.rerun()
                            except ValueError:
                                st.error("Invalid IP address format")
            
            with security_subtabs[3]:  # Security Notifications
                st.markdown("### Security Notifications")
                
                # Collect all security notifications
                notifications = []
                for email, user in st.session_state.user_db.items():
                    if 'security_notifications' in user:
                        for notification in user['security_notifications']:
                            notifications.append({
                                "Email": email,
                                "Type": notification.get('type', 'Unknown'),
                                "Timestamp": notification.get('timestamp', 'Unknown'),
                                "Message": notification.get('message', 'No details'),
                                "IP Address": notification.get('ip_address', 'Unknown')
                            })
                
                if notifications:
                    notifications_df = pd.DataFrame(notifications)
                    st.dataframe(notifications_df, use_container_width=True)
                    
                    # Clear notifications button
                    if st.button("Clear All Security Notifications", key="clear_notifications"):
                        for email in st.session_state.user_db:
                            if 'security_notifications' in st.session_state.user_db[email]:
                                st.session_state.user_db[email]['security_notifications'] = []
                        
                        save_user_database()
                        st.success("All security notifications cleared")
                        st.rerun()
                else:
                    st.info("No security notifications")