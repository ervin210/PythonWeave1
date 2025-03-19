import streamlit as st
import pandas as pd
import os
import json
import hashlib
import uuid
from datetime import datetime

# Define the root admin emails (protected from changes)
ROOT_ADMIN_EMAILS = [
    "ervin210@icloud.com",
    "radosavlevici.ervin@gmail.com"
]

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
    
    # Use a secure hashing algorithm with the salt
    key = hashlib.pbkdf2_hmac(
        'sha256',  # Hash algorithm
        password.encode('utf-8'),  # Convert password to bytes
        salt,  # Salt
        100000,  # Number of iterations (higher is more secure but slower)
        dklen=128  # Length of the derived key
    )
    
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
                # Create temporary random password for root admins
                # They should change this on first login
                temp_password = hashlib.sha256(email.encode()).hexdigest()[:12]
                password_data = generate_password_hash(temp_password)
                
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

def authenticate_user(email, password, oauth=False):
    """Authenticate a user with email and password or via OAuth"""
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

def render_login_form():
    """Render the login form"""
    st.subheader("Login")
    
    # Create tabs for password login and social login
    login_tab, social_tab = st.tabs(["Email & Password", "Social Login"])
    
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
    
    with social_tab:
        # Import the social login component
        from components.social_auth import social_login_page
        # Initialize social auth state if needed
        if "social_auth_state" not in st.session_state:
            st.session_state.social_auth_state = str(uuid.uuid4())
        # Show social login buttons
        social_login_page()

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
        
        # Tabs for user management actions
        user_list_tab, create_user_tab, edit_users_tab = st.tabs(["User List", "Create User", "Edit Users"])
        
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