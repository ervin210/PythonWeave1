import streamlit as st
import stripe
import pandas as pd
import os
import json
from datetime import datetime

# Define the subscription plans
SUBSCRIPTION_PLANS = {
    "basic": {
        "name": "Basic Plan",
        "price": 9.99,
        "features": [
            "Access to Quantum Assistant",
            "Basic W&B integration",
            "Standard support"
        ],
        "description": "Perfect for individuals getting started with quantum computing"
    },
    "professional": {
        "name": "Professional Plan",
        "price": 29.99,
        "features": [
            "Everything in Basic",
            "Advanced quantum circuit analysis",
            "Priority support",
            "Unlimited W&B project integrations"
        ],
        "description": "Ideal for professionals and researchers"
    },
    "enterprise": {
        "name": "Enterprise Plan",
        "price": 99.99,
        "features": [
            "Everything in Professional",
            "Cross-platform connector",
            "Dedicated customer support",
            "Custom integrations",
            "Batch operations"
        ],
        "description": "Complete solution for organizations"
    }
}

def initialize_subscription_state():
    """Initialize subscription-related session state variables"""
    if "subscription_active" not in st.session_state:
        st.session_state.subscription_active = False
    if "current_plan" not in st.session_state:
        st.session_state.current_plan = None
    if "subscription_end_date" not in st.session_state:
        st.session_state.subscription_end_date = None
    if "payment_info" not in st.session_state:
        st.session_state.payment_info = {}
    if "subscription_history" not in st.session_state:
        st.session_state.subscription_history = []

def activate_subscription(plan_id):
    """Activate a subscription for the user"""
    # In a real implementation, this would connect to a payment processor
    # and validate the payment before activating
    st.session_state.subscription_active = True
    st.session_state.current_plan = plan_id
    
    # Simulate subscription dates (in a real app, this would come from payment processor)
    import datetime
    start_date = datetime.datetime.now()
    end_date = start_date + datetime.timedelta(days=30)  # 30-day subscription
    
    # Record the subscription
    subscription_record = {
        "plan_id": plan_id,
        "plan_name": SUBSCRIPTION_PLANS[plan_id]["name"],
        "price": SUBSCRIPTION_PLANS[plan_id]["price"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "transaction_id": f"txn_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    }
    
    st.session_state.subscription_end_date = end_date
    st.session_state.subscription_history.append(subscription_record)
    
    return subscription_record

def cancel_subscription():
    """Cancel the current subscription"""
    st.session_state.subscription_active = False
    st.session_state.current_plan = None
    st.session_state.subscription_end_date = None
    
    return True

def check_subscription_status():
    """Check if the user has an active subscription"""
    if not st.session_state.subscription_active:
        return False
    
    # Check if subscription has expired
    if st.session_state.subscription_end_date:
        import datetime
        current_date = datetime.datetime.now()
        if current_date > st.session_state.subscription_end_date:
            st.session_state.subscription_active = False
            st.session_state.current_plan = None
            return False
    
    return st.session_state.subscription_active

def subscription_manager():
    """Main subscription management component"""
    st.title("Subscription Management")
    
    # Initialize subscription state
    initialize_subscription_state()
    
    # Check current subscription status
    is_subscribed = check_subscription_status()
    
    # Display current subscription info if active
    if is_subscribed:
        st.success(f"You have an active subscription to the {SUBSCRIPTION_PLANS[st.session_state.current_plan]['name']}")
        
        # Show subscription details
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Subscription Details")
            st.write(f"**Plan:** {SUBSCRIPTION_PLANS[st.session_state.current_plan]['name']}")
            st.write(f"**Price:** ${SUBSCRIPTION_PLANS[st.session_state.current_plan]['price']}/month")
            st.write(f"**Expires:** {st.session_state.subscription_end_date.strftime('%Y-%m-%d')}")
        
        with col2:
            st.subheader("Plan Features")
            for feature in SUBSCRIPTION_PLANS[st.session_state.current_plan]['features']:
                st.write(f"✓ {feature}")
        
        # Manage subscription
        st.subheader("Manage Subscription")
        if st.button("Cancel Subscription"):
            cancel_success = cancel_subscription()
            if cancel_success:
                st.warning("Your subscription has been canceled.")
                st.rerun()
    
    # Show subscription plans if no active subscription
    else:
        st.header("Choose a Subscription Plan")
        st.markdown("Subscribe to unlock advanced features of the Quantum AI Assistant")
        
        # Display subscription plans in columns
        cols = st.columns(len(SUBSCRIPTION_PLANS))
        
        for i, (plan_id, plan_info) in enumerate(SUBSCRIPTION_PLANS.items()):
            with cols[i]:
                st.subheader(plan_info["name"])
                st.write(f"**${plan_info['price']}/month**")
                st.write(plan_info["description"])
                
                # List features
                for feature in plan_info["features"]:
                    st.write(f"✓ {feature}")
                
                # Subscribe button
                if st.button(f"Subscribe to {plan_info['name']}", key=f"btn_{plan_id}"):
                    st.session_state.selected_plan = plan_id
                    # Show payment form
                    payment_success = payment_form(plan_id)
                    if payment_success:
                        # Activate subscription
                        subscription = activate_subscription(plan_id)
                        st.success(f"Thank you for subscribing to {plan_info['name']}!")
                        st.rerun()

def payment_form(plan_id):
    """Display payment form and process payment"""
    st.subheader(f"Complete your {SUBSCRIPTION_PLANS[plan_id]['name']} subscription")
    
    # Create payment form
    with st.form("payment_form"):
        st.write(f"Total: ${SUBSCRIPTION_PLANS[plan_id]['price']}/month")
        
        # Payment details
        st.subheader("Payment Information")
        card_number = st.text_input("Card Number", placeholder="4242 4242 4242 4242")
        col1, col2 = st.columns(2)
        with col1:
            expiry = st.text_input("Expiry Date (MM/YY)", placeholder="12/25")
        with col2:
            cvc = st.text_input("CVC", placeholder="123")
        
        # Billing address
        st.subheader("Billing Address")
        name = st.text_input("Full Name", placeholder="John Doe")
        email = st.text_input("Email", placeholder="john@example.com")
        address = st.text_input("Address", placeholder="123 Main St")
        city = st.text_input("City", placeholder="New York")
        col3, col4 = st.columns(2)
        with col3:
            country = st.selectbox("Country", ["United States", "United Kingdom", "Canada", "Australia", "Other"])
        with col4:
            postal_code = st.text_input("Postal Code", placeholder="10001")
        
        # Terms and conditions
        agree = st.checkbox("I agree to the terms and conditions")
        
        # Submit button
        submitted = st.form_submit_button("Complete Payment")
        
        if submitted:
            if not agree:
                st.error("You must agree to the terms and conditions")
                return False
            
            if not all([card_number, expiry, cvc, name, email, address, city, postal_code]):
                st.error("Please fill in all required fields")
                return False
            
            # In a real implementation, this would connect to Stripe or another payment processor
            # For demonstration purposes, we'll consider the payment successful
            
            # Store payment info in session state (in a real app, this would be stored securely)
            st.session_state.payment_info = {
                "name": name,
                "email": email,
                "last4": card_number[-4:] if len(card_number) >= 4 else "****",
                "brand": "Visa",  # Simplified for demo
                "address": {
                    "line1": address,
                    "city": city,
                    "country": country,
                    "postal_code": postal_code
                }
            }
            
            return True
    
    return False

def subscription_history():
    """Display subscription history"""
    st.header("Subscription History")
    
    if not st.session_state.subscription_history:
        st.info("No subscription history available")
        return
    
    # Create a dataframe from subscription history
    history_df = pd.DataFrame(st.session_state.subscription_history)
    
    # Display as a table
    st.dataframe(history_df)
    
    # Option to export history
    if st.button("Export Subscription History"):
        # In a real app, this would generate a PDF or CSV
        st.success("Subscription history exported successfully")
    
def render_subscription_sidebar():
    """Add subscription status to sidebar"""
    is_subscribed = check_subscription_status()
    
    if is_subscribed:
        st.sidebar.success(f"Active: {SUBSCRIPTION_PLANS[st.session_state.current_plan]['name']}")
    else:
        st.sidebar.warning("No active subscription")
        if st.sidebar.button("Subscribe Now"):
            st.session_state.current_page = "subscription"

# Function to check if feature is available in current plan
def is_feature_available(feature_level):
    """Check if a feature is available in the current subscription plan
    
    Args:
        feature_level: "basic", "professional", or "enterprise"
    
    Returns:
        bool: Whether the feature is available
    """
    plan_levels = {
        "basic": 0,
        "professional": 1,
        "enterprise": 2
    }
    
    # If not subscribed, only basic features are available
    if not st.session_state.subscription_active:
        current_level = -1
    else:
        current_level = plan_levels.get(st.session_state.current_plan, -1)
    
    required_level = plan_levels.get(feature_level, 0)
    
    return current_level >= required_level