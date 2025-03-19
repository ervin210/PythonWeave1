import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import hashlib
import base64
import json
from datetime import datetime
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

# Quantum Security Component - provides quantum-enhanced security features

def quantum_security():
    """
    Quantum Security Tools for Enterprise-level protection
    """
    st.title("Quantum Security Center")
    st.markdown("""
    This component provides advanced quantum-enabled security features for enterprise deployments.
    Use these tools to analyze and enhance your system's security posture in the post-quantum era.
    """)
    
    # Security feature tabs
    security_tab1, security_tab2, security_tab3 = st.tabs([
        "Quantum-Safe Encryption Analysis", 
        "Quantum Risk Assessment", 
        "Quantum Random Number Generator"
    ])
    
    with security_tab1:
        quantum_safe_encryption_analyzer()
    
    with security_tab2:
        quantum_risk_assessment()
    
    with security_tab3:
        quantum_random_generator()

def quantum_safe_encryption_analyzer():
    """Tool to analyze and compare classical vs quantum-resistant encryption"""
    st.header("Quantum-Safe Encryption Analysis")
    st.markdown("""
    Compare classical encryption methods with quantum-resistant algorithms to understand
    your vulnerabilities in a post-quantum computing world.
    """)
    
    # Encryption comparison
    st.subheader("Encryption Methods Comparison")
    
    # Create comparison data
    encryption_data = {
        "Algorithm Type": [
            "RSA-2048", 
            "ECC-256", 
            "AES-256", 
            "Lattice-based (NTRU)", 
            "Hash-based (SPHINCS+)", 
            "Code-based (McEliece)", 
            "Multivariate (Rainbow)"
        ],
        "Classical Security (bits)": [112, 128, 256, 128, 128, 128, 128],
        "Quantum Security (bits)": [0, 0, 128, 128, 128, 128, 128],
        "Key Size (bytes)": [256, 32, 32, 1025, 1088, 1MB_placeholder, 150000],
        "Category": [
            "Classical", 
            "Classical", 
            "Classical (Symmetric)", 
            "Post-Quantum", 
            "Post-Quantum", 
            "Post-Quantum", 
            "Post-Quantum"
        ]
    }
    
    # Create DataFrame
    encryption_df = pd.DataFrame(encryption_data)
    encryption_df["Key Size (bytes)"] = [256, 32, 32, 1025, 1088, 1024 * 1024, 150000]  # Fix placeholder
    
    # Display comparison table
    st.dataframe(encryption_df)
    
    # Visualize security strength
    st.subheader("Encryption Strength Against Quantum Attacks")
    
    # Create a bar chart to compare classical vs quantum security
    fig = px.bar(
        encryption_df, 
        x="Algorithm Type", 
        y=["Classical Security (bits)", "Quantum Security (bits)"],
        barmode="group",
        color_discrete_sequence=["#00AC8C", "#F8A248"],
        title="Classical vs. Quantum Security Strength"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show key size comparison
    st.subheader("Key Size Comparison")
    
    # Create a log scale bar chart for key sizes
    fig2 = px.bar(
        encryption_df, 
        x="Algorithm Type", 
        y="Key Size (bytes)",
        color="Category",
        log_y=True,  # Log scale for better visualization
        title="Key Size Comparison (log scale)",
        color_discrete_map={
            "Classical": "#00AC8C",
            "Post-Quantum": "#F8A248",
            "Classical (Symmetric)": "#57D2BD"
        }
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Recommendations section
    st.subheader("Security Recommendations")
    st.markdown("""
    1. **Start transitioning to quantum-resistant algorithms** for sensitive data that needs to remain secure for years to come
    2. **Use hybrid approaches** combining classical and post-quantum cryptography for critical systems
    3. **Monitor NIST post-quantum cryptography standardization** efforts and implement standards as they become available
    4. **Conduct regular security audits** that include quantum threat assessments
    5. **Develop a quantum-safe migration strategy** that prioritizes your most sensitive systems
    """)

def quantum_risk_assessment():
    """Tool to assess quantum computing risks to your systems"""
    st.header("Quantum Risk Assessment")
    st.markdown("""
    Evaluate your organization's vulnerability to quantum computing attacks and get
    recommendations for mitigating these risks.
    """)
    
    # User input for risk assessment
    st.subheader("Assess Your Quantum Risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # System information
        st.markdown("#### System Information")
        encryption_types = st.multiselect(
            "What encryption methods do you currently use?",
            ["RSA", "ECC", "AES", "3DES", "Diffie-Hellman", "DSA", "ECDSA", "NTRU", "McEliece", "SPHINCS+"],
            default=["RSA", "AES"]
        )
        
        data_sensitivity = st.select_slider(
            "Data Sensitivity Level",
            options=["Low", "Medium", "High", "Critical"],
            value="Medium"
        )
        
        data_longevity = st.slider(
            "How long must your data remain secure? (years)",
            min_value=1, 
            max_value=50,
            value=10
        )
    
    with col2:
        # Risk factors
        st.markdown("#### Risk Factors")
        
        regulatory_compliance = st.multiselect(
            "Regulatory Requirements",
            ["GDPR", "HIPAA", "PCI DSS", "CCPA", "NIST 800-53", "ISO 27001", "None"],
            default=["GDPR"]
        )
        
        deployment_model = st.radio(
            "Deployment Model",
            ["On-premises", "Hybrid Cloud", "Public Cloud", "Multi-Cloud"],
            index=1
        )
        
        security_resources = st.select_slider(
            "Security Resource Level",
            options=["Minimal", "Moderate", "Substantial", "Extensive"],
            value="Moderate"
        )
    
    # Calculate quantum risk score
    if st.button("Calculate Risk Assessment"):
        # Basic algorithm to calculate risk (would be more sophisticated in a real app)
        risk_score = 0
        
        # Encryption risk factors
        for enc in encryption_types:
            if enc in ["RSA", "ECC", "Diffie-Hellman", "DSA", "ECDSA"]:
                risk_score += 20
            elif enc in ["NTRU", "McEliece", "SPHINCS+"]:
                risk_score -= 15
            elif enc == "AES":
                if data_longevity > 15:  # Long-term concerns even for AES
                    risk_score += 5
        
        # Data sensitivity impact
        sensitivity_map = {"Low": 5, "Medium": 10, "High": 20, "Critical": 30}
        risk_score += sensitivity_map[data_sensitivity]
        
        # Data longevity impact
        if data_longevity > 10:
            risk_score += min(30, (data_longevity - 10) * 3)  # Higher risk for longer term needs
        
        # Regulatory compliance impact (more regulations = more potential impact)
        if "None" not in regulatory_compliance:
            risk_score += len(regulatory_compliance) * 5
        
        # Deployment model impact
        model_risk = {"On-premises": 5, "Hybrid Cloud": 10, "Public Cloud": 15, "Multi-Cloud": 20}
        risk_score += model_risk[deployment_model]
        
        # Security resources impact (inverse relationship)
        resource_risk = {"Minimal": 20, "Moderate": 10, "Substantial": 5, "Extensive": 0}
        risk_score += resource_risk[security_resources]
        
        # Normalize risk score (0-100)
        risk_score = min(100, max(0, risk_score))
        
        # Display risk score
        st.subheader("Quantum Risk Assessment Results")
        
        # Risk gauge
        risk_color = "green" if risk_score < 33 else "orange" if risk_score < 66 else "red"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            title = {'text': "Quantum Risk Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "lightyellow"},
                    {'range': [66, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk breakdown
        st.subheader("Risk Breakdown")
        
        # Determine risk areas
        risk_areas = {
            "Encryption Vulnerability": 0,
            "Data Sensitivity Risk": 0,
            "Time Horizon Risk": 0,
            "Compliance Impact": 0,
            "Deployment Risk": 0,
            "Resource Adequacy": 0
        }
        
        # Calculate individual risk components (simplified version)
        # In a real app, this would be more sophisticated and precise
        high_risk_count = 0
        
        # Encryption risk
        classic_crypto = sum(1 for enc in encryption_types if enc in ["RSA", "ECC", "Diffie-Hellman", "DSA", "ECDSA"])
        quantum_safe = sum(1 for enc in encryption_types if enc in ["NTRU", "McEliece", "SPHINCS+"])
        if classic_crypto > 0:
            enc_risk = 70 - (quantum_safe * 20)
            risk_areas["Encryption Vulnerability"] = max(0, min(100, enc_risk))
            if risk_areas["Encryption Vulnerability"] > 50:
                high_risk_count += 1
        
        # Data sensitivity risk
        risk_areas["Data Sensitivity Risk"] = {"Low": 20, "Medium": 40, "High": 70, "Critical": 90}[data_sensitivity]
        if risk_areas["Data Sensitivity Risk"] > 50:
            high_risk_count += 1
        
        # Time horizon risk
        if data_longevity <= 5:
            risk_areas["Time Horizon Risk"] = 30
        elif data_longevity <= 10:
            risk_areas["Time Horizon Risk"] = 50
        elif data_longevity <= 20:
            risk_areas["Time Horizon Risk"] = 70
        else:
            risk_areas["Time Horizon Risk"] = 90
            high_risk_count += 1
        
        # Compliance impact
        if "None" in regulatory_compliance or not regulatory_compliance:
            risk_areas["Compliance Impact"] = 20
        else:
            risk_areas["Compliance Impact"] = min(90, len(regulatory_compliance) * 20)
            if risk_areas["Compliance Impact"] > 50:
                high_risk_count += 1
        
        # Deployment risk
        risk_areas["Deployment Risk"] = {"On-premises": 30, "Hybrid Cloud": 50, "Public Cloud": 70, "Multi-Cloud": 80}[deployment_model]
        if risk_areas["Deployment Risk"] > 60:
            high_risk_count += 1
        
        # Resource adequacy
        risk_areas["Resource Adequacy"] = {"Minimal": 80, "Moderate": 60, "Substantial": 30, "Extensive": 10}[security_resources]
        if risk_areas["Resource Adequacy"] > 60:
            high_risk_count += 1
        
        # Display risk breakdown
        risk_df = pd.DataFrame({
            "Risk Area": list(risk_areas.keys()),
            "Risk Score": list(risk_areas.values())
        })
        
        # Sort by risk level
        risk_df = risk_df.sort_values("Risk Score", ascending=False).reset_index(drop=True)
        
        # Show as horizontal bar chart
        fig2 = px.bar(
            risk_df,
            x="Risk Score",
            y="Risk Area",
            orientation='h',
            color="Risk Score",
            color_continuous_scale=["green", "yellow", "red"],
            title="Risk Breakdown by Area"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Risk matrix
        st.subheader("Risk Matrix")
        
        # Create a risk matrix with impact vs likelihood
        impact_levels = ["Low", "Medium", "High", "Critical"]
        likelihood_levels = ["Unlikely", "Possible", "Likely", "Almost Certain"]
        
        # Determine overall likelihood based on timeline
        if data_longevity <= 5:
            likelihood = "Unlikely"
        elif data_longevity <= 10:
            likelihood = "Possible"
        elif data_longevity <= 20:
            likelihood = "Likely"
        else:
            likelihood = "Almost Certain"
        
        # Determine overall impact based on data sensitivity
        impact = data_sensitivity
        
        # Create risk matrix
        risk_matrix = np.zeros((4, 4))
        
        # Fill matrix with risk levels (1-16)
        for i in range(4):
            for j in range(4):
                risk_matrix[i, j] = (i + 1) * (j + 1)
        
        # Create a heatmap
        fig3 = px.imshow(
            risk_matrix,
            x=likelihood_levels,
            y=impact_levels,
            color_continuous_scale=["green", "yellow", "orange", "red"],
            title="Risk Matrix: Impact vs. Likelihood"
        )
        
        # Mark the position for this assessment
        impact_index = impact_levels.index(impact)
        likelihood_index = likelihood_levels.index(likelihood)
        
        # Add a marker for the current position
        fig3.add_annotation(
            x=likelihood,
            y=impact,
            text="You are here",
            showarrow=True,
            arrowhead=1,
            arrowcolor="black",
            arrowsize=1.5,
            font=dict(color="black", size=14)
        )
        
        # Customize axis labels
        fig3.update_layout(
            xaxis_title="Likelihood of Quantum Threat",
            yaxis_title="Impact to Organization",
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Recommendations based on risk level
        st.subheader("Recommendations")
        
        recommendation_priority = []
        
        # Provide tailored recommendations based on risk areas
        if risk_areas["Encryption Vulnerability"] > 50:
            recommendation_priority.append({
                "area": "Encryption",
                "risk": risk_areas["Encryption Vulnerability"],
                "recommendation": "Transition to quantum-resistant encryption algorithms for sensitive data and communications."
            })
        
        if risk_areas["Time Horizon Risk"] > 60:
            recommendation_priority.append({
                "area": "Data Longevity",
                "risk": risk_areas["Time Horizon Risk"],
                "recommendation": "Implement crypto-agility frameworks to enable rapid transition between cryptographic algorithms."
            })
        
        if risk_areas["Compliance Impact"] > 50:
            recommendation_priority.append({
                "area": "Regulatory Compliance",
                "risk": risk_areas["Compliance Impact"],
                "recommendation": "Develop a quantum-readiness compliance program aligned with your regulatory requirements."
            })
        
        if risk_areas["Resource Adequacy"] > 60:
            recommendation_priority.append({
                "area": "Security Resources",
                "risk": risk_areas["Resource Adequacy"],
                "recommendation": "Invest in training and resources specifically focused on post-quantum cryptography and security."
            })
        
        if risk_areas["Deployment Risk"] > 60:
            recommendation_priority.append({
                "area": "Deployment Model",
                "risk": risk_areas["Deployment Risk"],
                "recommendation": "Create quantum-safe boundaries between different deployment environments and enforce strong isolation."
            })
        
        # Add general recommendations
        general_recommendations = [
            "Conduct a detailed quantum-vulnerable cryptography inventory across your entire organization",
            "Develop a quantum-safe migration roadmap with clear timelines and priorities",
            "Implement a secure key management system that can support post-quantum algorithms",
            "Monitor the quantum computing threat landscape through specialized threat intelligence",
            "Participate in quantum-safe standards development and industry working groups"
        ]
        
        # Sort recommendations by risk level
        recommendation_priority.sort(key=lambda x: x["risk"], reverse=True)
        
        # Display prioritized recommendations
        st.markdown("#### Priority Actions")
        for i, rec in enumerate(recommendation_priority):
            st.markdown(f"**{i+1}. {rec['area']} ({rec['risk']}% risk)**: {rec['recommendation']}")
        
        st.markdown("#### General Recommendations")
        for i, rec in enumerate(general_recommendations):
            st.markdown(f"- {rec}")
        
        # Generate a report
        st.subheader("Download Assessment Report")
        
        report_data = {
            "assessment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_risk_score": int(risk_score),
            "risk_areas": risk_areas,
            "high_risk_areas": high_risk_count,
            "impact_level": impact,
            "likelihood_level": likelihood,
            "recommendations": recommendation_priority,
            "general_recommendations": general_recommendations,
            "inputs": {
                "encryption_types": encryption_types,
                "data_sensitivity": data_sensitivity,
                "data_longevity": data_longevity,
                "regulatory_compliance": regulatory_compliance,
                "deployment_model": deployment_model,
                "security_resources": security_resources
            }
        }
        
        # Convert to JSON for download
        report_json = json.dumps(report_data, indent=2)
        report_bytes = report_json.encode()
        
        st.download_button(
            label="Download Assessment Report (JSON)",
            data=report_bytes,
            file_name=f"quantum_risk_assessment_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
        
        # Also offer PDF report (placeholder - would require more implementation)
        st.markdown("*For a detailed PDF report with executive summary and visualization, upgrade to Enterprise tier*")

def quantum_random_generator():
    """Quantum Random Number Generator tool"""
    st.header("Quantum Random Number Generator")
    st.markdown("""
    Generate true quantum random numbers for cryptographic purposes using
    quantum circuit simulation. Unlike classical pseudo-random number generators,
    quantum randomness is based on inherent quantum mechanical phenomena.
    """)
    
    st.subheader("Generate Quantum Random Numbers")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        num_bits = st.slider("Number of random bits", 1, 64, 32)
        num_samples = st.number_input("Number of samples", 1, 100, 1)
        output_format = st.radio(
            "Output format",
            ["Binary", "Decimal", "Hexadecimal"],
            index=2
        )
    
    with col2:
        st.markdown("#### About Quantum Randomness")
        st.markdown("""
        Quantum random numbers are generated using quantum 
        mechanical phenomena, which are truly random and 
        unpredictable according to quantum physics.
        
        This tool uses quantum circuit simulation to 
        demonstrate true quantum randomness.
        """)
    
    if st.button("Generate Quantum Random Numbers"):
        with st.spinner("Generating quantum random numbers..."):
            # Generate and display the random numbers
            random_values = []
            circuits = []
            
            # Create a quantum circuit for demonstration
            if num_bits <= 32:  # Reasonable size for visualization
                demo_circuit = QuantumCircuit(num_bits)
                demo_circuit.h(range(num_bits))  # Apply Hadamard to all qubits
                demo_circuit.measure_all()
                
                # Show the circuit
                circuit_img = demo_circuit.draw(output="mpl")
                st.pyplot(circuit_img)
            
            # Generate the random numbers
            for _ in range(num_samples):
                # Create a quantum circuit
                qc = QuantumCircuit(num_bits)
                
                # Apply Hadamard gate to each qubit to create superposition
                for i in range(num_bits):
                    qc.h(i)
                
                # Measure all qubits
                qc.measure_all()
                
                # Save circuit for visualization
                if len(circuits) < 3:  # Save only a few for display
                    circuits.append(qc)
                
                # Simulate the circuit
                simulator = Aer.get_backend('qasm_simulator')
                job = simulator.run(qc, shots=1)
                result = job.result()
                counts = result.get_counts(qc)
                
                # Get the result (single bitstring)
                bitstring = list(counts.keys())[0]
                
                # Convert to appropriate format
                if output_format == "Binary":
                    value = bitstring
                elif output_format == "Decimal":
                    value = int(bitstring, 2)
                else:  # Hexadecimal
                    value = hex(int(bitstring, 2))
                
                random_values.append(value)
            
            # Display the results
            st.subheader("Generated Quantum Random Numbers")
            
            # Format the output
            if output_format == "Binary":
                result_text = "\n".join(random_values)
            elif output_format == "Decimal":
                result_text = "\n".join([str(val) for val in random_values])
            else:  # Hexadecimal
                result_text = "\n".join([str(val) for val in random_values])
            
            # Display the results
            st.code(result_text)
            
            # Visualize the distribution of results (only for multiple samples)
            if num_samples > 1 and output_format == "Decimal":
                st.subheader("Distribution of Random Numbers")
                
                # Create a histogram
                fig = px.histogram(
                    x=random_values,
                    nbins=min(20, num_samples),
                    title="Distribution of Quantum Random Numbers",
                    labels={"x": "Value", "y": "Frequency"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Allow downloading the random numbers
            st.download_button(
                label=f"Download Random Numbers ({output_format})",
                data=result_text,
                file_name=f"quantum_random_{num_bits}bits_{num_samples}samples.txt",
                mime="text/plain"
            )
            
            # Statistical tests for randomness (for larger samples)
            if num_samples >= 10:
                st.subheader("Statistical Analysis")
                
                st.markdown("""
                For true cryptographic applications, quantum random numbers should be 
                subjected to statistical tests like NIST's randomness test suite. 
                
                This demo provides basic statistical measures:
                """)
                
                if output_format == "Decimal":
                    # Calculate simple statistics
                    mean_val = sum(random_values) / len(random_values)
                    max_possible = 2**num_bits - 1
                    expected_mean = max_possible / 2
                    
                    st.markdown(f"**Mean value:** {mean_val:.2f} (Expected: {expected_mean:.2f})")
                    st.markdown(f"**Min value:** {min(random_values)}")
                    st.markdown(f"**Max value:** {max(random_values)}")
                    st.markdown(f"**Range:** {max(random_values) - min(random_values)}")
                
                # Note on true quantum randomness
                st.info("""
                **Note:** While this demo uses simulation, true quantum random number 
                generators use quantum hardware to produce random numbers based on 
                quantum phenomena like photon detection paths or quantum fluctuations.
                """)

def get_quantum_hash(text, circuit_type="SHA-2"):
    """
    Simulates a quantum-resistant hash function
    This is for demonstration only
    """
    # First apply classical SHA-256
    classical_hash = hashlib.sha256(text.encode()).digest()
    
    # Then simulate adding a quantum component (demonstration only)
    # In a real implementation, this would use actual quantum-resistant algorithms
    
    if circuit_type == "SHA-2":
        # Simulated augmented SHA-2
        return hashlib.sha256(classical_hash).hexdigest()
    elif circuit_type == "SHA-3":
        # Simulated quantum-resistant SHA-3
        return hashlib.sha3_256(classical_hash).hexdigest()
    elif circuit_type == "BLAKE2":
        # Simulated quantum-resistant BLAKE2
        import hashlib
        return hashlib.blake2b(classical_hash).hexdigest()
    else:
        # Default fallback
        return hashlib.sha256(classical_hash).hexdigest()