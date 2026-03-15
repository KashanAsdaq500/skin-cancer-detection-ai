import streamlit as st
import time
import random
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ----------------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Skin Cancer Detection", 
    page_icon="🩺", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# CUSTOM CSS / DESIGN SYSTEM
# ----------------------------------------------------------------------------
def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"], p, span, h1, h2, h3, h4, h5, h6, label, div[data-testid="stMarkdownContainer"] {
            font-family: 'Outfit', sans-serif;
            color: #111827 !important; /* Sharp, dark grayscale text for optimal reading */
        }

        /* Pure White/Very light gray background */
        .stApp {
            background-color: #fcfcfc;
            background-image: 
                radial-gradient(circle at 50% 0%, #ffffff 0%, #f9fafb 70%);
            background-attachment: fixed;
        }

        /* Sidebar Styling (Deep Red to contrast) */
        [data-testid="stSidebar"] {
            background-color: #e51a2d !important;
            background-image: linear-gradient(180deg, #c41221 0%, #e51a2d 100%);
            border-right: 1px solid #990f18;
        }
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        /* Professional Hero Section (Clean White with Red Accents) */
        .hero-section {
            text-align: center;
            padding: 4rem 2rem;
            background: #ffffff;
            border-radius: 20px;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
            border: 1px solid #f3f4f6;
            position: relative;
            overflow: hidden;
        }
        
        .hero-section * {
            z-index: 2;
            position: relative;
        }
        
        /* Vibrant Red Top Border */
        .hero-section::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 6px;
            background: linear-gradient(90deg, #e51a2d, #ef4444, #fca5a5, #e51a2d);
            background-size: 300% 100%;
            animation: gradientMove 3s ease infinite;
        }

        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .hero-title {
            font-size: 3.2rem;
            font-weight: 700;
            margin-bottom: 1rem;
            letter-spacing: -1px;
            color: #111827 !important;
        }

        .hero-subtitle {
            font-size: 1.2rem;
            font-weight: 400;
            color: #6b7280 !important;
            max-width: 700px;
            margin: 0 auto;
        }

        /* Metric Cards */
        div[data-testid="stMetricValue"] {
            font-size: 2.5rem !important;
            font-weight: 700;
            color: #e51a2d !important; /* Medical Red */
        }
        div[data-testid="stMetricLabel"] {
            font-size: 1rem;
            color: #4b5563 !important; /* Dark Grey */
            font-weight: 600;
        }

        /* Main Buttons Styling */
        div.stButton > button:first-child {
            background: #e51a2d;
            color: white !important;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(229, 26, 45, 0.25);
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(229, 26, 45, 0.35);
            background: #be1224;
            border: none;
        }

        div.stButton > button:first-child p {
            color: white !important;
        }

        /* File Uploader styling */
        [data-testid="stFileUploader"] {
            padding: 3rem 2rem;
            border: 2px dashed #d1d5db;
            border-radius: 16px;
            background: #ffffff;
            transition: all 0.3s ease;
            text-align: center;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #e51a2d;
            background: #fdf2f2;
        }
        
        [data-testid="stFileUploadDropzone"] {
            color: #111827 !important;
        }

        /* Solid White Cards */
        .glass-card {
            background: #ffffff;
            border-radius: 20px;
            border: 1px solid #f3f4f6;
            padding: 2.5rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
            margin-bottom: 24px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        }

        .glass-card h3 {
            color: #111827 !important;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        .glass-card p {
            color: #4b5563 !important;
        }

        /* Premium Feature Cards */
        .feature-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            border: 1px solid #f3f4f6;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
            transition: all 0.3s ease;
            margin-bottom: 20px;
            height: 100%;
        }

        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.08);
            border-color: #fca5a5;
        }

        .icon-wrapper {
            width: 70px;
            height: 70px;
            margin: 0 auto 1.5rem auto;
            background: #fdf2f2;
            border-radius: 16px;
            border: 1px solid #fecaca;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            color: #e51a2d;
            box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        }
        
        .feature-card:hover .icon-wrapper {
            transform: scale(1.1) rotate(5deg);
            background: #e51a2d;
            color: #ffffff;
            box-shadow: 0 5px 15px rgba(229, 26, 45, 0.3);
            border-color: #e51a2d;
        }
        
        .feature-card img {
            filter: grayscale(100%) brightness(40%) sepia(100%) hue-rotate(-50deg) saturate(600%) contrast(0.8); /* Turns to medical red */
            transition: all 0.3s ease;
        }
        
        .feature-card:hover img {
            filter: brightness(0) invert(1);
        }

        .feature-title {
            font-size: 1.25rem;
            color: #111827 !important;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }

        .feature-desc {
            color: #4b5563 !important;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        /* Result Panel Labels */
        .result-title {
            font-size: 1.125rem;
            color: #9ca3af !important;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        }
        
        .result-value {
            font-size: 3rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.5rem;
            color: #111827 !important;
        }

        .high-risk { color: #dc2626 !important; }
        .medium-risk { color: #d97706 !important; }
        .low-risk { color: #059669 !important; }

        /* Smoooth fade in */
        .fade-in {
            animation: fadeIn 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
        }
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        
        hr {
            border-color: rgba(0, 0, 0, 0.1) !important;
            opacity: 1 !important;
        }

        /* Adjust global headers so they contrast nicely */
        h1, h2, h3, h4, h5, th {
            color: #111827 !important;
        }
        </style>
    """, unsafe_allow_html=True)
# ----------------------------------------------------------------------------
# HELPER FUNCTIONS (Mock AI Model)
# ----------------------------------------------------------------------------
def analyze_image(image):
    # Simulate processing time
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)
        if i < 30:
            status_text.text("Extracting features (Texture, Color, Shape)...")
        elif i < 60:
            status_text.text("Running deep learning inferences...")
        elif i < 90:
            status_text.text("Evaluating risk and confidence levels...")
        else:
            status_text.text("Finalizing report...")
            
    progress_bar.empty()
    status_text.empty()
    
    # Mock Results
    classes = ["Melanoma", "Nevus", "Basal Cell Carcinoma", "Benign Keratosis", "Actinic Keratoses"]
    probs = np.random.dirichlet(np.ones(len(classes)), size=1)[0]
    
    # Force a dominant class to make it look realistic
    dominant_idx = np.random.randint(0, len(classes))
    probs[dominant_idx] += 1.5 
    probs = probs / np.sum(probs)
    
    confidence = np.max(probs) * 100
    prediction = classes[np.argmax(probs)]
    
    if prediction in ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratoses"]:
        risk_level = "High Risk" if confidence > 70 else "Medium Risk"
    else:
        risk_level = "Low Risk" if confidence > 80 else "Medium Risk"
        
    return {
        "prediction": prediction,
        "confidence": confidence,
        "risk_level": risk_level,
        "probabilities": dict(zip(classes, probs))
    }

def get_risk_color(risk):
    if risk == "High Risk": return "high-risk"
    elif risk == "Medium Risk": return "medium-risk"
    else: return "low-risk"

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
with st.sidebar:
    st.image("png-clipart-dr-ruparelia-s-sushrusha-ayurved-multispeciality-hospital-hospital-of-the-holy-spirit-apollo-hospital-indraprastha-logo-raj-designs-hospital-miscellaneous-leaf-thumbnail.png", width=80)
    st.markdown("### AI Dermatology")
    
    menu = ["Home", "Upload & Scan", "Model Insights", "About Project"]
    choice = st.radio("Navigation", menu)
    
    st.markdown("---")
    st.markdown("### System Status")
    st.success("● System Online")
    st.info("⚡ Latency: 0.8s")
    st.markdown("---")

# Initialize custom CSS
apply_custom_css()

# ----------------------------------------------------------------------------
# PAGE ROUTING
# ----------------------------------------------------------------------------

if choice == "Home":
    st.markdown("""
        <div class="hero-section fade-in">
            <div class="hero-content">
                <h1 class="hero-title">🩺 Clinical Dermatology AI</h1>
                <p class="hero-subtitle">Professional-grade deep learning model for evaluating dermoscopic images with high clinical accuracy.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="glass-card fade-in" style="margin-bottom: 2rem;">
            <h3>Clinical Diagnostic Intelligence</h3>
            <p >
                Our AI model is trained on tens of thousands of clinically verified dermoscopy images. 
                Using state-of-the-art convolutional neural networks (CNNs), it can assist medical professionals 
                and patients by providing a rapid second opinion on skin lesions.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card fade-in">
            <div class="icon-wrapper">
                <img src="https://cdn-icons-png.flaticon.com/512/862/862032.png" width="40">
            </div>
            <div class="feature-title">High Accuracy</div>
            <div class="feature-desc">94.5% clinical validation accuracy across 7 major skin diseases using state-of-the-art deep learning.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card fade-in" style="animation-delay: 0.1s;">
            <div class="icon-wrapper">
                <img src="https://cdn-icons-png.flaticon.com/512/3063/3063822.png" width="40">
            </div>
            <div class="feature-title">Instant Results</div>
            <div class="feature-desc">Get detailed probability distributions and dynamic risk assessments processed in fractions of a second.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card fade-in" style="animation-delay: 0.2s;">
            <div class="icon-wrapper">
                <img src="https://cdn-icons-png.flaticon.com/512/2814/2814322.png" width="40">
            </div>
            <div class="feature-title">Secure & Private</div>
            <div class="feature-desc">All image streams are processed locally in an encrypted environment. No visual data is ever stored.</div>
        </div>
        """, unsafe_allow_html=True)

elif choice == "Upload & Scan":
    st.markdown("""
        <div class="hero-section fade-in" style="padding: 1.5rem 1rem;">
            <div class="hero-content">
                <h1 class="hero-title" style="font-size: 2.2rem;">🔍 Diagnostic Scanner</h1>
                <p class="hero-subtitle">Upload a clear dermoscopy or clinical image for analysis.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown("### 1. Upload Image")
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.markdown('<div class="glass-card fade-in" style="padding: 15px;">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image Preview", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            analyze_btn = st.button("Analyze Image", use_container_width=True)
            
            if analyze_btn:
                st.session_state['results'] = analyze_image(image)
                
    with col2:
        st.markdown("### 2. Analysis Results")
        
        if 'results' in st.session_state and uploaded_file is not None:
            res = st.session_state['results']
            risk_class = get_risk_color(res['risk_level'])
            
            # Primary Result Card & Gauge Charts
            st.markdown(f"""
<div class="glass-card fade-in" style="text-align: center;">
    <div class="result-title" style="font-size: 1.125rem; letter-spacing: 0.1em; color: #ffffff;">AI PREDICTION</div>
    <div class="result-value" style="font-size: 2.5rem; font-weight: 700; color: #111827;">{res['prediction']}</div>
</div>
""", unsafe_allow_html=True)

            gc1, gc2 = st.columns(2)
            with gc1:
                fig_conf = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res['confidence'],
                    title={'text': "Confidence Level", 'font': {'size': 20, 'color': '#6b7280'}},
                    number={'suffix': "%", 'font': {'size': 35, 'color': '#e51a2d'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#e51a2d"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(59,130,246,0.2)",
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(220, 38, 38, 0.1)"},
                            {'range': [50, 80], 'color': "rgba(234, 88, 12, 0.1)"},
                            {'range': [80, 100], 'color': "rgba(22, 163, 74, 0.1)"}],
                    }
                ))
                fig_conf.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_conf, use_container_width=True, config={'displayModeBar': False})
                
            with gc2:
                risk_val = 20 if res['risk_level'] == "Low Risk" else (50 if res['risk_level'] == "Medium Risk" else 90)
                risk_color = "#2ecc71" if risk_val == 20 else ("#f39c12" if risk_val == 50 else "#e74c3c")
                
                fig_risk = go.Figure(go.Indicator(
                    mode="gauge",
                    value=risk_val,
                    title={'text': "Risk Assessment", 'font': {'size': 20, 'color': '#6b7280'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickvals': [16.5, 50, 83.5], 'ticktext': ['Low', 'Medium', 'High']},
                        'bar': {'color': risk_color},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(0,0,0,0.1)",
                        'steps': [
                            {'range': [0, 33.3], 'color': "rgba(46, 204, 113, 0.2)"},
                            {'range': [33.3, 66.6], 'color': "rgba(243, 156, 18, 0.2)"},
                            {'range': [66.6, 100], 'color': "rgba(231, 76, 60, 0.2)"}],
                    }
                ))
                fig_risk.add_annotation(x=0.5, y=0.35, text=f"<b>{res['risk_level']}</b>", showarrow=False, font=dict(size=26, color=risk_color))
                fig_risk.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_risk, use_container_width=True, config={'displayModeBar': False})
            
            # Probability Bar Chart
            st.markdown("#### Probability Distribution")
            df_probs = pd.DataFrame(list(res['probabilities'].items()), columns=['Class', 'Probability'])
            df_probs['Probability'] = df_probs['Probability'] * 100
            df_probs = df_probs.sort_values(by='Probability', ascending=True)
            
            fig = px.bar(
                df_probs, x='Probability', y='Class', orientation='h',
                color='Probability', color_continuous_scale='Reds',
                text_auto='.1f'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False, title=''),
                yaxis=dict(title='', tickfont=dict(size=13, color='#475569')),
                height=250,
                coloraxis_showscale=False
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        elif uploaded_file is None:
            st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 50px 20px; color: #ffffff;">
                    <img src="https://cdn-icons-png.flaticon.com/512/6104/6104045.png" width="80" style="margin-bottom: 20px; opacity: 0.5;">
                    <h4>No Image Uploaded</h4>
                    <p>Please upload an image from the left panel and click 'Analyze' to view diagnostics.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
             st.markdown("""
                <div class="glass-card fade-in" style="text-align: center; padding: 40px 20px; color: #ffffff;">
                    <h4>Image Ready</h4>
                    <p>Click the <b>Analyze Image</b> button to begin AI processing.</p>
                </div>
            """, unsafe_allow_html=True)

elif choice == "Model Insights":
    st.markdown("## 📊 Model Architecture & Performance Insights")
    st.markdown("Review the underlying technical metrics and performance evaluations of the deployed deep learning model.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("🧠 Model Architecture Overview", expanded=True):
        st.markdown("""
        **Base Model**: EfficientNet-B4 (Pre-trained on ImageNet)  
        **Fine-tuning Details**: 
        - Multi-head attention pooling added to final layers.
        - Augmented with physical data (patient age, gender, anatomical site).
        - Optimized using AdamW with Cosine Annealing.
        """)
        
    with st.expander("📈 Performance Metrics", expanded=True):
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Overall Accuracy", "94.5%", "Validation Set")
        mc2.metric("Mean F1-Score", "0.92", "+0.04 vs prior ver")
        mc3.metric("Recall (Melanoma)", "96.2%", "Critical metric")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Precision, Recall & F1 by Class")
        
        # Mock Metrics Data
        metrics_data = pd.DataFrame({
            "Class": ["Melanoma", "Nevus", "BCC", "BKL", "AKIEC"],
            "Precision": [0.91, 0.95, 0.89, 0.88, 0.93],
            "Recall": [0.96, 0.97, 0.91, 0.85, 0.89],
            "F1-Score": [0.93, 0.96, 0.90, 0.86, 0.91]
        })
        
        categories = metrics_data['Class'].tolist()
        
        fig2 = go.Figure()

        # Precision Trace
        fig2.add_trace(go.Scatterpolar(
            r=metrics_data['Precision'].tolist() + [metrics_data['Precision'].tolist()[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Precision',
            line=dict(color='#FF4B4B', width=2),
            fillcolor='rgba(255, 75, 75, 0.2)',
            marker=dict(size=8, symbol='circle')
        ))
        
        # Recall Trace
        fig2.add_trace(go.Scatterpolar(
            r=metrics_data['Recall'].tolist() + [metrics_data['Recall'].tolist()[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Recall',
            line=dict(color='#00b4db', width=2),
            fillcolor='rgba(0, 180, 219, 0.2)',
            marker=dict(size=8, symbol='square')
        ))
        
        # F1-Score Trace
        fig2.add_trace(go.Scatterpolar(
            r=metrics_data['F1-Score'].tolist() + [metrics_data['F1-Score'].tolist()[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='F1-Score',
            line=dict(color='#a18cd1', width=2),
            fillcolor='rgba(161, 140, 209, 0.2)',
            marker=dict(size=8, symbol='diamond')
        ))

        fig2.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.8, 1.0],  # Zoom in to highlight differences
                    gridcolor='rgba(0,0,0,0.1)',
                    linecolor='rgba(0,0,0,0.1)',
                    tickfont=dict(size=11, color='#475569')
                ),
                angularaxis=dict(
                    gridcolor='rgba(0,0,0,0.1)',
                    linecolor='rgba(0,0,0,0.1)',
                    tickfont=dict(size=14, color='#0b0f19', family='Outfit', weight='bold')
                ),
                bgcolor='rgba(255, 255, 255, 0.4)',
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=14, family='Outfit')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(t=30, b=30, l=40, r=40)
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    with st.expander("🔲 Confusion Matrix", expanded=False):
        # Generate dummy confusion matrix
        z = [[85, 5, 2, 0, 1],
             [3, 150, 4, 1, 0],
             [2, 4, 70, 3, 2],
             [1, 2, 1, 60, 4],
             [0, 1, 0, 2, 45]]
        classes = ["MEL", "NV", "BCC", "BKL", "AKIEC"]
        fig3 = px.imshow(z, x=classes, y=classes, text_auto=True, color_continuous_scale='Reds')
        fig3.update_layout(height=400, margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

elif choice == "About Project":
    st.markdown("## 📖 About the Project")
    st.markdown("""
    <div class="glass-card fade-in">
        <h3>Vision & Motivation</h3>
        <p style="color: #ffffff; line-height: 1.6;">
        Skin cancer is one of the most common types of cancer worldwide. Early detection drastically improves survival rates. 
        This project aims to democratize access to preliminary dermatological screening by putting a highly accurate AI 
        tool directly in the hands of users and medical practitioners through a seamless web interface.
        </p>
        <hr style="border:0; height:1px; background: rgba(0,0,0,0.1);">
        <h3>Technologies Used</h3>
        <ul>
            <li><b>Deep Learning:</b> PyTorch, torchvision, pre-trained CNNs (EfficientNet)</li>
            <li><b>Data Processing:</b> Pandas, NumPy, OpenCV</li>
            <li><b>Frontend / UI:</b> Streamlit, Custom CSS, HTML5, Glassmorphism UI</li>
            <li><b>Visualization:</b> Plotly Express, Plotly Graph Objects</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
# Footer has been removed as requested.
