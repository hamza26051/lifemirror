import streamlit as st
import requests
import json
import time
from PIL import Image
import io
import base64
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
FASTAPI_URL = "http://localhost:8000"
OLD_API_URL = "http://127.0.0.1:5000"

def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def call_old_api(image):
    """Call the old lifemirror_api.py for comparison"""
    try:
        img_base64 = encode_image_to_base64(image)
        response = requests.post(
            f"{OLD_API_URL}/analyze",
            json={"image": img_base64},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def call_new_api(image):
    """Call the new FastAPI with 18-agent system"""
    try:
        # First upload the image
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
        upload_response = requests.post(f"{FASTAPI_URL}/upload", files=files)
        
        if upload_response.status_code != 200:
            return {"error": f"Upload failed: {upload_response.status_code}"}
        
        upload_data = upload_response.json()
        media_id = upload_data['media_id']
        
        # Start comprehensive analysis
        analysis_response = requests.post(
            f"{FASTAPI_URL}/analyze/comprehensive",
            json={"media_id": media_id}
        )
        
        if analysis_response.status_code != 200:
            return {"error": f"Analysis failed: {analysis_response.status_code}"}
        
        analysis_data = analysis_response.json()
        workflow_id = analysis_data['workflow_id']
        
        # Poll for results
        max_attempts = 30
        for attempt in range(max_attempts):
            status_response = requests.get(f"{FASTAPI_URL}/workflow/{workflow_id}/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data['status'] == 'completed':
                    return status_data['result']
                elif status_data['status'] == 'failed':
                    return {"error": f"Analysis failed: {status_data.get('error', 'Unknown error')}"}
            time.sleep(2)
        
        return {"error": "Analysis timed out"}
        
    except Exception as e:
        return {"error": str(e)}

def display_old_analysis(result):
    """Display old API analysis results"""
    st.subheader("🔄 Legacy Analysis (lifemirror_api.py)")
    
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return
    
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overall_score = result.get('overall_score', 0)
        st.metric("Overall Score", f"{overall_score}/10")
    
    with col2:
        face_score = result.get('face_score', 0)
        st.metric("Face Score", f"{face_score}/10")
    
    with col3:
        fashion_score = result.get('fashion_score', 0)
        st.metric("Fashion Score", f"{fashion_score}/10")
    
    # Simple analysis
    if 'analysis' in result:
        st.write("**Analysis:**")
        st.write(result['analysis'])
    
    # Raw data
    with st.expander("Raw Response Data"):
        st.json(result)

def display_new_analysis(result):
    """Display new 18-agent analysis results"""
    st.subheader("🚀 Enhanced 18-Agent Analysis (FastAPI + LangGraph)")
    
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return
    
    # Enhanced metrics with confidence
    st.write("### 📊 Comprehensive Scoring")
    
    # Main scores
    scores = result.get('scores', {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall = scores.get('overall', {})
        score = overall.get('score', 0)
        confidence = overall.get('confidence', 0)
        st.metric(
            "Overall Score", 
            f"{score}/10",
            help=f"Confidence: {confidence}%"
        )
    
    with col2:
        face = scores.get('face', {})
        score = face.get('score', 0)
        confidence = face.get('confidence', 0)
        st.metric(
            "Face Analysis", 
            f"{score}/10",
            help=f"Confidence: {confidence}%"
        )
    
    with col3:
        fashion = scores.get('fashion', {})
        score = fashion.get('score', 0)
        confidence = fashion.get('confidence', 0)
        st.metric(
            "Fashion Score", 
            f"{score}/10",
            help=f"Confidence: {confidence}%"
        )
    
    with col4:
        vibe = scores.get('vibe', {})
        score = vibe.get('score', 0)
        confidence = vibe.get('confidence', 0)
        st.metric(
            "Vibe Analysis", 
            f"{score}/10",
            help=f"Confidence: {confidence}%"
        )
    
    # Detailed analysis sections
    analysis = result.get('analysis', {})
    
    # Face Analysis Details
    if 'face_analysis' in analysis:
        st.write("### 👤 Advanced Face Analysis")
        face_data = analysis['face_analysis']
        
        col1, col2 = st.columns(2)
        with col1:
            if 'demographics' in face_data:
                demo = face_data['demographics']
                st.write(f"**Age:** {demo.get('age', 'N/A')}")
                st.write(f"**Gender:** {demo.get('gender', 'N/A')}")
                st.write(f"**Emotion:** {demo.get('emotion', 'N/A')}")
        
        with col2:
            if 'features' in face_data:
                features = face_data['features']
                st.write(f"**Attractiveness:** {features.get('attractiveness', 'N/A')}/10")
                st.write(f"**Symmetry:** {features.get('symmetry', 'N/A')}/10")
                st.write(f"**Skin Quality:** {features.get('skin_quality', 'N/A')}/10")
    
    # Fashion Analysis
    if 'fashion_analysis' in analysis:
        st.write("### 👗 Fashion & Style Analysis")
        fashion_data = analysis['fashion_analysis']
        
        if 'items_detected' in fashion_data:
            st.write("**Detected Items:**")
            for item in fashion_data['items_detected']:
                st.write(f"- {item.get('type', 'Unknown')}: {item.get('description', 'N/A')}")
        
        if 'style_assessment' in fashion_data:
            style = fashion_data['style_assessment']
            st.write(f"**Style Category:** {style.get('category', 'N/A')}")
            st.write(f"**Coordination Score:** {style.get('coordination', 'N/A')}/10")
    
    # Personality Insights
    if 'personality' in analysis:
        st.write("### 🧠 Personality Insights")
        personality = analysis['personality']
        
        if 'traits' in personality:
            traits_df = pd.DataFrame([
                {'Trait': trait, 'Score': score} 
                for trait, score in personality['traits'].items()
            ])
            
            fig = px.bar(traits_df, x='Trait', y='Score', 
                        title='Personality Trait Analysis')
            st.plotly_chart(fig, use_container_width=True)
    
    # Social Media Optimization
    if 'social_optimization' in analysis:
        st.write("### 📱 Social Media Optimization")
        social = analysis['social_optimization']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Instagram Score:** {social.get('instagram_score', 'N/A')}/10")
            st.write(f"**TikTok Score:** {social.get('tiktok_score', 'N/A')}/10")
        
        with col2:
            st.write(f"**LinkedIn Score:** {social.get('linkedin_score', 'N/A')}/10")
            st.write(f"**Dating App Score:** {social.get('dating_score', 'N/A')}/10")
    
    # Recommendations
    if 'recommendations' in result:
        st.write("### 💡 AI Recommendations")
        recommendations = result['recommendations']
        
        for category, recs in recommendations.items():
            st.write(f"**{category.title()}:**")
            for rec in recs:
                st.write(f"- {rec}")
    
    # Performance Metrics
    if 'performance' in result:
        st.write("### ⚡ Analysis Performance")
        perf = result['performance']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processing Time", f"{perf.get('total_time', 0):.2f}s")
        with col2:
            st.metric("Agents Used", perf.get('agents_executed', 0))
        with col3:
            st.metric("Confidence", f"{perf.get('overall_confidence', 0):.1f}%")
    
    # Raw data
    with st.expander("Complete Analysis Data"):
        st.json(result)

def create_comparison_chart(old_result, new_result):
    """Create comparison chart between old and new analysis"""
    st.write("### 📈 Analysis Comparison")
    
    # Extract scores for comparison
    old_scores = {
        'Overall': old_result.get('overall_score', 0),
        'Face': old_result.get('face_score', 0),
        'Fashion': old_result.get('fashion_score', 0)
    }
    
    new_scores = new_result.get('scores', {})
    new_scores_flat = {
        'Overall': new_scores.get('overall', {}).get('score', 0),
        'Face': new_scores.get('face', {}).get('score', 0),
        'Fashion': new_scores.get('fashion', {}).get('score', 0),
        'Vibe': new_scores.get('vibe', {}).get('score', 0)
    }
    
    # Create comparison dataframe
    comparison_data = []
    for metric in ['Overall', 'Face', 'Fashion']:
        comparison_data.append({
            'Metric': metric,
            'Legacy API': old_scores.get(metric, 0),
            'Enhanced 18-Agent': new_scores_flat.get(metric, 0)
        })
    
    # Add new metrics only available in enhanced version
    comparison_data.append({
        'Metric': 'Vibe Analysis',
        'Legacy API': 0,  # Not available in old API
        'Enhanced 18-Agent': new_scores_flat.get('Vibe', 0)
    })
    
    df = pd.DataFrame(comparison_data)
    
    fig = px.bar(df, x='Metric', y=['Legacy API', 'Enhanced 18-Agent'],
                 title='Score Comparison: Legacy vs Enhanced Analysis',
                 barmode='group')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature comparison table
    st.write("### 🔍 Feature Comparison")
    
    features_comparison = pd.DataFrame({
        'Feature': [
            'Basic Face Scoring',
            'Fashion Analysis',
            'Overall Rating',
            'Age Detection',
            'Gender Detection',
            'Emotion Analysis',
            'Personality Insights',
            'Social Media Optimization',
            'Style Recommendations',
            'Confidence Scores',
            'Multi-Agent Processing',
            'Historical Tracking',
            'Detailed Breakdowns',
            'Performance Metrics'
        ],
        'Legacy API': [
            '✅', '✅', '✅', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌'
        ],
        'Enhanced 18-Agent': [
            '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅'
        ]
    })
    
    st.dataframe(features_comparison, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Life Mirror - Enhanced Analysis Comparison",
        page_icon="🪞",
        layout="wide"
    )
    
    st.title("🪞 Life Mirror - Enhanced Analysis Comparison")
    st.write("Compare the legacy analysis with the new 18-agent enhanced system")
    
    # Sidebar for API status
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Check API availability
        try:
            old_health = requests.get(f"{OLD_API_URL}/health", timeout=5)
            if old_health.status_code == 200:
                st.success("✅ Legacy API (Port 5000)")
            else:
                st.error("❌ Legacy API (Port 5000)")
        except:
            st.error("❌ Legacy API (Port 5000)")
        
        try:
            new_health = requests.get(f"{FASTAPI_URL}/health", timeout=5)
            if new_health.status_code == 200:
                st.success("✅ Enhanced API (Port 8000)")
            else:
                st.error("❌ Enhanced API (Port 8000)")
        except:
            st.error("❌ Enhanced API (Port 8000)")
        
        st.markdown("---")
        st.write("**Key Improvements:**")
        st.write("• 18 specialized agents")
        st.write("• Parallel processing")
        st.write("• Confidence scoring")
        st.write("• Personality insights")
        st.write("• Social media optimization")
        st.write("• Advanced face analysis")
        st.write("• Style recommendations")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image for analysis",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to compare analysis results"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        # Analysis buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Run Legacy Analysis", use_container_width=True):
                with st.spinner("Running legacy analysis..."):
                    old_result = call_old_api(image)
                    st.session_state['old_result'] = old_result
        
        with col2:
            if st.button("🚀 Run Enhanced Analysis", use_container_width=True):
                with st.spinner("Running enhanced 18-agent analysis..."):
                    new_result = call_new_api(image)
                    st.session_state['new_result'] = new_result
        
        with col3:
            if st.button("⚡ Run Both & Compare", use_container_width=True):
                with st.spinner("Running both analyses..."):
                    col_old, col_new = st.columns(2)
                    
                    with col_old:
                        st.write("Running legacy...")
                        old_result = call_old_api(image)
                        st.session_state['old_result'] = old_result
                    
                    with col_new:
                        st.write("Running enhanced...")
                        new_result = call_new_api(image)
                        st.session_state['new_result'] = new_result
        
        # Display results
        if 'old_result' in st.session_state or 'new_result' in st.session_state:
            st.markdown("---")
            
            # Show comparison if both results exist
            if 'old_result' in st.session_state and 'new_result' in st.session_state:
                create_comparison_chart(st.session_state['old_result'], st.session_state['new_result'])
                st.markdown("---")
            
            # Display individual results
            col1, col2 = st.columns(2)
            
            with col1:
                if 'old_result' in st.session_state:
                    display_old_analysis(st.session_state['old_result'])
            
            with col2:
                if 'new_result' in st.session_state:
                    display_new_analysis(st.session_state['new_result'])
    
    else:
        st.info("👆 Upload an image to start the analysis comparison")
        
        # Show example of what to expect
        st.write("### 🎯 What You'll See")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Legacy Analysis Provides:**")
            st.write("• Basic overall score (1-10)")
            st.write("• Simple face score")
            st.write("• Basic fashion score")
            st.write("• General text analysis")
        
        with col2:
            st.write("**Enhanced 18-Agent Analysis Provides:**")
            st.write("• Detailed scoring with confidence levels")
            st.write("• Age, gender, emotion detection")
            st.write("• Personality trait analysis")
            st.write("• Social media platform optimization")
            st.write("• Style recommendations")
            st.write("• Fashion item detection")
            st.write("• Performance metrics")
            st.write("• Historical comparison")

if __name__ == "__main__":
    main()