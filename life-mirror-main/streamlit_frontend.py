import streamlit as st
import requests
import json
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Life Mirror API Tester",
    page_icon="🔍",
    layout="wide"
)

# API Base URL
API_BASE_URL = "http://localhost:8000"
BASE_URL = "http://localhost:8000"

def test_api_connection():
    """Test if the API is accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/docs")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def refresh_access_token():
    """Refresh the access token using the refresh token"""
    if not st.session_state.refresh_token:
        return False
    
    try:
        response = requests.post(
            f"{BASE_URL}/auth/refresh",
            json={"refresh_token": st.session_state.refresh_token}
        )
        
        if response.status_code == 200:
            token_data = response.json()
            st.session_state.access_token = token_data.get('access_token')
            st.session_state.refresh_token = token_data.get('refresh_token')
            return True
        else:
            # Refresh failed, clear tokens
            st.session_state.access_token = None
            st.session_state.refresh_token = None
            return False
    except Exception as e:
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        return False

def make_authenticated_request(url, method="GET", data=None, files=None, headers=None):
    """Make authenticated API request with automatic token refresh"""
    if not st.session_state.access_token:
        return {
            "status_code": 401,
            "response": "No access token available",
            "success": False
        }
    
    # Prepare headers
    auth_headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    if headers:
        auth_headers.update(headers)
    
    try:
        # Make the request
        if method == "GET":
            response = requests.get(url, headers=auth_headers)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, headers=auth_headers)
            else:
                auth_headers["Content-Type"] = "application/json"
                response = requests.post(url, json=data, headers=auth_headers)
        elif method == "PUT":
            auth_headers["Content-Type"] = "application/json"
            response = requests.put(url, json=data, headers=auth_headers)
        elif method == "DELETE":
            response = requests.delete(url, headers=auth_headers)
        
        # If we get 401, try to refresh token and retry once
        if response.status_code == 401:
            if refresh_access_token():
                # Update headers with new token
                auth_headers["Authorization"] = f"Bearer {st.session_state.access_token}"
                
                # Retry the request
                if method == "GET":
                    response = requests.get(url, headers=auth_headers)
                elif method == "POST":
                    if files:
                        response = requests.post(url, files=files, headers=auth_headers)
                    else:
                        auth_headers["Content-Type"] = "application/json"
                        response = requests.post(url, json=data, headers=auth_headers)
                elif method == "PUT":
                    auth_headers["Content-Type"] = "application/json"
                    response = requests.put(url, json=data, headers=auth_headers)
                elif method == "DELETE":
                    response = requests.delete(url, headers=auth_headers)
        
        return {
            "status_code": response.status_code,
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
            "success": response.status_code < 400
        }
    except requests.exceptions.RequestException as e:
        return {
            "status_code": None,
            "response": str(e),
            "success": False
        }

def make_api_request(endpoint, method="GET", data=None):
    """Make API request and return response"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        return {
            "status_code": response.status_code,
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
            "success": response.status_code < 400
        }
    except requests.exceptions.RequestException as e:
        return {
            "status_code": None,
            "response": str(e),
            "success": False
        }

# Main app
st.title("🔍 Life Mirror API Tester")
st.markdown("---")

# API Connection Status
st.subheader("📡 API Connection Status")
if test_api_connection():
    st.success("✅ API is accessible at http://localhost:8000")
    st.info("📖 API Documentation: http://localhost:8000/docs")
else:
    st.error("❌ Cannot connect to API. Make sure your FastAPI server is running.")
    st.stop()

st.markdown("---")

# Tabs for different functionalities
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Health Check", "👤 Users", "📁 Media", "🔧 Custom Request", "🔍 Quick Analysis"])

with tab1:
    st.subheader("Health Check")
    if st.button("Check API Health"):
        result = make_api_request("/health")
        if result["success"]:
            st.success(f"✅ API Health: {result['response']}")
        else:
            st.error(f"❌ Health check failed: {result['response']}")
        
        st.json(result)

with tab2:
    st.subheader("User Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Get All Users**")
        if st.button("Fetch Users"):
            result = make_api_request("/api/users")
            if result["success"]:
                st.success("✅ Users fetched successfully")
                st.json(result["response"])
            else:
                st.error(f"❌ Failed to fetch users: {result['response']}")
    
    with col2:
        st.write("**Create New User**")
        with st.form("create_user"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Create User"):
                if username and email and password:
                    user_data = {
                        "username": username,
                        "email": email,
                        "password": password
                    }
                    result = make_api_request("/api/users", method="POST", data=user_data)
                    if result["success"]:
                        st.success("✅ User created successfully")
                        st.json(result["response"])
                    else:
                        st.error(f"❌ Failed to create user: {result['response']}")
                else:
                    st.warning("Please fill in all fields")

with tab3:
    st.subheader("📁 Media Management")
    
    # Authentication Section
    st.write("**Authentication Required for File Upload**")
    
    # Session state for tokens
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'refresh_token' not in st.session_state:
        st.session_state.refresh_token = None
    
    if st.session_state.access_token is None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Login**")
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                if login_email and login_password:
                    try:
                        login_data = {
                            "email": login_email,
                            "password": login_password
                        }
                        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
                        
                        if response.status_code == 200:
                            token_data = response.json()
                            st.session_state.access_token = token_data.get('access_token')
                            st.session_state.refresh_token = token_data.get('refresh_token')
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error(f"❌ Login failed: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Login error: {str(e)}")
        
        with col2:
            st.write("**Register New User**")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_alias = st.text_input("Public Alias (Optional)", key="reg_alias")
            
            if st.button("Register"):
                if reg_email and reg_password:
                    try:
                        register_data = {
                            "email": reg_email,
                            "password": reg_password,
                            "public_alias": reg_alias if reg_alias else None
                        }
                        response = requests.post(f"{BASE_URL}/auth/register", json=register_data)
                        
                        if response.status_code == 200:
                            st.success("✅ Registration successful! Please login.")
                        else:
                            st.error(f"❌ Registration failed: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Registration error: {str(e)}")
    
    else:
        st.success("✅ Authenticated! You can now upload files.")
        if st.button("Logout"):
            st.session_state.access_token = None
            st.session_state.refresh_token = None
            st.rerun()
        
        # File Upload Section (only when authenticated)
        st.write("**Upload Media File**")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['jpg', 'jpeg', 'png', 'webp', 'mp4', 'mov'],
            help="Supported formats: JPG, JPEG, PNG, WEBP, MP4, MOV (Max 50MB)"
        )
        
        if uploaded_file is not None:
            st.write(f"Selected file: {uploaded_file.name}")
            st.write(f"File size: {uploaded_file.size} bytes")
            
            if st.button("Upload and Analyze File"):
                try:
                    # Prepare the file for upload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    # Upload the file using authenticated request
                    upload_result = make_authenticated_request(
                        f"{BASE_URL}/media/upload", 
                        method="POST", 
                        files=files
                    )
                    
                    if upload_result["success"]:
                        upload_data = upload_result["response"]
                        st.success(f"✅ File uploaded successfully! Media ID: {upload_data.get('media_id')}")
                        
                        # Trigger analysis
                        media_id = upload_data.get('media_id')
                        
                        if media_id:
                            st.write("🔄 Starting analysis...")
                            analysis_result = make_authenticated_request(
                                f"{BASE_URL}/analysis/analyze",
                                method="POST",
                                data={"media_id": media_id}
                            )
                            
                            if analysis_result["success"]:
                                analysis_data = analysis_result["response"]
                                st.success("✅ Analysis completed!")
                                st.json(analysis_data)
                            else:
                                st.error(f"❌ Analysis failed: {analysis_result['response']}")
                        
                    else:
                        st.error(f"❌ Upload failed: {upload_result['response']}")
                        
                except Exception as e:
                    st.error(f"❌ Error during upload: {str(e)}")
    
    # Media List (works without auth)
    st.write("**Media List**")
    if st.button("Get Media List"):
        result = make_api_request("/media")
        if result["success"]:
            st.json(result["response"])
        else:
            st.error(f"Failed to get media list: {result['response']}")
    
    # Media Details (works without auth)
    st.write("**Get Media Details**")
    media_id = st.number_input("Media ID", min_value=1, value=1, step=1, key="media_details_id")
    if st.button("Get Media Details"):
        result = make_api_request(f"/media/{media_id}")
        if result["success"]:
            st.json(result["response"])
        else:
            st.error(f"Failed to get media details: {result['response']}")

with tab4:
    st.subheader("Custom API Request")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        method = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])
        endpoint = st.text_input("Endpoint", placeholder="/api/endpoint")
        
        if method in ["POST", "PUT"]:
            st.write("**Request Body (JSON)**")
            request_body = st.text_area("JSON Data", placeholder='{"key": "value"}')
        else:
            request_body = None
    
    with col2:
        if st.button("Send Request"):
            if endpoint:
                try:
                    data = json.loads(request_body) if request_body else None
                    result = make_api_request(endpoint, method=method, data=data)
                    
                    st.write(f"**Status Code:** {result['status_code']}")
                    if result["success"]:
                        st.success("✅ Request successful")
                    else:
                        st.error("❌ Request failed")
                    
                    st.write("**Response:**")
                    st.json(result["response"])
                    
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON in request body")
            else:
                st.warning("Please enter an endpoint")

with tab5:
    st.subheader("🔍 Quick Analysis (No Auth Required)")
    
    st.write("**Test Analysis Endpoints Without Authentication**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Social Graph Analysis**")
        user_id_input = st.number_input("User ID", min_value=1, value=1, step=1, key="social_graph_user_id")
        if st.button("Get Social Graph"):
            result = make_api_request(f"/graph/social-graph?user_id={user_id_input}")
            if result["success"]:
                st.success("✅ Social graph analysis completed!")
                st.json(result["response"])
            else:
                st.error(f"❌ Social graph analysis failed: {result['response']}")
    
    with col2:
        st.write("**Reverse Analysis**")
        goal_input = st.text_input("Desired Goal/Vibe", placeholder="confident and professional")
        recent_limit = st.slider("Recent Uploads to Consider", 1, 10, 5)
        if st.button("Get Reverse Analysis") and goal_input:
            result = make_api_request(f"/analysis/reverse-analysis?user_id={user_id_input}&goal={goal_input}&recent_limit={recent_limit}")
            if result["success"]:
                st.success("✅ Reverse analysis completed!")
                st.json(result["response"])
            else:
                st.error(f"❌ Reverse analysis failed: {result['response']}")
    
    st.write("**Media Perception Analysis**")
    media_id_input = st.number_input("Media ID", min_value=1, value=1, step=1, key="perception_media_id")
    if st.button("Get Media Perception"):
        result = make_api_request(f"/media/media/{media_id_input}/perception")
        if result["success"]:
            st.success("✅ Media perception analysis completed!")
            st.json(result["response"])
        else:
            st.error(f"❌ Media perception analysis failed: {result['response']}")
    
    st.info("💡 **Note:** These endpoints work with existing data in the database. In production mode, they will use real AI analysis instead of mock data.")

st.markdown("---")
st.markdown("**💡 Tips:**")
st.markdown("""
- Make sure your FastAPI server is running on http://localhost:8000
- Check the API documentation at http://localhost:8000/docs
- Use the Custom Request tab to test any endpoint
- Use the Quick Analysis tab to test analysis without authentication
- Monitor your FastAPI server logs for debugging
""")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Life Mirror API Tester - Built with Streamlit</div>", unsafe_allow_html=True)