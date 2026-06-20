import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as html

# Page Config
st.set_page_config(
    page_title="DARP - Data Analysis & Reporting Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to fetch backend URL from environment variables, streamlit secrets, or fallback to Render production
BACKEND_URL = os.environ.get("BACKEND_URL") or os.environ.get("API_BASE_URL")

if not BACKEND_URL:
    try:
        BACKEND_URL = st.secrets.get("BACKEND_URL") or st.secrets.get("API_BASE_URL")
    except:
        pass

if not BACKEND_URL:
    BACKEND_URL = "https://readynest-internship-week1-2.onrender.com"

BACKEND_URL = BACKEND_URL.rstrip("/")

# Session State for Login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Render Login Page if not authenticated
if not st.session_state["logged_in"]:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #1E0C26 0%, #0A020F 100%) !important;
            font-family: 'Outfit', sans-serif !important;
        }
        
        /* Center login wrapper */
        .login-card {
            background-color: #1B0625;
            border-radius: 16px;
            padding: 40px;
            border: 1px solid #FF2E93;
            box-shadow: 0 10px 35px rgba(255, 46, 147, 0.15);
            margin-top: 15%;
            color: #FFFFFF;
        }
        
        .login-title {
            font-size: 3.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #FF2E93 0%, #D2FF28 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 5px;
        }
        
        .login-subtitle {
            font-size: 0.85rem;
            color: #BCA3C4;
            text-align: center;
            margin-bottom: 30px;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 600;
        }
        
        /* Styled Submit Button for login */
        .stButton>button {
            width: 100% !important;
            background: linear-gradient(135deg, #FF2E93 0%, #B51D75 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 20px !important;
            box-shadow: 0 4px 15px rgba(255, 46, 147, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton>button:hover {
            box-shadow: 0 6px 20px rgba(255, 46, 147, 0.5) !important;
            transform: translateY(-2px) !important;
            color: white !important;
        }
        
        /* Custom input fields override */
        div[data-baseweb="input"] {
            background-color: #2F1240 !important;
            border: 1px solid #4D2566 !important;
            border-radius: 8px !important;
        }
        
        div[data-baseweb="input"] input {
            color: #FFFFFF !important;
        }
        
        label {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        
        .cred-info {
            text-align: center;
            font-size: 0.8rem;
            color: #7A6284;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<h1 class="login-title">DARP</h1>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Data Analysis & Reporting Platform</div>', unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        
        login_btn = st.button("Log In", type="primary")
        st.markdown('<div class="cred-info">Default Credentials: admin / admin</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if login_btn:
            if username == "admin" and password == "admin":
                st.session_state["logged_in"] = True
                st.success("Successfully Logged In!")
                st.rerun()
            else:
                st.error("Invalid credentials, please try again.")
    st.stop()

# Custom Premium Styling (Contentsquare inspired theme for main app)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #F8F6F9;
    }
    
    /* Title and main header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1E0C26 0%, #FF2E93 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 1.05rem;
        color: #7A6284;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1E0C26 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #4D2566 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    
    /* Sidebar selectbox */
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #2F1240 !important;
        border: 1px solid #4D2566 !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #FFFFFF !important;
    }
    
    /* KPI Card styling inspired by Contentsquare */
    .kpi-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 30px rgba(30, 12, 38, 0.03);
        border: 1px solid #F1EBF5;
        border-left: 5px solid #FF2E93;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(30, 12, 38, 0.06);
    }
    
    .kpi-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E0C26;
    }
    
    .kpi-lbl {
        font-size: 0.85rem;
        color: #7A6284;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    /* Tabs customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #F1EBF5;
        padding: 6px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        color: #7A6284;
        transition: all 0.2s ease;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #1E0C26;
        background-color: rgba(255, 46, 147, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E0C26 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(30, 12, 38, 0.15);
    }
    
    /* Nested tabs styling */
    .stTabs .stTabs [data-baseweb="tab-list"] {
        background-color: #E6DFEB !important;
    }
    
    .stTabs .stTabs [aria-selected="true"] {
        background-color: #FF2E93 !important;
        color: #FFFFFF !important;
    }
    
    /* Streamlit buttons custom styling */
    .stButton>button {
        background-color: #1E0C26 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #FF2E93 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 46, 147, 0.2);
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.markdown('<div class="main-header">📊 DARP</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data Analysis & Reporting Platform &mdash; Decoupled Architecture inspired by Contentsquare.</div>', unsafe_allow_html=True)

# Helper function to query backend
def api_request(method, endpoint, **kwargs):
    url = f"{BACKEND_URL}{endpoint}"
    try:
        if method.lower() == "get":
            r = requests.get(url, **kwargs)
        elif method.lower() == "post":
            r = requests.post(url, **kwargs)
        else:
            raise ValueError("Unsupported HTTP method")
        
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Backend API Error ({r.status_code}): {r.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend server.\n\nConfigured API URL:\n`{BACKEND_URL}`")
        st.stop()

# Fetch uploaded datasets from database (stored files details)
datasets_list = api_request("get", "/datasets")

st.sidebar.markdown("### ⚙️ Workspace Configuration")

# Selection Mode
options = ["Upload New Dataset"] + [f"{d['filename']} ({d['id'][:8]})" for d in (datasets_list or [])]

# Sync selection from new upload before selectbox is rendered
if "new_upload_selection" in st.session_state and st.session_state["new_upload_selection"]:
    st.session_state["data_selection_sb"] = st.session_state["new_upload_selection"]
    del st.session_state["new_upload_selection"]

# Sync selection state
if "data_selection_sb" not in st.session_state:
    st.session_state["data_selection_sb"] = "Upload New Dataset"
elif st.session_state["data_selection_sb"] not in options:
    st.session_state["data_selection_sb"] = "Upload New Dataset"

data_selection = st.sidebar.selectbox(
    "Active Dataset Selection",
    options,
    key="data_selection_sb"
)

selected_id = None
dataset_details = None

if data_selection == "Upload New Dataset":
    st.session_state["selected_dataset_id"] = None
    
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
        
    file = st.sidebar.file_uploader(
        "Upload dataset file", 
        type=["csv", "xlsx", "json"],
        key=f"file_uploader_{st.session_state['uploader_key']}"
    )
    if file:
        # Validate file size (100MB upload limit)
        MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
        if file.size > MAX_UPLOAD_SIZE:
            st.sidebar.error("File size exceeds the 100MB upload limit. Please upload a smaller file.")
        else:
            with st.spinner("Uploading file and storing details in DB..."):
                files = {"file": (file.name, file.getvalue())}
                res = api_request("post", "/datasets/upload", files=files)
                if res:
                    st.sidebar.success(f"Uploaded and stored: {file.name}")
                    st.session_state["selected_dataset_id"] = res["id"]
                    st.session_state["uploader_key"] += 1
                    st.session_state["new_upload_selection"] = f"{res['filename']} ({res['id'][:8]})"
                    st.rerun()
else:
    # Extract ID from selection label e.g. "sales.csv (dc7285e0)"
    filename_part = data_selection.split(" (")[0]
    id_part = data_selection.split(" (")[1][:-1]
    
    # Find matching dataset record
    for d in (datasets_list or []):
        if d["filename"] == filename_part and d["id"].startswith(id_part):
            selected_id = d["id"]
            dataset_details = d
            break
            
    if selected_id:
        st.session_state["selected_dataset_id"] = selected_id

# Verify dataset is selected and load details
if "selected_dataset_id" not in st.session_state or not st.session_state["selected_dataset_id"]:
    st.warning("Please upload a new dataset or select one of the existing files details in the sidebar to get started.")
    st.stop()

selected_id = st.session_state["selected_dataset_id"]

# Fetch updated dataset details from backend
dataset_details = api_request("get", f"/datasets/{selected_id}")
if not dataset_details:
    st.error("Failed to load selected dataset details.")
    st.stop()

# Short info in sidebar
row_count = dataset_details["shape_rows"]
col_count = dataset_details["shape_cols"]
st.sidebar.markdown(f"**Filename:** {dataset_details['filename']}")
st.sidebar.markdown(f"**DB Record ID:** `{selected_id}`")
st.sidebar.markdown(f"**Shape:** {row_count} rows × {col_count} columns")
st.sidebar.markdown(f"**Status:** `{dataset_details['status'].upper()}`")

# Sidebar Log Out option
st.sidebar.markdown("---")
if st.sidebar.button("🔓 Log Out"):
    st.session_state["logged_in"] = False
    st.rerun()

# DARP Parent Tabs
main_tabs = st.tabs([
    "🏠 Workspace Overview",
    "🧼 Data Preparation",
    "📊 Exploratory Analytics",
    "🧠 Advanced Analytics",
    "📥 Report Center"
])

# ----------------- TAB 0: WORKSPACE OVERVIEW -----------------
with main_tabs[0]:
    st.header("🏠 Workspace Overview")
    st.markdown("Overview of your active dataset, database metadata, and automated insights.")
    
    col_overview_left, col_overview_right = st.columns([1, 1.2])
    
    with col_overview_left:
        st.subheader("Dataset Schema Preview")
        st.markdown(f"**Shape:** {row_count} rows, {col_count} columns")
        
        # Details table
        details_df = pd.DataFrame({
            "Data Type": pd.Series(dataset_details["data_types"]),
            "Missing Values": pd.Series(dataset_details["missing_counts"]),
            "Missing %": pd.Series({col: f"{(count / row_count * 100):.1f}%" if row_count > 0 else "0.0%" for col, count in dataset_details["missing_counts"].items()})
        })
        st.dataframe(details_df, use_container_width=True)
        
    with col_overview_right:
        st.subheader("🧠 Automated Data Insights")
        with st.spinner("Extracting automated business findings..."):
            insights = api_request("get", f"/datasets/{selected_id}/insights")
            if insights:
                st.info("💡 **Key Findings**")
                for finding in insights.get('findings', []):
                    st.markdown(f"- {finding}")
                    
                st.warning("📋 **Recommendations & Actions**")
                for rec in insights.get('recommendations', []):
                    st.markdown(f"**Rec:** {rec}")
                for act in insights.get('action_items', []):
                    st.markdown(f"**Action:** {act}")

# ----------------- TAB 1: DATA PREPARATION -----------------
with main_tabs[1]:
    sub_prep_tab1, sub_prep_tab2 = st.tabs(["🧼 Data Cleaning", "⚙️ Feature Engineering"])
    
    # Sub-tab 1: Data Cleaning
    with sub_prep_tab1:
        st.header("🧼 Data Cleaning Workspace")
        st.markdown("Prepare your dataset by resolving missing values, removing duplicates, capping outliers, and correcting data types.")
        
        col_clean_left, col_clean_right = st.columns([1, 1])
        
        with col_clean_left:
            st.subheader("Basic Cleaning")
            # Action fields
            remove_dups = st.checkbox("Remove Duplicate Rows")
            
            # Missing values
            null_cols = [col for col, count in dataset_details["missing_counts"].items() if count > 0]
            impute_col = None
            impute_strategy = None
            fill_val = None
            
            if null_cols:
                st.markdown("#### Missing Value Imputation")
                impute_col = st.selectbox("Select Column to Impute", null_cols)
                impute_strategy = st.selectbox("Select Imputation Strategy", ["mean", "median", "mode", "drop", "constant"])
                if impute_strategy == "constant":
                    fill_val = st.text_input("Impute Fill Value")
            else:
                st.info("No missing values detected in the current dataset!")
                
        with col_clean_right:
            # Outlier treatment
            numeric_cols = [col for col, dtype in dataset_details["data_types"].items() if dtype == "numeric"]
            outlier_col = None
            outlier_method = None
            outlier_action = None
            
            if numeric_cols:
                st.subheader("Outlier Treatment")
                outlier_col = st.selectbox("Select Column for Outliers", numeric_cols)
                outlier_method = st.radio("Detection Method", ["iqr", "zscore"])
                outlier_action = st.radio("Action to Take", ["cap", "drop", "replace"])
                
        st.markdown("---")
        if st.button("Apply Cleaning Options", type="primary", key="btn_clean"):
            with st.spinner("Cleaning dataset and storing changes..."):
                payload = {
                    "remove_duplicates": remove_dups,
                    "impute_col": impute_col,
                    "impute_strategy": impute_strategy,
                    "fill_value": fill_val,
                    "outlier_col": outlier_col,
                    "outlier_method": outlier_method,
                    "outlier_action": outlier_action
                }
                res = api_request("post", f"/datasets/{selected_id}/clean", json=payload)
                if res:
                    st.success("Dataset cleaned and updated in database successfully!")
                    st.rerun()

    # Sub-tab 2: Feature Engineering
    with sub_prep_tab2:
        st.header("⚙️ Feature Engineering Workbench")
        st.markdown("Scale features, encode labels, extract temporal variables, or calculate business metrics.")
        
        col_fe_l, col_fe_r = st.columns([1, 2])
        
        with col_fe_l:
            st.subheader("Transformation Actions")
            
            # 1. Date features
            date_candidates = [col for col, dtype in dataset_details["data_types"].items() if dtype in ["datetime", "object"]]
            date_col = st.selectbox("Select Date Column", ["None"] + date_candidates)
            
            # 2. Scaling
            num_cols = [col for col, dtype in dataset_details["data_types"].items() if dtype == "numeric"]
            scale_cols = st.multiselect("Select columns to scale", num_cols)
            scale_method = st.selectbox("Scaling Method", ["standard", "minmax"])
            
            # 3. Categorical encoding
            cat_cols = [col for col, dtype in dataset_details["data_types"].items() if dtype in ["object", "category"]]
            encode_cols = st.multiselect("Select columns to encode", cat_cols)
            encode_method = st.selectbox("Encoding Method", ["Label Encoding", "One-Hot Encoding"])
            
            # 4. Custom Business Ratio
            st.markdown("#### Custom Ratios")
            ratio_type = st.selectbox("Derived Ratio Type", ["None", "profit_margin", "custom_ratio"])
            num_ratio = None
            den_ratio = None
            if ratio_type == "custom_ratio":
                num_ratio = st.selectbox("Numerator Column", num_cols)
                den_ratio = st.selectbox("Denominator Column", num_cols)
                
            if st.button("Apply Transformations", type="primary", key="btn_fe"):
                with st.spinner("Engineering features in backend..."):
                    payload = {
                        "date_col": None if date_col == "None" else date_col,
                        "scale_cols": scale_cols if scale_cols else None,
                        "scale_method": scale_method if scale_cols else None,
                        "encode_cols": encode_cols if encode_cols else None,
                        "encode_method": encode_method if encode_cols else None,
                        "ratio_type": None if ratio_type == "None" else ratio_type,
                        "num_col": num_ratio,
                        "den_col": den_ratio
                    }
                    res = api_request("post", f"/datasets/{selected_id}/feature-engineering", json=payload)
                    if res:
                        st.success("Transformed features saved successfully!")
                        st.rerun()
                        
        with col_fe_r:
            st.subheader("Transformed Dataset Schema")
            st.write("Current dataset columns:")
            st.write(dataset_details["columns_list"])

# ----------------- TAB 2: EXPLORATORY ANALYTICS -----------------
with main_tabs[2]:
    sub_expl_tab1, sub_expl_tab2, sub_expl_tab3 = st.tabs([
        "📈 Stats Summary",
        "📊 Univariate Analysis",
        "🔗 Bivariate Analysis"
    ])
    
    # Sub-tab 1: Stats Summary
    with sub_expl_tab1:
        st.header("📈 Descriptive Statistics Summary")
        st.markdown("Statistical distribution metrics for numerical columns.")
        
        with st.spinner("Fetching descriptive statistics..."):
            stats_data = api_request("get", f"/datasets/{selected_id}/stats")
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df.style.background_gradient(cmap="Purples", axis=0), use_container_width=True)

    # Sub-tab 2: Univariate Analysis
    with sub_expl_tab2:
        st.header("📊 Univariate Feature Analysis")
        st.markdown("Study individual variables in isolation via plots and distributions.")
        
        col_uni_left, col_uni_right = st.columns([1, 3])
        
        with col_uni_left:
            target_col = st.selectbox("Select Variable to Analyze", dataset_details["columns_list"])
            is_numeric = dataset_details["data_types"].get(target_col) == "numeric"
            
        with col_uni_right:
            if target_col:
                with st.spinner(f"Analyzing {target_col}..."):
                    res = api_request("post", f"/datasets/{selected_id}/univariate", json={"column": target_col})
                    if res:
                        st.subheader(f"Analysis: {target_col}")
                        
                        if is_numeric:
                            col_plot1, col_plot2 = st.columns(2)
                            with col_plot1:
                                if "histogram_fig" in res:
                                    st.plotly_chart(res["histogram_fig"], use_container_width=True)
                            with col_plot2:
                                if "kde_fig" in res:
                                    st.plotly_chart(res["kde_fig"], use_container_width=True)
                                    
                            # Percentiles
                            if "percentiles" in res:
                                st.markdown("#### Percentiles and Value Ranges")
                                st.dataframe(pd.Series(res["percentiles"]).to_frame(name="Value"), use_container_width=True)
                                
                            # Distribution
                            if "frequency_distribution" in res:
                                st.markdown("#### Binned Frequency Table")
                                st.dataframe(pd.DataFrame(res["frequency_distribution"]).head(10), use_container_width=True)
                        else:
                            col_plot1, col_plot2 = st.columns(2)
                            with col_plot1:
                                if "bar_fig" in res:
                                    st.plotly_chart(res["bar_fig"], use_container_width=True)
                            with col_plot2:
                                if "pie_fig" in res:
                                    st.plotly_chart(res["pie_fig"], use_container_width=True)
                                    
                            if "frequency_distribution" in res:
                                st.markdown("#### Category Counts & Frequencies")
                                st.dataframe(pd.DataFrame(res["frequency_distribution"]), use_container_width=True)

    # Sub-tab 3: Bivariate Analysis
    with sub_expl_tab3:
        st.header("🔗 Bivariate & Relationship Analysis")
        st.markdown("Discover correlations, dependencies, and testing significance between variables.")
        
        col_bi_left, col_bi_right = st.columns([1, 3])
        
        with col_bi_left:
            col_x = st.selectbox("Select Independent Variable (X)", dataset_details["columns_list"], index=0)
            col_y = st.selectbox("Select Dependent Variable (Y)", dataset_details["columns_list"], index=min(1, len(dataset_details["columns_list"])-1))
            
            is_x_num = dataset_details["data_types"].get(col_x) == "numeric"
            is_y_num = dataset_details["data_types"].get(col_y) == "numeric"
            
        with col_bi_right:
            if col_x and col_y:
                with st.spinner(f"Analyzing relationship between {col_x} and {col_y}..."):
                    res = api_request("post", f"/datasets/{selected_id}/bivariate", json={"column_x": col_x, "column_y": col_y})
                    if res:
                        st.subheader(f"Relationship: {col_y} vs {col_x}")
                        
                        if is_x_num and is_y_num:
                            if "pearson_correlation" in res:
                                col_m1, col_m2 = st.columns(2)
                                col_m1.metric("Pearson Correlation", f"{res['pearson_correlation']:.3f}", f"p-val: {res['pearson_p_value']:.4e}")
                                col_m2.metric("Spearman Correlation", f"{res['spearman_correlation']:.3f}", f"p-val: {res['spearman_p_value']:.4e}")
                            
                            if "scatter_fig" in res:
                                st.plotly_chart(res["scatter_fig"], use_container_width=True)
                                
                        elif (is_x_num and not is_y_num) or (not is_x_num and is_y_num):
                            if "violin_fig" in res:
                                st.plotly_chart(res["violin_fig"], use_container_width=True)
                            if "group_statistics" in res:
                                st.write("Group-wise Statistics Summary:")
                                st.dataframe(pd.DataFrame(res["group_statistics"]), use_container_width=True)
                                
                        else: # Cat vs Cat
                            if "stacked_bar_fig" in res:
                                st.plotly_chart(res["stacked_bar_fig"], use_container_width=True)
                            
                            # Chi-Square Details
                            if "chi_square_test" in res:
                                test = res["chi_square_test"]
                                if "error" not in test:
                                    st.write("📊 **Chi-Square Test of Independence**")
                                    col_t1, col_t2, col_t3 = st.columns(3)
                                    col_t1.metric("Chi2 Statistic", f"{test['chi2_statistic']:.3f}")
                                    col_t2.metric("p-value", f"{test['p_value']:.4e}")
                                    col_t3.metric("Statistically Significant?", "YES" if test['is_significant'] else "NO")
                            
                            if "contingency_table" in res:
                                st.write("Cross Tabulation contingency table:")
                                st.dataframe(pd.DataFrame(res["contingency_table"]), use_container_width=True)

        st.markdown("---")
        st.subheader("🗺️ Global Correlation Matrix")
        with st.spinner("Generating correlation heatmap..."):
            corr_res = api_request("get", f"/datasets/{selected_id}/correlation")
            if corr_res and "heatmap_fig" in corr_res:
                st.plotly_chart(corr_res["heatmap_fig"], use_container_width=True)
            else:
                st.info("Not enough numerical columns to compute a correlation matrix.")

# ----------------- TAB 3: ADVANCED ANALYTICS -----------------
with main_tabs[3]:
    sub_adv_tab1, sub_adv_tab2, sub_adv_tab3, sub_adv_tab4 = st.tabs([
        "🔮 Time Series Forecasting",
        "🤖 Predictive Modeling",
        "🗺️ Geographic Mapping",
        "🅰️ A/B Testing"
    ])
    
    # Sub-tab 1: Time Series Forecasting
    with sub_adv_tab1:
        st.header("🔮 Time Series Forecasting")
        st.markdown("Model trends and forecast future values based on chronological metrics.")
        
        date_candidates = [col for col, dtype in dataset_details["data_types"].items() if dtype in ["datetime", "object"]]
        num_cols = [col for col, dtype in dataset_details["data_types"].items() if dtype == "numeric"]
        
        if len(date_candidates) > 0 and len(num_cols) > 0:
            col_fore_l, col_fore_r = st.columns([1, 3])
            
            with col_fore_l:
                sel_date_col = st.selectbox("Select Date Column", date_candidates, key="fore_date")
                sel_val_col = st.selectbox("Select Value to Forecast", num_cols, key="fore_val")
                steps = st.slider("Forecast Steps (Periods)", 3, 24, 12)
                model_t = st.selectbox("Forecasting Algorithm", ["arima", "exponential_smoothing", "moving_average"])
                resample_freq = st.selectbox("Resampling Interval", ["ME (Monthly)", "W (Weekly)", "D (Daily)"])
                freq_code = resample_freq.split(" ")[0]
                
                run_fore = st.button("Run Time Series Forecast", type="primary")
                
            with col_fore_r:
                if run_fore:
                    with st.spinner("Fitting time series forecast..."):
                        payload = {
                            "date_col": sel_date_col,
                            "val_col": sel_val_col,
                            "steps": steps,
                            "model_type": model_t,
                            "freq": freq_code
                        }
                        res = api_request("post", f"/datasets/{selected_id}/forecast", json=payload)
                        if res:
                            if "forecast_fig" in res:
                                st.plotly_chart(res["forecast_fig"], use_container_width=True)
                                
                            if "metrics" in res:
                                metrics = res["metrics"]
                                st.subheader("📈 Model Calibration Metrics")
                                col_m1, col_m2, col_m3 = st.columns(3)
                                col_m1.metric("RMSE", f"{metrics['RMSE']:.3f}" if metrics['RMSE'] is not None else "N/A")
                                col_m2.metric("MAE", f"{metrics['MAE']:.3f}" if metrics['MAE'] is not None else "N/A")
                                col_m3.metric("MAPE", f"{metrics['MAPE']:.2f}%" if metrics['MAPE'] is not None else "N/A")
                                
                            if "forecast_df" in res:
                                st.subheader("🔮 Predictions Output Table")
                                st.dataframe(pd.DataFrame(res["forecast_df"]), use_container_width=True)
        else:
            st.warning("Forecasting requires at least one date column and one numerical values column.")

    # Sub-tab 2: Predictive Modeling
    with sub_adv_tab2:
        st.header("🤖 Machine Learning Predictive Modeling")
        st.markdown("Train supervised regressions or classifications, evaluate score thresholds, and identify key drivers.")
        
        num_cols = [col for col, dtype in dataset_details["data_types"].items() if dtype == "numeric"]
        
        if len(num_cols) >= 2:
            col_ml_l, col_ml_r = st.columns([1, 3])
            
            with col_ml_l:
                target = st.selectbox("Select Target Variable (Y)", dataset_details["columns_list"])
                features = st.multiselect("Select Feature Columns (X)", [c for c in dataset_details["columns_list"] if c != target])
                
                # Suggest model type based on target cardinality
                is_y_numeric = dataset_details["data_types"].get(target) == "numeric"
                
                task_mode = st.selectbox("ML Task Mode", ["Auto Detect", "Regression", "Classification"])
                
                if task_mode == "Regression" or (task_mode == "Auto Detect" and is_y_numeric):
                    algo = st.selectbox("Regression Algorithm", ["linear", "ridge", "decision_tree", "random_forest"])
                else:
                    algo = st.selectbox("Classification Algorithm", ["logistic", "decision_tree", "random_forest"])
                    
                train_btn = st.button("Train Predictive Model", type="primary")
                
            with col_ml_r:
                if train_btn:
                    if not features:
                        st.error("Please select at least one feature column.")
                    else:
                        with st.spinner("Training machine learning model in backend..."):
                            payload = {
                                "target": target,
                                "features": features,
                                "task_mode": task_mode,
                                "algo": algo
                            }
                            res = api_request("post", f"/datasets/{selected_id}/predictive-modeling", json=payload)
                            if res:
                                # 1. Performance scorecard
                                if "metrics" in res:
                                    metrics = res["metrics"]
                                    if "R2_Score" in metrics:
                                        st.subheader("🎯 Regression Scorecard")
                                        col_sc1, col_sc2, col_sc3 = st.columns(3)
                                        col_sc1.metric("R² Score (Goodness of Fit)", f"{metrics['R2_Score']:.3f}")
                                        col_sc2.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.3f}")
                                        col_sc3.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.3f}")
                                    else:
                                        st.subheader("🎯 Classification Scorecard")
                                        col_sc1, col_sc2, col_sc3, col_sc4 = st.columns(4)
                                        col_sc1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
                                        col_sc2.metric("Precision", f"{metrics['Precision']:.2%}")
                                        col_sc3.metric("Recall", f"{metrics['Recall']:.2%}")
                                        col_sc4.metric("F1 Score", f"{metrics['F1_Score']:.2%}")
                                        
                                # 2. Driver weights/coefficients
                                if "feature_importances" in res and res["feature_importances"]:
                                    st.subheader("💡 Feature Importances / Key Drivers")
                                    imp_df = pd.DataFrame({
                                        "Feature": list(res["feature_importances"].keys()),
                                        "Coefficient/Importance": list(res["feature_importances"].values())
                                    })
                                    # construct plotly figure
                                    fig = go.Figure(go.Bar(
                                        x=imp_df["Coefficient/Importance"],
                                        y=imp_df["Feature"],
                                        orientation='h',
                                        marker_color='#FF2E93' # Carmine Pink matching Contentsquare
                                    ))
                                    fig.update_layout(title="Driver Weights", yaxis={'categoryorder':'total ascending'})
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                # 3. Preview predictions comparison
                                if "comparison_df" in res:
                                    st.subheader("🔍 Predictions Comparison Preview")
                                    st.dataframe(pd.DataFrame(res["comparison_df"]).head(15), use_container_width=True)
        else:
            st.warning("Predictive modeling requires a numerical target and at least one feature column.")

    # Sub-tab 3: Geographic Mapping
    with sub_adv_tab3:
        st.header("🗺️ Spatial Geographic Visualizations")
        st.markdown("Render geospatial maps when GPS coordinates or state locations are present.")
        
        lat_cols = [col for col in dataset_details["columns_list"] if col.lower() in ['latitude', 'lat', 'y']]
        lon_cols = [col for col in dataset_details["columns_list"] if col.lower() in ['longitude', 'lon', 'lng', 'x']]
        state_cols = [col for col in dataset_details["columns_list"] if col.lower() in ['state', 'country', 'region']]
        num_cols = [col for col, dtype in dataset_details["data_types"].items() if dtype == "numeric"]
        
        col_map_l, col_map_r = st.columns([1, 3])
        
        with col_map_l:
            map_type = st.radio("Map Style", ["Scatter Mapbox (Plotly)", "Marker Cluster (Folium)", "Geospatial Heatmap (Folium)", "Choropleth (USA States)"])
            
            weight_col = st.selectbox("Weight Marker (Size/Color)", ["None"] + num_cols)
            w_val = None if weight_col == "None" else weight_col
            
            render_btn = st.button("Generate Interactive Map", type="primary")
            
        with col_map_r:
            if render_btn:
                payload = {
                    "map_style": map_type,
                    "lat_col": lat_cols[0] if lat_cols else None,
                    "lon_col": lon_cols[0] if lon_cols else None,
                    "state_col": state_cols[0] if state_cols else None,
                    "weight_col": w_val
                }
                with st.spinner("Rendering map..."):
                    res = api_request("post", f"/datasets/{selected_id}/geo-map", json=payload)
                    if res:
                        if "plotly_fig" in res:
                            st.plotly_chart(res["plotly_fig"], use_container_width=True)
                        elif "folium_html" in res:
                            html(res["folium_html"], height=500)

    # Sub-tab 4: A/B Testing
    with sub_adv_tab4:
        st.header("🅰️ Statistical A/B Testing")
        st.markdown("Compare conversion rates or transaction means between Test and Control groups.")
        
        col_ab_l, col_ab_r = st.columns([1, 3])
        
        with col_ab_l:
            # Pick categorical binary fields
            binary_cols = [col for col, dtype in dataset_details["data_types"].items() if dtype in ["object", "category", "numeric"]]
            ab_col_sel = st.selectbox("Select Variant Group Column (e.g. Group A/B)", binary_cols)
            
            # User defined controls
            var_a = st.text_input("Control Variant Label (A)", "A")
            var_b = st.text_input("Treatment Variant Label (B)", "B")
            
            test_metric = st.selectbox("Select Metric to Compare", dataset_details["columns_list"])
            is_metric_num = dataset_details["data_types"].get(test_metric) == "numeric"
            
            test_type = "T-Test"
            if not is_metric_num:
                test_type = st.selectbox("Select Test Type", ["Chi-Square Test", "Proportions Z-Test"])
                
            run_ab_btn = st.button("Evaluate A/B Significance", type="primary")
            
        with col_ab_r:
            if run_ab_btn:
                with st.spinner("Computing significance test in backend..."):
                    payload = {
                        "group_col": ab_col_sel,
                        "metric_col": test_metric,
                        "variant_a": var_a,
                        "variant_b": var_b,
                        "test_type": test_type
                    }
                    res = api_request("post", f"/datasets/{selected_id}/ab-test", json=payload)
                    if res:
                        st.subheader("📊 Hypothesis Test Outcomes")
                        if res.get("is_significant"):
                            st.success(f"🎉 **Statistically Significant:** {res.get('summary')}")
                        else:
                            st.info(f"⚖️ **Not Significant:** {res.get('summary')}")
                            
                        if "comparison_fig" in res:
                            st.plotly_chart(res["comparison_fig"], use_container_width=True)

# ----------------- TAB 4: REPORT CENTER -----------------
with main_tabs[4]:
    st.header("📥 Automated Document Export Pack")
    st.markdown("Download generated briefing packages containing statistical tables, insights, and forecast summaries.")
    
    if st.button("Generate Export Package (PDF, Excel, PPT)", type="primary"):
        with st.spinner("Compiling and generating reports on backend server..."):
            res = api_request("post", f"/datasets/{selected_id}/report")
            if res:
                st.success("Briefings package compiled successfully!")
                
                col_d1, col_d2, col_d3 = st.columns(3)
                
                pdf_dl = f"{BACKEND_URL}{res['pdf_url']}"
                excel_dl = f"{BACKEND_URL}{res['excel_url']}"
                ppt_dl = f"{BACKEND_URL}{res['ppt_url']}"
                
                with col_d1:
                    st.markdown(f'<a href="{pdf_dl}" target="_blank" style="text-decoration:none;"><button style="padding:10px;background-color:#E06666;color:white;border:none;border-radius:5px;width:100%;">📥 Download PDF Briefing</button></a>', unsafe_allow_html=True)
                with col_d2:
                    st.markdown(f'<a href="{excel_dl}" target="_blank" style="text-decoration:none;"><button style="padding:10px;background-color:#6AA84F;color:white;border:none;border-radius:5px;width:100%;">📥 Download Excel Workbooks</button></a>', unsafe_allow_html=True)
                with col_d3:
                    st.markdown(f'<a href="{ppt_dl}" target="_blank" style="text-decoration:none;"><button style="padding:10px;background-color:#4A86E8;color:white;border:none;border-radius:5px;width:100%;">📥 Download PowerPoint Slides</button></a>', unsafe_allow_html=True)
