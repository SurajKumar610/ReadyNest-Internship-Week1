import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# Add root folder to system path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loading import get_dataset_metadata
from src.data_cleaning import impute_missing_values, remove_duplicates, convert_data_types, handle_outliers, standardize_text
from src.descriptive_statistics import calculate_descriptive_statistics
from src.univariate_analysis import analyze_numerical_variable, analyze_categorical_variable
from src.bivariate_analysis import analyze_numerical_vs_numerical, analyze_numerical_vs_categorical, analyze_categorical_vs_categorical, generate_correlation_matrix
from src.visualization import plot_line_chart, plot_bar_chart, plot_pie_chart, plot_scatter_plot, plot_treemap, plot_sunburst, plot_sankey, plot_radar, plot_bubble, plot_waterfall, plot_candlestick
from src.insight_extraction import extract_insights
from src.feature_engineering import perform_label_encoding, perform_one_hot_encoding, scale_features, add_date_features, create_derived_features
from src.forecasting import time_series_forecast
from src.geo_mapping import generate_plotly_scatter_map, generate_folium_marker_map, generate_folium_heatmap, generate_plotly_choropleth_map
from src.predictive_modeling import run_regression_modeling, run_classification_modeling
from src.ab_testing import run_numerical_ab_test, run_conversion_ab_test, run_proportion_z_test
from src.reporting import generate_pdf_report, generate_excel_report, generate_ppt_report

# Page Config
st.set_page_config(
    page_title="Dataset Analytics & Reporting Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #F8F9FA;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1A2530 0%, #2A5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #2A5298;
        margin-bottom: 1rem;
    }
    
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1A2530;
    }
    
    .kpi-lbl {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    /* Tabs customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f3f5;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
        color: #495057;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2A5298 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Generate Mock Dataset for instant demo
@st.cache_data
def generate_mock_dataset():
    np.random.seed(42)
    rows = 1000
    dates = pd.date_range(start='2024-01-01', periods=rows, freq='h')
    
    categories = ['Electronics', 'Clothing', 'Home Decor', 'Office Supplies', 'Fitness']
    regions = ['North', 'East', 'South', 'West']
    states = ['NY', 'CA', 'TX', 'FL', 'IL', 'WA', 'MA', 'CO']
    
    # Latitudes/Longitudes centering around key state hubs
    state_coordinates = {
        'NY': (40.7128, -74.0060),
        'CA': (34.0522, -118.2437),
        'TX': (29.7604, -95.3698),
        'FL': (25.7617, -80.1918),
        'IL': (41.8781, -87.6298),
        'WA': (47.6062, -122.3321),
        'MA': (42.3601, -71.0589),
        'CO': (39.7392, -104.9903)
    }
    
    mock_states = np.random.choice(states, size=rows)
    lats = [state_coordinates[s][0] + np.random.normal(0, 0.5) for s in mock_states]
    lons = [state_coordinates[s][1] + np.random.normal(0, 0.5) for s in mock_states]
    
    sales = np.random.exponential(scale=200, size=rows) + np.random.normal(loc=50, scale=10, size=rows)
    sales = np.clip(sales, 10, None) # positive sales
    profit = sales * np.random.uniform(0.1, 0.45, size=rows) - np.random.exponential(scale=20, size=rows)
    
    ages = np.random.normal(loc=35, scale=12, size=rows).astype(int)
    ages = np.clip(ages, 18, 80)
    
    # Customer Churn (binary target)
    churn_prob = 1 / (1 + np.exp(-(-2.0 + 0.03 * ages - 0.001 * sales)))
    churn = np.random.binomial(1, churn_prob)
    
    # Experiment Group for A/B Testing
    groups = np.random.choice(['A', 'B'], size=rows, p=[0.5, 0.5])
    # Add conversion bump for group B
    conv_prob = np.where(groups == 'B', 0.14, 0.11)
    conversions = np.random.binomial(1, conv_prob)
    
    # Inject missing values for data cleaning demo
    sales_with_nan = sales.copy()
    nan_indices = np.random.choice(rows, size=int(rows * 0.05), replace=False)
    sales_with_nan[nan_indices] = np.nan
    
    # Inject outlier values
    sales_with_nan[np.random.choice(rows, size=5, replace=False)] = sales_with_nan.max() * 5
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales_with_nan,
        'Profit': profit,
        'Category': np.random.choice(categories, size=rows),
        'Region': np.random.choice(regions, size=rows),
        'State': mock_states,
        'Latitude': lats,
        'Longitude': lons,
        'Customer_Age': ages,
        'Churned': churn,
        'AB_Group': groups,
        'Converted': conversions
    })
    return df

# Main Header
st.markdown('<div class="main-header">📊 Dataset Analytics & Reporting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Load raw data, clean columns, build predictive models, run forecasting, and generate business reports instantly.</div>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("### ⚙️ Workspace Configuration")
data_source = st.sidebar.radio("Data Source Selection", ["Use Interactive Demo Dataset", "Upload CSV / Excel / JSON"])

uploaded_df = None
if data_source == "Use Interactive Demo Dataset":
    uploaded_df = generate_mock_dataset()
    st.sidebar.success("Demo dataset loaded successfully!")
else:
    file = st.sidebar.file_uploader("Upload dataset file", type=["csv", "xlsx", "json"])
    if file:
        file_ext = file.name.split('.')[-1]
        try:
            from src.data_loading import load_from_file
            uploaded_df = load_from_file(file, file_ext)
            st.sidebar.success("Uploaded file loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")

# Initialize Session State to store cleaned dataset across tabs
if 'df_cleaned' not in st.session_state or uploaded_df is not None:
    if uploaded_df is not None:
        st.session_state['df_raw'] = uploaded_df.copy()
        st.session_state['df_cleaned'] = uploaded_df.copy()
    else:
        st.session_state['df_raw'] = pd.DataFrame()
        st.session_state['df_cleaned'] = pd.DataFrame()

# Verify dataset is loaded
if st.session_state['df_cleaned'].empty:
    st.warning("Please upload a dataset or select the Demo dataset in the sidebar to get started.")
    st.stop()

# Short info in sidebar
row_count, col_count = st.session_state['df_cleaned'].shape
st.sidebar.markdown(f"**Current shape:** {row_count} rows × {col_count} columns")

# App Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🧼 Loading & Cleaning",
    "📈 Stats Summary",
    "📊 Univariate Analysis",
    "🔗 Bivariate Analysis",
    "⚙️ Feature Engineering",
    "🔮 Time Series Forecasting",
    "🗺️ Geographic Mapping",
    "🤖 Predictive Modeling",
    "🅰️ A/B Testing",
    "📥 Export Reports"
])

# ----------------- TAB 1: DATA CLEANING -----------------
with tab1:
    st.header("🧼 Data Cleaning Workspace")
    st.markdown("Prepare your dataset by resolving missing values, removing duplicates, capping outliers, and correcting data types.")
    
    df_raw = st.session_state['df_raw']
    df_cleaning = st.session_state['df_cleaned'].copy()
    
    col_clean_left, col_clean_right = st.columns([1, 2])
    
    with col_clean_left:
        st.subheader("Cleaning Options")
        
        # Drop duplicates
        if st.checkbox("Remove Duplicate Rows"):
            df_cleaning = remove_duplicates(df_cleaning)
            st.toast("Duplicates removed!")
            
        # Impute missing values
        null_cols = df_cleaning.columns[df_cleaning.isnull().any()].tolist()
        if null_cols:
            st.markdown("#### Missing Value Imputation")
            impute_col = st.selectbox("Select Column to Impute", null_cols)
            impute_strategy = st.selectbox("Select Imputation Strategy", ["mean", "median", "mode", "drop", "constant"])
            
            fill_value = None
            if impute_strategy == "constant":
                fill_value = st.text_input("Impute Fill Value")
                
            if st.button("Apply Imputation", key="btn_impute"):
                df_cleaning = impute_missing_values(df_cleaning, [impute_col], strategy=impute_strategy, fill_value=fill_value)
                st.session_state['df_cleaned'] = df_cleaning
                st.success(f"Imputed missing values in '{impute_col}'!")
                st.rerun()
        else:
            st.info("No missing values detected in the current dataset!")
            
        # Outlier handling
        numeric_cols = df_cleaning.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.markdown("#### Outlier Treatment")
            outlier_col = st.selectbox("Select Column for Outliers", numeric_cols)
            outlier_method = st.radio("Detection Method", ["iqr", "zscore"])
            outlier_action = st.radio("Action to Take", ["cap", "drop", "replace"])
            
            if st.button("Apply Outlier Treatment", key="btn_outliers"):
                df_cleaning = handle_outliers(df_cleaning, outlier_col, method=outlier_method, action=outlier_action)
                st.session_state['df_cleaned'] = df_cleaning
                st.success(f"Treated outliers in '{outlier_col}'!")
                st.rerun()
                
        # Reset data button
        if st.button("Reset Cleaned Data to Raw", type="primary"):
            st.session_state['df_cleaned'] = df_raw.copy()
            st.toast("Cleaned data reset successfully!")
            st.rerun()
            
    with col_clean_right:
        st.subheader("Dataset Schema Preview")
        
        meta = get_dataset_metadata(st.session_state['df_cleaned'])
        
        st.markdown(f"**Shape:** {meta['shape'][0]} rows, {meta['shape'][1]} columns")
        
        # Display sample rows
        st.write("First 5 rows of current dataset:")
        st.dataframe(meta['preview'].head(5), use_container_width=True)
        
        # Types and Missing summaries
        st.write("Column details (Data Types & Missing Value counts):")
        details_df = pd.DataFrame({
            "Data Type": pd.Series(meta['dtypes']),
            "Missing Values": pd.Series(meta['missing_values']),
            "Missing %": pd.Series({col: f"{(v / len(st.session_state['df_cleaned']) * 100):.1f}%" for col, v in meta['missing_values'].items()})
        })
        st.dataframe(details_df, use_container_width=True)

# ----------------- TAB 2: STATS SUMMARY -----------------
with tab2:
    st.header("📈 Descriptive Statistics Summary")
    st.markdown("Statistical distribution metrics for numerical columns.")
    
    stats_df = calculate_descriptive_statistics(st.session_state['df_cleaned'])
    st.dataframe(stats_df.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)
    
    # Automated insights summary block
    st.subheader("🧠 Automated Data Insights")
    insights = extract_insights(st.session_state['df_cleaned'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Key Findings**")
        for finding in insights['findings']:
            st.markdown(f"- {finding}")
            
    with col2:
        st.warning("📋 **Recommendations & Actions**")
        for rec in insights['recommendations']:
            st.markdown(f"**Rec:** {rec}")
        for act in insights['action_items']:
            st.markdown(f"**Action:** {act}")

# ----------------- TAB 3: UNIVARIATE ANALYSIS -----------------
with tab3:
    st.header("📊 Univariate Feature Analysis")
    st.markdown("Study individual variables in isolation via plots and distributions.")
    
    df = st.session_state['df_cleaned']
    
    col_uni_left, col_uni_right = st.columns([1, 3])
    
    with col_uni_left:
        target_col = st.selectbox("Select Variable to Analyze", df.columns.tolist())
        is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
        
    with col_uni_right:
        if is_numeric:
            analysis = analyze_numerical_variable(df, target_col)
            if analysis:
                st.subheader(f"Numerical Analysis: {target_col}")
                
                # Plotly Plots side by side
                col_plot1, col_plot2 = st.columns(2)
                with col_plot1:
                    st.plotly_chart(analysis["histogram_fig"], use_container_width=True)
                with col_plot2:
                    st.plotly_chart(analysis["kde_fig"], use_container_width=True)
                    
                # Metrics
                st.markdown("#### Percentiles and Value Ranges")
                st.dataframe(pd.DataFrame(analysis["percentiles"]).T, use_container_width=True)
                
                st.markdown("#### Binned Frequency Table")
                st.dataframe(analysis["frequency_distribution"].head(10), use_container_width=True)
        else:
            analysis = analyze_categorical_variable(df, target_col)
            if analysis:
                st.subheader(f"Categorical Analysis: {target_col}")
                
                col_plot1, col_plot2 = st.columns(2)
                with col_plot1:
                    st.plotly_chart(analysis["bar_fig"], use_container_width=True)
                with col_plot2:
                    st.plotly_chart(analysis["pie_fig"], use_container_width=True)
                    
                st.markdown("#### Category Counts & Frequencies")
                st.dataframe(analysis["frequency_distribution"], use_container_width=True)

# ----------------- TAB 4: BIVARIATE ANALYSIS -----------------
with tab4:
    st.header("🔗 Bivariate & Relationship Analysis")
    st.markdown("Discover correlations, dependencies, and testing significance between variables.")
    
    df = st.session_state['df_cleaned']
    
    col_bi_left, col_bi_right = st.columns([1, 3])
    
    with col_bi_left:
        col_x = st.selectbox("Select Independent Variable (X)", df.columns.tolist(), index=0)
        col_y = st.selectbox("Select Dependent Variable (Y)", df.columns.tolist(), index=min(1, len(df.columns)-1))
        
        is_x_num = pd.api.types.is_numeric_dtype(df[col_x])
        is_y_num = pd.api.types.is_numeric_dtype(df[col_y])
        
    with col_bi_right:
        # Case 1: Numeric vs Numeric
        if is_x_num and is_y_num:
            st.subheader(f"Correlation Analysis: {col_y} vs {col_x}")
            res = analyze_numerical_vs_numerical(df, col_x, col_y)
            if "error" not in res:
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Pearson Correlation", f"{res['pearson_correlation']:.3f}", f"p-val: {res['pearson_p_value']:.4e}")
                col_m2.metric("Spearman Correlation", f"{res['spearman_correlation']:.3f}", f"p-val: {res['spearman_p_value']:.4e}")
                
                st.plotly_chart(res["scatter_fig"], use_container_width=True)
            else:
                st.error(res["error"])
                
        # Case 2: Numeric vs Categorical
        elif (is_x_num and not is_y_num) or (not is_x_num and is_y_num):
            num = col_x if is_x_num else col_y
            cat = col_y if is_x_num else col_x
            
            st.subheader(f"Distribution Comparison: {num} by {cat}")
            res = analyze_numerical_vs_categorical(df, num, cat)
            if "error" not in res:
                st.plotly_chart(res["violin_fig"], use_container_width=True)
                st.write("Group-wise Statistics Summary:")
                st.dataframe(res["group_statistics"], use_container_width=True)
            else:
                st.error(res["error"])
                
        # Case 3: Categorical vs Categorical
        else:
            st.subheader(f"Contingency Analysis: {col_y} vs {col_x}")
            res = analyze_categorical_vs_categorical(df, col_x, col_y)
            if "error" not in res:
                st.plotly_chart(res["stacked_bar_fig"], use_container_width=True)
                
                # Chi-Square details
                test = res["chi_square_test"]
                if "error" not in test:
                    st.write("📊 **Chi-Square Test of Independence**")
                    col_t1, col_t2, col_t3 = st.columns(3)
                    col_t1.metric("Chi2 Statistic", f"{test['chi2_statistic']:.3f}")
                    col_t2.metric("p-value", f"{test['p_value']:.4e}")
                    col_t3.metric("Statistically Significant?", "YES" if test['is_significant'] else "NO")
                else:
                    st.warning(test["error"])
                    
                st.write("Cross Tabulation contingency table:")
                st.dataframe(res["contingency_table"], use_container_width=True)
            else:
                st.error(res["error"])
                
    st.markdown("---")
    st.subheader("🗺️ Global Correlation Matrix")
    corr_res = generate_correlation_matrix(df)
    if corr_res:
        st.plotly_chart(corr_res["heatmap_fig"], use_container_width=True)
    else:
        st.info("Not enough numerical columns to compute a correlation matrix.")

# ----------------- TAB 5: FEATURE ENGINEERING -----------------
with tab5:
    st.header("⚙️ Feature Engineering Workbench")
    st.markdown("Scale features, encode labels, extract temporal variables, or calculate business metrics.")
    
    df_fe = st.session_state['df_cleaned'].copy()
    
    col_fe_l, col_fe_r = st.columns([1, 2])
    
    with col_fe_l:
        st.subheader("Transformation Actions")
        
        # 1. Date features
        date_cols = df_fe.select_dtypes(include=['datetime', 'object']).columns.tolist()
        # Filter object columns that can represent date
        potential_dates = []
        for c in date_cols:
            try:
                # check if first non-null can be cast
                first_val = df_fe[c].dropna().iloc[0]
                pd.to_datetime(first_val)
                potential_dates.append(c)
            except:
                pass
                
        if potential_dates:
            date_col_sel = st.selectbox("Select Date Column", potential_dates)
            if st.button("Extract Date Features"):
                df_fe = add_date_features(df_fe, date_col_sel)
                st.session_state['df_cleaned'] = df_fe
                st.success("Extracted date parts (Year, Month, Quarter, Day, Weekend)!")
                st.rerun()
                
        # 2. Scaling
        num_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            scale_cols = st.multiselect("Select columns to scale", num_cols)
            scale_method = st.selectbox("Scaling Method", ["standard", "minmax"])
            if st.button("Apply Scaling"):
                df_fe, _ = scale_features(df_fe, scale_cols, method=scale_method)
                st.session_state['df_cleaned'] = df_fe
                st.success("Scaled selected columns successfully!")
                st.rerun()
                
        # 3. Categorical encoding
        cat_cols = df_fe.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            encode_cols = st.multiselect("Select columns to encode", cat_cols)
            encode_method = st.selectbox("Encoding Method", ["Label Encoding", "One-Hot Encoding"])
            if st.button("Apply Encoding"):
                if encode_method == "Label Encoding":
                    df_fe, _ = perform_label_encoding(df_fe, encode_cols)
                else:
                    df_fe = perform_one_hot_encoding(df_fe, encode_cols)
                st.session_state['df_cleaned'] = df_fe
                st.success("Encoded selected categories!")
                st.rerun()
                
        # 4. Custom Business Ratio
        st.markdown("#### Custom Ratios")
        ratio_type = st.selectbox("Derived Ratio Type", ["profit_margin", "custom_ratio"])
        if ratio_type == "custom_ratio":
            num_ratio = st.selectbox("Numerator Column", num_cols)
            den_ratio = st.selectbox("Denominator Column", num_cols)
            if st.button("Calculate Custom Ratio"):
                df_fe = create_derived_features(df_fe, 'custom_ratio', num_col=num_ratio, den_col=den_ratio)
                st.session_state['df_cleaned'] = df_fe
                st.success(f"Added custom ratio column: {num_ratio}_to_{den_ratio}_ratio")
                st.rerun()
        else:
            if st.button("Calculate Profit Margin"):
                df_fe = create_derived_features(df_fe, 'profit_margin')
                st.session_state['df_cleaned'] = df_fe
                st.success("Added profit margin column!")
                st.rerun()
                
    with col_fe_r:
        st.subheader("Transformed Dataset Preview")
        st.dataframe(st.session_state['df_cleaned'].head(15), use_container_width=True)

# ----------------- TAB 6: FORECASTING -----------------
with tab6:
    st.header("🔮 Time Series Forecasting")
    st.markdown("Model trends and forecast future values based on chronological metrics.")
    
    df = st.session_state['df_cleaned']
    
    # Filter potential datetime columns
    dt_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    if not dt_cols:
        # Check strings that could be dates
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(10))
                dt_cols.append(col)
            except:
                pass
                
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(dt_cols) > 0 and len(num_cols) > 0:
        col_fore_l, col_fore_r = st.columns([1, 3])
        
        with col_fore_l:
            date_col = st.selectbox("Select Date Column", dt_cols, key="fore_date")
            val_col = st.selectbox("Select Value to Forecast", num_cols, key="fore_val")
            steps = st.slider("Forecast Steps (Periods)", 3, 24, 12)
            model_t = st.selectbox("Forecasting Algorithm", ["arima", "exponential_smoothing", "moving_average"])
            resample_freq = st.selectbox("Resampling Interval", ["ME (Monthly)", "W (Weekly)", "D (Daily)"])
            freq_code = resample_freq.split(" ")[0]
            
            run_fore = st.button("Run Time Series Forecast", type="primary")
            
        with col_fore_r:
            if run_fore:
                with st.spinner("Fitting time series model..."):
                    try:
                        fore_results = time_series_forecast(df, date_col, val_col, forecast_steps=steps, model_type=model_t, freq=freq_code)
                        
                        # Plot
                        st.plotly_chart(fore_results["forecast_fig"], use_container_width=True)
                        
                        # Evaluation metrics
                        metrics = fore_results["metrics"]
                        st.subheader("📈 Model Calibration Metrics")
                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("RMSE", f"{metrics['RMSE']:.3f}" if not np.isnan(metrics['RMSE']) else "N/A")
                        col_m2.metric("MAE", f"{metrics['MAE']:.3f}" if not np.isnan(metrics['MAE']) else "N/A")
                        col_m3.metric("MAPE", f"{metrics['MAPE']:.2f}%" if not np.isnan(metrics['MAPE']) else "N/A")
                        
                        # Forecast Dataframe
                        st.subheader("🔮 Predictions Output Table")
                        st.dataframe(fore_results["forecast_df"], use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Forecasting calculation failed: {str(e)}")
    else:
        st.warning("Forecasting requires at least one date/timestamp column and one numerical values column.")

# ----------------- TAB 7: GEOGRAPHIC MAPPING -----------------
with tab7:
    st.header("🗺️ Spatial Geographic Visualizations")
    st.markdown("Render geospatial maps when GPS coordinates or state locations are present.")
    
    df = st.session_state['df_cleaned']
    
    lat_cols = [col for col in df.columns if col.lower() in ['latitude', 'lat', 'y']]
    lon_cols = [col for col in df.columns if col.lower() in ['longitude', 'lon', 'lng', 'x']]
    state_cols = [col for col in df.columns if col.lower() in ['state', 'country', 'region']]
    
    col_map_l, col_map_r = st.columns([1, 3])
    
    with col_map_l:
        map_type = st.radio("Map Style", ["Scatter Mapbox (Plotly)", "Marker Cluster (Folium)", "Geospatial Heatmap (Folium)", "Choropleth (USA States)"])
        
        # Determine features
        weight_options = ["None"] + df.select_dtypes(include=[np.number]).columns.tolist()
        weight_col = st.selectbox("Weight Marker (Size/Color)", weight_options)
        w_val = None if weight_col == "None" else weight_col
        
        render_btn = st.button("Generate Interactive Map", type="primary")
        
    with col_map_r:
        if render_btn:
            if map_type == "Choropleth (USA States)":
                if state_cols:
                    state_col_sel = state_cols[0]
                    if w_val:
                        fig = generate_plotly_choropleth_map(df, state_col_sel, w_val)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please choose a weight column for Choropleth shading.")
                else:
                    st.error("No state column found. Ensure your data has a column named 'State'.")
                    
            elif lat_cols and lon_cols:
                lat_sel = lat_cols[0]
                lon_sel = lon_cols[0]
                
                with st.spinner("Generating map view..."):
                    if map_type == "Scatter Mapbox (Plotly)":
                        fig = generate_plotly_scatter_map(df, lat_sel, lon_sel, size_col=w_val, color_col=w_val)
                        st.plotly_chart(fig, use_container_width=True)
                    elif map_type == "Marker Cluster (Folium)":
                        m = generate_folium_marker_map(df, lat_sel, lon_sel, popup_cols=[w_val] if w_val else None)
                        if m:
                            st_folium(m, width=800, height=500)
                        else:
                            st.error("Could not construct folium markers.")
                    else: # Heatmap
                        m = generate_folium_heatmap(df, lat_sel, lon_sel, weight_col=w_val)
                        if m:
                            st_folium(m, width=800, height=500)
                        else:
                            st.error("Could not construct folium heatmap.")
            else:
                st.error("Geospatial GPS mapping requires columns named 'Latitude' and 'Longitude' (or coordinates).")

# ----------------- TAB 8: PREDICTIVE MODELING -----------------
with tab8:
    st.header("🤖 Machine Learning Predictive Modeling")
    st.markdown("Train supervised regressions or classifications, evaluate score thresholds, and identify key drivers.")
    
    df = st.session_state['df_cleaned']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(num_cols) >= 2:
        col_ml_l, col_ml_r = st.columns([1, 3])
        
        with col_ml_l:
            target = st.selectbox("Select Target Variable (Y)", df.columns.tolist())
            features = st.multiselect("Select Feature Columns (X)", [c for c in df.columns if c != target])
            
            # Determine problem type based on Y type and cardinality
            is_y_numeric = pd.api.types.is_numeric_dtype(df[target])
            y_cardinality = len(df[target].dropna().unique())
            
            suggested_type = "Regression" if (is_y_numeric and y_cardinality > 10) else "Classification"
            st.info(f"Target distribution suggests: **{suggested_type}**")
            
            model_mode = st.selectbox("ML Task Mode", ["Auto Detect", "Regression", "Classification"])
            
            task = suggested_type
            if model_mode != "Auto Detect":
                task = model_mode
                
            if task == "Regression":
                algo = st.selectbox("Regression Algorithm", ["linear", "ridge", "decision_tree", "random_forest"])
            else:
                algo = st.selectbox("Classification Algorithm", ["logistic", "decision_tree", "random_forest"])
                
            train_btn = st.button("Train Predictive Model", type="primary")
            
        with col_ml_r:
            if train_btn:
                if not features:
                    st.error("Please select at least one feature column.")
                else:
                    with st.spinner("Training machine learning model..."):
                        try:
                            # Preprocess selected features to handle object fields locally via Label Encoding
                            df_ml = df.copy()
                            encoded_features = []
                            for col in features:
                                if col in df_ml.columns:
                                    if pd.api.types.is_numeric_dtype(df_ml[col]):
                                        encoded_features.append(col)
                                    else:
                                        # Simple label encoding inline
                                        df_ml[col] = df_ml[col].astype(str)
                                        df_ml[col] = pd.Categorical(df_ml[col]).codes
                                        encoded_features.append(col)
                                        
                            if task == "Regression":
                                results = run_regression_modeling(df_ml, encoded_features, target, model_type=algo)
                                
                                # Score KPI Cards
                                st.subheader("🎯 Regression Scorecard")
                                col_sc1, col_sc2, col_sc3 = st.columns(3)
                                col_sc1.metric("R² Score (Goodness of Fit)", f"{results['metrics']['R2_Score']:.3f}")
                                col_sc2.metric("Mean Absolute Error (MAE)", f"{results['metrics']['MAE']:.3f}")
                                col_sc3.metric("Root Mean Squared Error (RMSE)", f"{results['metrics']['RMSE']:.3f}")
                                
                            else: # Classification
                                results = run_classification_modeling(df_ml, encoded_features, target, model_type=algo)
                                
                                # Score KPI Cards
                                st.subheader("🎯 Classification Scorecard")
                                col_sc1, col_sc2, col_sc3, col_sc4 = st.columns(4)
                                col_sc1.metric("Accuracy", f"{results['metrics']['Accuracy']:.2%}")
                                col_sc2.metric("Precision", f"{results['metrics']['Precision']:.2%}")
                                col_sc3.metric("Recall", f"{results['metrics']['Recall']:.2%}")
                                col_sc4.metric("F1 Score", f"{results['metrics']['F1_Score']:.2%}")
                                
                            # Feature Importances Plot
                            st.subheader("💡 Feature Importances / Key Drivers")
                            importances = results["feature_importances"]
                            if importances:
                                imp_df = pd.DataFrame({
                                    "Feature": list(importances.keys()),
                                    "Coefficient/Importance": list(importances.values())
                                })
                                fig = plot_bar_chart(imp_df, "Feature", "Coefficient/Importance", title="Driver Weights")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Could not extract feature importances for this model.")
                                
                            # Predictions Preview
                            st.subheader("🔍 Predictions Comparison Preview")
                            st.dataframe(results["comparison_df"].head(15), use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
    else:
        st.warning("Predictive modeling requires a numerical target and at least one feature column.")

# ----------------- TAB 9: A/B TESTING -----------------
with tab9:
    st.header("🅰️ Statistical A/B Testing")
    st.markdown("Compare conversion rates or transaction means between Test and Control groups.")
    
    df = st.session_state['df_cleaned']
    
    col_ab_l, col_ab_r = st.columns([1, 3])
    
    with col_ab_l:
        ab_cols = [c for c in df.columns if len(df[c].dropna().unique()) == 2]
        if ab_cols:
            ab_col_sel = st.selectbox("Select Variant Group Column (e.g. Group A/B)", ab_cols)
            variants = list(df[ab_col_sel].dropna().unique())
            var_a = st.selectbox("Control Variant Label (A)", variants, index=0)
            var_b = st.selectbox("Treatment Variant Label (B)", [v for v in variants if v != var_a] + [var_a], index=0)
            
            test_metric = st.selectbox("Select Metric to Compare", df.columns.tolist())
            is_metric_num = pd.api.types.is_numeric_dtype(df[test_metric])
            
            test_type = "T-Test"
            if not is_metric_num:
                test_type = st.selectbox("Select Test Type", ["Chi-Square Test", "Proportions Z-Test"])
                
            run_ab_btn = st.button("Evaluate A/B Significance", type="primary")
        else:
            st.error("No A/B grouping columns (binary columns) detected in dataset.")
            run_ab_btn = False
            
    with col_ab_r:
        if run_ab_btn:
            with st.spinner("Computing significance test..."):
                try:
                    if is_metric_num:
                        # numerical test
                        res = run_numerical_ab_test(df, ab_col_sel, test_metric, variant_a_label=var_a, variant_b_label=var_b)
                    else:
                        # rate/conversion test
                        if test_type == "Proportions Z-Test":
                            res = run_proportion_z_test(df, ab_col_sel, test_metric, variant_a_label=var_a, variant_b_label=var_b)
                        else:
                            res = run_conversion_ab_test(df, ab_col_sel, test_metric, variant_a_label=var_a, variant_b_label=var_b)
                        
                    if "error" not in res:
                        st.subheader("📊 Hypothesis Test Outcomes")
                        
                        # Signifance banner
                        if res["is_significant"]:
                            st.success(f"🎉 **Statistically Significant:** {res['summary']}")
                        else:
                            st.info(f"⚖️ **Not Significant:** {res['summary']}")
                            
                        # Plot
                        st.plotly_chart(res["comparison_fig"], use_container_width=True)
                    else:
                        st.error(res["error"])
                except Exception as e:
                    st.error(f"Significance test computation failed: {str(e)}")

# ----------------- TAB 10: REPORT EXPORT -----------------
with tab10:
    st.header("📥 Export Reports")
    st.markdown("Download auto-generated PDF reports, Excel worksheets, or PowerPoint slide decks summarizing your analytical results.")
    
    df_c = st.session_state['df_cleaned']
    
    if st.button("Compile Executive Report Pack", type="primary"):
        with st.spinner("Compiling reports..."):
            try:
                # 1. Generate descriptive stats
                stats_df = calculate_descriptive_statistics(df_c)
                
                # 2. Extract insights
                insights = extract_insights(df_c)
                
                # 3. Write local report files
                pdf_path = "reports/pdf/executive_report.pdf"
                excel_path = "reports/excel/dataset_sheets.xlsx"
                ppt_path = "reports/ppt/executive_slides.pptx"
                
                generate_pdf_report(stats_df, insights, pdf_path)
                generate_excel_report(df_raw, df_c, stats_df, excel_path)
                generate_ppt_report(insights, ppt_path)
                
                st.success("✅ **Reports compiled successfully!** Local report folder structure populated.")
                
                # Render download buttons for each
                col_d1, col_d2, col_d3 = st.columns(3)
                
                with open(pdf_path, "rb") as f:
                    col_d1.download_button("Download PDF Document", f, file_name="executive_report.pdf", mime="application/pdf")
                    
                with open(excel_path, "rb") as f:
                    col_d2.download_button("Download Excel Workbook", f, file_name="dataset_sheets.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    
                with open(ppt_path, "rb") as f:
                    col_d3.download_button("Download PPT Presentation", f, file_name="executive_slides.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
                    
            except Exception as e:
                st.error(f"Failed compiling reports: {str(e)}")
