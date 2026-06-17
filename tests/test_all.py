import os
import sys
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loading import load_from_file, get_dataset_metadata
from src.data_cleaning import (
    impute_missing_values, remove_duplicates, convert_data_types,
    detect_outliers_iqr, detect_outliers_zscore, handle_outliers, standardize_text
)
from src.descriptive_statistics import calculate_descriptive_statistics
from src.univariate_analysis import analyze_numerical_variable, analyze_categorical_variable
from src.bivariate_analysis import (
    analyze_numerical_vs_numerical, analyze_numerical_vs_categorical,
    analyze_categorical_vs_categorical, generate_correlation_matrix
)
from src.visualization import (
    plot_line_chart, plot_bar_chart, plot_pie_chart, plot_scatter_plot,
    plot_treemap, plot_sunburst, plot_sankey, plot_radar, plot_bubble,
    plot_waterfall, plot_candlestick, generate_seaborn_pairplot
)
from src.insight_extraction import extract_insights
from src.feature_engineering import (
    perform_label_encoding, perform_one_hot_encoding, scale_features,
    add_date_features, create_derived_features
)
from src.forecasting import time_series_forecast
from src.geo_mapping import (
    generate_plotly_scatter_map, generate_plotly_choropleth_map,
    generate_folium_marker_map, generate_folium_heatmap
)
from src.predictive_modeling import run_regression_modeling, run_classification_modeling
from src.ab_testing import run_numerical_ab_test, run_conversion_ab_test, run_proportion_z_test
from src.reporting import generate_pdf_report, generate_excel_report, generate_ppt_report

class TestDatasetAnalysisPlatform(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a small realistic mock dataset
        cls.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        rows = 30
        
        dates = pd.date_range(start='2024-01-01', periods=rows, freq='D')
        categories = ['Electronics', 'Clothing', 'Home Decor']
        regions = ['North', 'East', 'South']
        states = ['NY', 'CA', 'TX']
        
        sales = np.random.exponential(scale=200, size=rows) + 50
        sales[5] = np.nan  # missing values
        sales[10] = 5000   # outlier
        profit = sales * 0.2
        
        cls.df = pd.DataFrame({
            'Date': dates,
            'Sales': sales,
            'Profit': profit,
            'Category': np.random.choice(categories, size=rows),
            'Region': np.random.choice(regions, size=rows),
            'State': np.random.choice(states, size=rows),
            'Latitude': np.random.uniform(25, 49, size=rows),
            'Longitude': np.random.uniform(-125, -70, size=rows),
            'Customer_Age': np.random.randint(18, 70, size=rows),
            'Churned': np.random.choice([0, 1], size=rows),
            'AB_Group': np.random.choice(['A', 'B'], size=rows),
            'Converted': np.random.choice([0, 1], size=rows)
        })
        
        # Save mock file
        cls.csv_path = os.path.join(cls.temp_dir, "mock_sales.csv")
        cls.df.to_csv(cls.csv_path, index=False)
        
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
        
    # Module 1: Data Loading
    def test_data_loading(self):
        loaded_df = load_from_file(self.csv_path, 'csv')
        self.assertEqual(loaded_df.shape[0], 30)
        self.assertIn('Sales', loaded_df.columns)
        
        meta = get_dataset_metadata(loaded_df)
        self.assertEqual(meta['shape'], (30, 12))
        self.assertIn('Sales', meta['columns'])
        
    # Module 2: Data Cleaning
    def test_data_cleaning(self):
        # Missing values imputation
        df_imputed = impute_missing_values(self.df, ['Sales'], strategy='mean')
        self.assertFalse(df_imputed['Sales'].isnull().any())
        
        # Duplicate removal
        df_dups = pd.concat([self.df, self.df.iloc[[0]]])
        df_dedup = remove_duplicates(df_dups)
        self.assertEqual(len(df_dedup), len(self.df))
        
        # Type conversion
        mapping = {'Sales': 'int', 'Category': 'category'}
        df_typed = convert_data_types(self.df, mapping)
        self.assertTrue(pd.api.types.is_integer_dtype(df_typed['Sales']))
        self.assertEqual(df_typed['Category'].dtype, 'category')
        
        # Outlier detection & handling
        outliers_iqr = detect_outliers_iqr(self.df, 'Sales')
        self.assertTrue(outliers_iqr[10]) # row 10 has the injected outlier (5000)
        
        outliers_z = detect_outliers_zscore(self.df, 'Sales')
        self.assertTrue(outliers_z[10])
        
        df_outliers_treated = handle_outliers(self.df, 'Sales', method='iqr', action='cap')
        self.assertLess(df_outliers_treated['Sales'].iloc[10], 5000)
        
        # Text standardization
        df_text = standardize_text(self.df, ['Category'], action='lowercase')
        self.assertTrue(df_text['Category'].str.islower().all())
        
    # Module 3: Descriptive Statistics
    def test_descriptive_stats(self):
        stats_df = calculate_descriptive_statistics(self.df, ['Profit', 'Customer_Age'])
        self.assertIn('Mean', stats_df.columns)
        self.assertIn('Profit', stats_df.index)
        self.assertIn('Customer_Age', stats_df.index)
        
    # Module 4: Univariate Analysis
    def test_univariate_analysis(self):
        # Numerical
        num_res = analyze_numerical_variable(self.df.dropna(), 'Customer_Age')
        self.assertIn('percentiles', num_res)
        self.assertIn('histogram_fig', num_res)
        self.assertIn('kde_fig', num_res)
        
        # Categorical
        cat_res = analyze_categorical_variable(self.df, 'Category')
        self.assertIn('frequency_distribution', cat_res)
        self.assertIn('bar_fig', cat_res)
        self.assertIn('pie_fig', cat_res)
        
    # Module 5: Bivariate Analysis
    def test_bivariate_analysis(self):
        # Num vs Num
        nvn = analyze_numerical_vs_numerical(self.df, 'Sales', 'Profit')
        self.assertIn('pearson_correlation', nvn)
        self.assertIn('scatter_fig', nvn)
        
        # Num vs Cat
        nvc = analyze_numerical_vs_categorical(self.df, 'Sales', 'Category')
        self.assertIn('group_statistics', nvc)
        self.assertIn('violin_fig', nvc)
        
        # Cat vs Cat
        cvc = analyze_categorical_vs_categorical(self.df, 'Category', 'Region')
        self.assertIn('contingency_table', cvc)
        self.assertIn('stacked_bar_fig', cvc)
        
        # Correlation Matrix
        corr_res = generate_correlation_matrix(self.df, ['Sales', 'Profit', 'Customer_Age'])
        self.assertIn('correlation_matrix', corr_res)
        self.assertIn('heatmap_fig', corr_res)
        
    # Module 6 & 12: Visualizations
    def test_visualizations(self):
        self.assertIsNotNone(plot_line_chart(self.df, 'Date', 'Sales'))
        self.assertIsNotNone(plot_bar_chart(self.df, 'Category', 'Sales'))
        self.assertIsNotNone(plot_pie_chart(self.df, 'Category', 'Sales'))
        self.assertIsNotNone(plot_scatter_plot(self.df, 'Sales', 'Profit'))
        self.assertIsNotNone(plot_treemap(self.df, ['Region', 'Category'], 'Sales'))
        self.assertIsNotNone(plot_sunburst(self.df, ['Region', 'Category'], 'Sales'))
        self.assertIsNotNone(plot_radar(self.df, 'Category', 'Sales'))
        self.assertIsNotNone(plot_bubble(self.df, 'Sales', 'Profit', 'Customer_Age'))
        self.assertIsNotNone(plot_waterfall(['A', 'B', 'C'], [10, -5, 5]))
        self.assertIsNotNone(plot_candlestick(
            pd.DataFrame({'D': ['2024-01-01'], 'O': [10], 'H': [15], 'L': [8], 'C': [12]}),
            'D', 'O', 'H', 'L', 'C'
        ))
        self.assertIsNotNone(generate_seaborn_pairplot(self.df, ['Sales', 'Profit']))
        
    # Module 7: Insight Extraction
    def test_insight_extraction(self):
        insights = extract_insights(self.df)
        self.assertIn('findings', insights)
        self.assertIn('recommendations', insights)
        self.assertIn('action_items', insights)
        self.assertGreater(len(insights['findings']), 0)
        
    # Module 9: Feature Engineering
    def test_feature_engineering(self):
        # Label Encoding
        df_le, encoders = perform_label_encoding(self.df, ['Category'])
        self.assertIn('Category', df_le.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(df_le['Category']))
        
        # One-Hot Encoding
        df_ohe = perform_one_hot_encoding(self.df, ['Region'])
        self.assertTrue(any(col.startswith('Region_') for col in df_ohe.columns))
        
        # Scaling
        df_scaled, scaler = scale_features(self.df, ['Sales', 'Profit'])
        self.assertAlmostEqual(df_scaled['Profit'].mean(), 0.0, places=2)
        
        # Date Features
        df_date = add_date_features(self.df, 'Date')
        self.assertIn('Date_year', df_date.columns)
        self.assertIn('Date_month', df_date.columns)
        
        # Derived
        df_derived = create_derived_features(self.df, 'profit_margin', profit='Profit', revenue='Sales')
        self.assertIn('profit_margin', df_derived.columns)
        
    # Module 10: Forecasting
    def test_forecasting(self):
        fore_res = time_series_forecast(self.df, 'Date', 'Profit', forecast_steps=3, model_type='arima', freq='D')
        self.assertIn('forecast_df', fore_res)
        self.assertIn('metrics', fore_res)
        self.assertIn('forecast_fig', fore_res)
        self.assertEqual(len(fore_res['forecast_df']), 3)
        
        # Test Moving Average forecasting
        fore_ma = time_series_forecast(self.df, 'Date', 'Profit', forecast_steps=3, model_type='moving_average', freq='D')
        self.assertIn('forecast_df', fore_ma)
        
        # Test Prophet fallback
        fore_prophet = time_series_forecast(self.df, 'Date', 'Profit', forecast_steps=3, model_type='prophet', freq='D')
        self.assertIn('forecast_df', fore_prophet)
        
    # Module 11: Geographic Mapping
    def test_geographic_mapping(self):
        self.assertIsNotNone(generate_plotly_scatter_map(self.df, 'Latitude', 'Longitude'))
        self.assertIsNotNone(generate_plotly_choropleth_map(self.df, 'State', 'Profit'))
        self.assertIsNotNone(generate_folium_marker_map(self.df, 'Latitude', 'Longitude'))
        self.assertIsNotNone(generate_folium_heatmap(self.df, 'Latitude', 'Longitude'))
        
    # Module 13: Predictive Modelling
    def test_predictive_modeling(self):
        # Regression (Test NameError is resolved!)
        reg_res = run_regression_modeling(self.df.dropna(), ['Customer_Age'], 'Profit', model_type='linear')
        self.assertIn('metrics', reg_res)
        self.assertIn('feature_importances', reg_res)
        self.assertIn('comparison_df', reg_res)
        self.assertIn('R2_Score', reg_res['metrics'])
        
        # Classification
        clf_res = run_classification_modeling(self.df.dropna(), ['Sales', 'Customer_Age'], 'Churned', model_type='logistic')
        self.assertIn('metrics', clf_res)
        self.assertIn('Accuracy', clf_res['metrics'])
        
    # Module 14: A/B Testing
    def test_ab_testing(self):
        # Numerical A/B Test
        res_num = run_numerical_ab_test(self.df, 'AB_Group', 'Sales')
        self.assertIn('p_value', res_num)
        self.assertIn('is_significant', res_num)
        
        # Conversion A/B Test (Chi-Square)
        res_conv = run_conversion_ab_test(self.df, 'AB_Group', 'Converted')
        self.assertIn('p_value', res_conv)
        self.assertIn('is_significant', res_conv)
        
        # Conversion A/B Test (Z-Test)
        res_z = run_proportion_z_test(self.df, 'AB_Group', 'Converted')
        self.assertIn('p_value', res_z)
        self.assertIn('z_statistic', res_z)
        
    # Module 15: Reporting
    def test_reporting(self):
        stats_df = calculate_descriptive_statistics(self.df)
        insights = extract_insights(self.df)
        
        pdf_out = os.path.join(self.temp_dir, "test_report.pdf")
        excel_out = os.path.join(self.temp_dir, "test_sheets.xlsx")
        ppt_out = os.path.join(self.temp_dir, "test_slides.pptx")
        
        self.assertTrue(generate_pdf_report(stats_df, insights, pdf_out))
        self.assertTrue(generate_excel_report(self.df, self.df, stats_df, excel_out))
        self.assertTrue(generate_ppt_report(insights, ppt_out))
        
        self.assertTrue(os.path.exists(pdf_out))
        self.assertTrue(os.path.exists(excel_out))
        self.assertTrue(os.path.exists(ppt_out))

if __name__ == '__main__':
    unittest.main()
