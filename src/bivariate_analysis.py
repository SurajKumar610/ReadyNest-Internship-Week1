import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr, chi2_contingency

def analyze_numerical_vs_numerical(df, x_col, y_col):
    """
    Analyze the relationship between two numerical variables.
    Returns:
        dict: A dictionary containing correlation metrics and a scatter plot.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("One or both columns do not exist in the DataFrame.")
        
    temp_df = df[[x_col, y_col]].dropna()
    if len(temp_df) < 2:
        return {"error": "Not enough data points after dropping missing values."}
        
    # Correlations
    pearson_coef, p_pearson = pearsonr(temp_df[x_col], temp_df[y_col])
    spearman_coef, p_spearman = spearmanr(temp_df[x_col], temp_df[y_col])
    
    # Scatter Plot with Trendline
    fig_scatter = px.scatter(
        temp_df, x=x_col, y=y_col,
        trendline="ols",
        title=f"Scatter Plot: {y_col} vs {x_col}",
        labels={x_col: x_col, y_col: y_col},
        opacity=0.7,
        color_discrete_sequence=['#1f77b4']
    )
    fig_scatter.update_layout(
        template="plotly_white",
        title_font=dict(size=18, family="Outfit, sans-serif")
    )
    
    return {
        "pearson_correlation": pearson_coef,
        "pearson_p_value": p_pearson,
        "spearman_correlation": spearman_coef,
        "spearman_p_value": p_spearman,
        "scatter_fig": fig_scatter
    }

def analyze_numerical_vs_categorical(df, num_col, cat_col):
    """
    Analyze the relationship between a numerical and a categorical variable.
    Returns:
        dict: A dictionary with statistics and box/violin plots.
    """
    if num_col not in df.columns or cat_col not in df.columns:
        raise ValueError("One or both columns do not exist in the DataFrame.")
        
    temp_df = df[[num_col, cat_col]].dropna()
    if temp_df.empty:
        return {"error": "No data available."}
        
    # Groupby statistics
    group_stats = temp_df.groupby(cat_col)[num_col].describe()
    
    # Violin Plot with Box representation inside
    fig_violin = px.violin(
        temp_df, x=cat_col, y=num_col, color=cat_col,
        box=True, points="outliers",
        title=f"Violin & Box Plot: {num_col} by {cat_col}",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_violin.update_layout(
        template="plotly_white",
        title_font=dict(size=18, family="Outfit, sans-serif"),
        showlegend=False
    )
    
    return {
        "group_statistics": group_stats,
        "violin_fig": fig_violin
    }

def analyze_categorical_vs_categorical(df, col1, col2):
    """
    Analyze the relationship between two categorical variables.
    Returns:
        dict: A dictionary with a contingency table, Chi-Square test result, and stacked bar chart.
    """
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("One or both columns do not exist in the DataFrame.")
        
    temp_df = df[[col1, col2]].dropna()
    if temp_df.empty:
        return {"error": "No data available."}
        
    # Contingency Table
    contingency_tab = pd.crosstab(temp_df[col1], temp_df[col2])
    contingency_pct = pd.crosstab(temp_df[col1], temp_df[col2], normalize='index') * 100
    
    # Chi-Square Test
    try:
        chi2, p_val, dof, expected = chi2_contingency(contingency_tab)
        chi_square_result = {
            "chi2_statistic": chi2,
            "p_value": p_val,
            "degrees_of_freedom": dof,
            "is_significant": p_val < 0.05
        }
    except Exception as e:
        chi_square_result = {"error": f"Chi-Square test could not be calculated: {str(e)}"}
        
    # Stacked Bar Chart (percentages)
    melted_pct = contingency_pct.reset_index().melt(id_vars=col1)
    fig_bar = px.bar(
        melted_pct, x=col1, y="value", color=col2,
        title=f"Stacked Bar Chart: Percentage distribution of {col2} by {col1}",
        labels={"value": "Percentage (%)"},
        barmode="stack",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_bar.update_layout(
        template="plotly_white",
        title_font=dict(size=16, family="Outfit, sans-serif")
    )
    
    return {
        "contingency_table": contingency_tab,
        "contingency_percentages": contingency_pct,
        "chi_square_test": chi_square_result,
        "stacked_bar_fig": fig_bar
    }

def generate_correlation_matrix(df, columns=None):
    """
    Generate correlation matrix for numeric columns and return correlation dataframe and heatmap figure.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    if len(columns) < 2:
        return None
        
    corr_matrix = df[columns].corr()
    
    # Heatmap Figure
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0, zmax=1.0,
        title="Correlation Heatmap Matrix",
        labels=dict(x="Features", y="Features", color="Correlation")
    )
    fig_heatmap.update_layout(
        template="plotly_white",
        title_font=dict(size=18, family="Outfit, sans-serif")
    )
    
    return {
        "correlation_matrix": corr_matrix,
        "heatmap_fig": fig_heatmap
    }
