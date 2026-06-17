import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

def run_numerical_ab_test(df, group_col, metric_col, variant_a_label='A', variant_b_label='B', alpha=0.05):
    """
    Perform a two-sample Independent T-Test to compare numerical means between Variant A and Variant B.
    """
    if group_col not in df.columns or metric_col not in df.columns:
        raise ValueError("Group or Metric column not found in dataset.")
        
    group_a = df[df[group_col] == variant_a_label][metric_col].dropna()
    group_b = df[df[group_col] == variant_b_label][metric_col].dropna()
    
    if len(group_a) < 2 or len(group_b) < 2:
        return {"error": "Insufficient data in one or both groups for T-test (minimum 2 samples required)."}
        
    mean_a = group_a.mean()
    mean_b = group_b.mean()
    std_a = group_a.std()
    std_b = group_b.std()
    
    # Absolute difference and lift percentage
    diff = mean_b - mean_a
    lift = (diff / mean_a) * 100 if mean_a != 0 else 0
    
    # Perform t-test (Welch's t-test which does not assume equal variances)
    t_stat, p_val = stats.ttest_ind(group_b, group_a, equal_var=False)
    
    # Summary message
    direction = "increased" if diff >= 0 else "decreased"
    sig_text = "statistically significant" if p_val < alpha else "NOT statistically significant"
    summary_msg = (
        f"Variant {variant_b_label} {direction} average {metric_col} by {abs(lift):.2f}% "
        f"compared to Variant {variant_a_label} (Mean {variant_b_label}: {mean_b:.2f} vs "
        f"Mean {variant_a_label}: {mean_a:.2f}). p-value = {p_val:.4f}. Result is {sig_text}."
    )
    
    # Visualization: Box plot comparing the groups
    fig = px.box(
        df[df[group_col].isin([variant_a_label, variant_b_label])],
        x=group_col, y=metric_col, color=group_col,
        points="outliers",
        title=f"A/B Test: {metric_col} Comparison",
        color_discrete_sequence=['#9467bd', '#bcbd22']
    )
    fig.update_layout(template="plotly_white", title_font=dict(size=18, family="Outfit, sans-serif"))
    
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "std_a": std_a,
        "std_b": std_b,
        "difference": diff,
        "lift_percentage": lift,
        "t_statistic": t_stat,
        "p_value": p_val,
        "is_significant": p_val < alpha,
        "summary": summary_msg,
        "comparison_fig": fig
    }

def run_conversion_ab_test(df, group_col, conversion_col, variant_a_label='A', variant_b_label='B', alpha=0.05):
    """
    Perform a Chi-Square Test of independence to compare conversion/binary rates between groups.
    conversion_col should contain binary values (0 or 1, or True/False).
    """
    if group_col not in df.columns or conversion_col not in df.columns:
        raise ValueError("Group or Conversion column not found in dataset.")
        
    temp_df = df[df[group_col].isin([variant_a_label, variant_b_label])].copy()
    temp_df[conversion_col] = temp_df[conversion_col].astype(int)
    
    # Contingency table
    contingency = pd.crosstab(temp_df[group_col], temp_df[conversion_col])
    
    if contingency.shape != (2, 2):
        return {"error": "Contingency table must be 2x2. Make sure conversion column has binary values."}
        
    # Calculate conversion rates
    trials_a = contingency.loc[variant_a_label].sum()
    conversions_a = contingency.loc[variant_a_label, 1]
    rate_a = (conversions_a / trials_a) * 100 if trials_a > 0 else 0
    
    trials_b = contingency.loc[variant_b_label].sum()
    conversions_b = contingency.loc[variant_b_label, 1]
    rate_b = (conversions_b / trials_b) * 100 if trials_b > 0 else 0
    
    diff = rate_b - rate_a
    lift = (diff / rate_a) * 100 if rate_a > 0 else 0
    
    # Chi-Square Test
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    
    direction = "increased" if diff >= 0 else "decreased"
    sig_text = "statistically significant" if p_val < alpha else "NOT statistically significant"
    summary_msg = (
        f"Variant {variant_b_label} {direction} conversion rate by {abs(lift):.2f}% "
        f"compared to Variant {variant_a_label} (Rate {variant_b_label}: {rate_b:.2f}% vs "
        f"Rate {variant_a_label}: {rate_a:.2f}%). p-value = {p_val:.4f}. Result is {sig_text}."
    )
    
    # Plotly Bar chart for comparison
    plot_df = pd.DataFrame({
        "Variant": [variant_a_label, variant_b_label],
        "Conversion Rate (%)": [rate_a, rate_b]
    })
    
    fig = px.bar(
        plot_df, x="Variant", y="Conversion Rate (%)", text="Conversion Rate (%)",
        title=f"A/B Test: Conversion Rate Comparison",
        color="Variant",
        color_discrete_sequence=['#17becf', '#1f77b4']
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(template="plotly_white", yaxis_range=[0, max(rate_a, rate_b) * 1.25], title_font=dict(size=18, family="Outfit, sans-serif"))
    
    return {
        "trials_a": trials_a,
        "conversions_a": conversions_a,
        "rate_a_percent": rate_a,
        "trials_b": trials_b,
        "conversions_b": conversions_b,
        "rate_b_percent": rate_b,
        "difference_percent": diff,
        "lift_percentage": lift,
        "chi2_statistic": chi2,
        "p_value": p_val,
        "is_significant": p_val < alpha,
        "summary": summary_msg,
        "comparison_fig": fig
    }

def run_proportion_z_test(df, group_col, conversion_col, variant_a_label='A', variant_b_label='B', alpha=0.05):
    """
    Perform a two-sample Z-Test for proportions to compare conversion/binary rates between groups.
    conversion_col should contain binary values (0 or 1, or True/False).
    """
    if group_col not in df.columns or conversion_col not in df.columns:
        raise ValueError("Group or Conversion column not found in dataset.")
        
    temp_df = df[df[group_col].isin([variant_a_label, variant_b_label])].copy()
    temp_df[conversion_col] = temp_df[conversion_col].astype(int)
    
    # Contingency table
    contingency = pd.crosstab(temp_df[group_col], temp_df[conversion_col])
    
    if contingency.shape != (2, 2):
        return {"error": "Contingency table must be 2x2. Make sure conversion column has binary values."}
        
    trials_a = contingency.loc[variant_a_label].sum()
    conversions_a = contingency.loc[variant_a_label, 1]
    rate_a = (conversions_a / trials_a) if trials_a > 0 else 0
    
    trials_b = contingency.loc[variant_b_label].sum()
    conversions_b = contingency.loc[variant_b_label, 1]
    rate_b = (conversions_b / trials_b) if trials_b > 0 else 0
    
    if trials_a == 0 or trials_b == 0:
        return {"error": "Variant groups cannot be empty."}
        
    diff = rate_b - rate_a
    lift = (diff / rate_a) * 100 if rate_a > 0 else 0
    
    # Z-test formula
    p_pooled = (conversions_a + conversions_b) / (trials_a + trials_b)
    if p_pooled == 0 or p_pooled == 1:
        z_stat = 0.0
        p_val = 1.0
    else:
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / trials_a + 1 / trials_b))
        z_stat = diff / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
    direction = "increased" if diff >= 0 else "decreased"
    sig_text = "statistically significant" if p_val < alpha else "NOT statistically significant"
    summary_msg = (
        f"Variant {variant_b_label} {direction} conversion rate by {abs(lift):.2f}% "
        f"compared to Variant {variant_a_label} (Rate {variant_b_label}: {rate_b*100:.2f}% vs "
        f"Rate {variant_a_label}: {rate_a*100:.2f}%). Z-statistic = {z_stat:.4f}, p-value = {p_val:.4f}. Result is {sig_text}."
    )
    
    # Plotly Bar chart for comparison
    plot_df = pd.DataFrame({
        "Variant": [variant_a_label, variant_b_label],
        "Conversion Rate (%)": [rate_a * 100, rate_b * 100]
    })
    
    fig = px.bar(
        plot_df, x="Variant", y="Conversion Rate (%)", text="Conversion Rate (%)",
        title=f"A/B Test: Conversion Rate Comparison (Z-Test)",
        color="Variant",
        color_discrete_sequence=['#ff7f0e', '#2ca02c']
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(template="plotly_white", yaxis_range=[0, max(rate_a, rate_b) * 125], title_font=dict(size=18, family="Outfit, sans-serif"))
    
    return {
        "trials_a": trials_a,
        "conversions_a": conversions_a,
        "rate_a_percent": rate_a * 100,
        "trials_b": trials_b,
        "conversions_b": conversions_b,
        "rate_b_percent": rate_b * 100,
        "difference_percent": diff * 100,
        "lift_percentage": lift,
        "z_statistic": z_stat,
        "p_value": p_val,
        "is_significant": p_val < alpha,
        "summary": summary_msg,
        "comparison_fig": fig
    }
