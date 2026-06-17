import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def analyze_numerical_variable(df, column, bins=20):
    """
    Perform univariate analysis on a numerical variable.
    Returns:
        dict: A dictionary containing frequency distribution, percentiles, and interactive plotly figures.
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is either missing or not numerical.")
        
    series = df[column].dropna()
    if series.empty:
        return {}
        
    # Percentiles
    percentiles = {
        "1%": series.quantile(0.01),
        "5%": series.quantile(0.05),
        "10%": series.quantile(0.10),
        "25%": series.quantile(0.25),
        "50% (Median)": series.quantile(0.50),
        "75%": series.quantile(0.75),
        "90%": series.quantile(0.90),
        "95%": series.quantile(0.95),
        "99%": series.quantile(0.99)
    }
    
    # Frequency distribution / Histogram details
    counts, bin_edges = np.histogram(series, bins=bins)
    freq_df = pd.DataFrame({
        "Bin Start": bin_edges[:-1],
        "Bin End": bin_edges[1:],
        "Frequency": counts,
        "Percentage": (counts / len(series)) * 100
    })
    
    # Plotly Figures
    fig_hist = px.histogram(
        df, x=column, nbins=bins, 
        title=f"Histogram of {column}", 
        marginal="box", # Adds a box plot on top
        labels={column: column},
        color_discrete_sequence=['#636EFA']
    )
    fig_hist.update_layout(
        template="plotly_white",
        title_font=dict(size=18, family="Outfit, sans-serif"),
        hovermode="x"
    )
    
    # KDE approximation via Plotly violin/line
    fig_kde = px.violin(
        df, y=column, box=True, points="outliers",
        title=f"Violin & Box Plot of {column}",
        color_discrete_sequence=['#EF553B']
    )
    fig_kde.update_layout(
        template="plotly_white",
        title_font=dict(size=18, family="Outfit, sans-serif")
    )
    
    return {
        "percentiles": pd.Series(percentiles),
        "frequency_distribution": freq_df,
        "histogram_fig": fig_hist,
        "kde_fig": fig_kde
    }

def analyze_categorical_variable(df, column, max_categories=10):
    """
    Perform univariate analysis on a categorical variable.
    Returns:
        dict: A dictionary containing value counts, percentages, and plotly figures.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        
    series = df[column].dropna().astype(str)
    if series.empty:
        return {}
        
    # Frequency distribution
    counts = series.value_counts()
    percentages = series.value_counts(normalize=True) * 100
    
    summary_df = pd.DataFrame({
        "Category": counts.index,
        "Count": counts.values,
        "Percentage": percentages.values
    })
    
    # Cap categories for cleaner plotting if needed
    plot_df = summary_df.copy()
    if len(plot_df) > max_categories:
        top_cats = plot_df.head(max_categories)
        others_count = plot_df.iloc[max_categories:]["Count"].sum()
        others_pct = plot_df.iloc[max_categories:]["Percentage"].sum()
        
        # Append "Others" row using concat
        others_row = pd.DataFrame([{"Category": "Others", "Count": others_count, "Percentage": others_pct}])
        plot_df = pd.concat([top_cats, others_row], ignore_index=True)
        
    # Bar Chart
    fig_bar = px.bar(
        plot_df, x="Category", y="Count", text="Count",
        title=f"Bar Chart of {column} (Top {max_categories})",
        color="Count",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_bar.update_layout(
        template="plotly_white",
        title_font=dict(size=18, family="Outfit, sans-serif")
    )
    fig_bar.update_traces(textposition='outside')
    
    # Pie Chart
    fig_pie = px.pie(
        plot_df, names="Category", values="Count",
        title=f"Pie Chart of {column}",
        hole=0.4, # Donut chart
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie.update_layout(
        template="plotly_white",
        title_font=dict(size=18, family="Outfit, sans-serif")
    )
    
    return {
        "frequency_distribution": summary_df,
        "bar_fig": fig_bar,
        "pie_fig": fig_pie
    }
