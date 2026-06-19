import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Standard Chart Types
def plot_line_chart(df, x_col, y_col, color_col=None, title=None):
    title = title or f"Line Chart: {y_col} vs {x_col}"
    fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
    fig.update_layout(template="plotly_white", title_font=dict(size=16, family="Outfit, sans-serif"))
    return fig

def plot_bar_chart(df, x_col, y_col, color_col=None, barmode='group', title=None):
    title = title or f"Bar Chart: {y_col} vs {x_col}"
    fig = px.bar(df, x=x_col, y=y_col, color=color_col, barmode=barmode, title=title)
    fig.update_layout(template="plotly_white", title_font=dict(size=16, family="Outfit, sans-serif"))
    return fig

def plot_pie_chart(df, names_col, values_col, title=None):
    title = title or f"Pie Chart: Distribution of {names_col}"
    fig = px.pie(df, names=names_col, values=values_col, title=title, hole=0.3)
    fig.update_layout(template="plotly_white", title_font=dict(size=16, family="Outfit, sans-serif"))
    return fig

def plot_scatter_plot(df, x_col, y_col, size_col=None, color_col=None, title=None):
    title = title or f"Scatter Plot: {y_col} vs {x_col}"
    # Ensure size column has no NaNs if specified
    if size_col:
        size_min = df[size_col].min()
        if size_min < 0:
            # Shift size to positive values to avoid plotting errors
            size_data = df[size_col] - size_min + 1
        else:
            size_data = df[size_col].fillna(0)
        df_temp = df.copy()
        df_temp[size_col + "_scaled"] = size_data
        size_arg = size_col + "_scaled"
    else:
        df_temp = df
        size_arg = None
        
    fig = px.scatter(df_temp, x=x_col, y=y_col, size=size_arg, color=color_col, title=title)
    fig.update_layout(template="plotly_white", title_font=dict(size=16, family="Outfit, sans-serif"))
    return fig

# Advanced Chart Types (Module 12)
def plot_treemap(df, path_cols, values_col, title=None):
    """
    Treemap for hierarchical proportions.
    path_cols is list of category column names from root to leaf.
    """
    title = title or f"Treemap of {values_col} by {', '.join(path_cols)}"
    fig = px.treemap(df, path=path_cols, values=values_col, title=title)
    fig.update_layout(template="plotly_white", title_font=dict(size=16, family="Outfit, sans-serif"))
    return fig

def plot_sunburst(df, path_cols, values_col, title=None):
    """
    Sunburst for multi-level hierarchy.
    """
    title = title or f"Sunburst Chart of {values_col} by {', '.join(path_cols)}"
    fig = px.sunburst(df, path=path_cols, values=values_col, title=title)
    fig.update_layout(template="plotly_white", title_font=dict(size=16, family="Outfit, sans-serif"))
    return fig

def plot_sankey(df, source_col, target_col, value_col, title=None):
    """
    Sankey diagram showing flow between categories.
    """
    title = title or "Sankey Flow Diagram"
    
    # Get unique labels
    labels = list(pd.concat([df[source_col].astype(str), df[target_col].astype(str)]).unique())
    label_map = {label: i for i, label in enumerate(labels)}
    
    sources = df[source_col].astype(str).map(label_map).tolist()
    targets = df[target_col].astype(str).map(label_map).tolist()
    values = df[value_col].tolist()
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="blue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    fig.update_layout(title_text=title, font_size=12, template="plotly_white", title_font=dict(size=16, family="Outfit, sans-serif"))
    return fig

def plot_radar(df, categories_col, values_col, group_col=None, title=None):
    """
    Radar Chart for performance comparison.
    """
    title = title or f"Radar Chart of {values_col} by {categories_col}"
    
    fig = go.Figure()
    if group_col:
        groups = df[group_col].unique()
        for g in groups:
            sub_df = df[df[group_col] == g]
            fig.add_trace(go.Scatterpolar(
                r=sub_df[values_col].tolist(),
                theta=sub_df[categories_col].tolist(),
                fill='toself',
                name=str(g)
            ))
    else:
        fig.add_trace(go.Scatterpolar(
            r=df[values_col].tolist(),
            theta=df[categories_col].tolist(),
            fill='toself'
        ))
        
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True if group_col else False,
        title=title,
        template="plotly_white",
        title_font=dict(size=16, family="Outfit, sans-serif")
    )
    return fig

def plot_bubble(df, x_col, y_col, size_col, color_col=None, title=None):
    title = title or f"Bubble Chart: {y_col} vs {x_col} (Size: {size_col})"
    # Shift sizes to ensure strictly positive sizes for plotly bubble
    size_min = df[size_col].min()
    if size_min <= 0:
        sizes = df[size_col] - size_min + 1
    else:
        sizes = df[size_col]
        
    fig = px.scatter(df, x=x_col, y=y_col, size=sizes, color=color_col, title=title, size_max=60)
    fig.update_layout(template="plotly_white", title_font=dict(size=16, family="Outfit, sans-serif"))
    return fig

def plot_waterfall(labels, values, title=None):
    """
    Waterfall chart showing incremental changes.
    labels: list of strings (e.g. ['Start', 'Sales', 'Refunds', 'Total'])
    values: list of numbers (e.g. [100, 30, -10, 120])
    """
    title = title or "Waterfall Chart"
    
    # Calculate running sum or determine which is total
    measures = []
    for i, val in enumerate(values):
        if i == len(values) - 1:
            measures.append("total")
        elif i == 0:
            measures.append("absolute")
        else:
            measures.append("relative")
            
    fig = go.Figure(go.Waterfall(
        name="Waterfall",
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        textposition="outside",
        text=[f"{v:+}" if m == "relative" else f"{v}" for m, v in zip(measures, values)],
        connector=dict(line=dict(color="rgb(63, 63, 63)")),
    ))
    fig.update_layout(title=title, template="plotly_white", title_font=dict(size=16, family="Outfit, sans-serif"))
    return fig

def plot_candlestick(df, date_col, open_col, high_col, low_col, close_col, title=None):
    title = title or "Candlestick Chart"
    fig = go.Figure(data=[go.Candlestick(
        x=df[date_col],
        open=df[open_col],
        high=df[high_col],
        low=df[low_col],
        close=df[close_col]
    )])
    fig.update_layout(title=title, template="plotly_white", title_font=dict(size=16, family="Outfit, sans-serif"))
    return fig

# Seaborn Pairplot for local PDF reporting / image export
def generate_seaborn_pairplot(df, columns=None, hue=None):
    """
    Generates a Seaborn pairplot and returns it as a bytes buffer.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    if hue and hue in df.columns:
        if hue not in columns:
            plot_cols = columns + [hue]
        else:
            plot_cols = columns
        g = sns.pairplot(df[plot_cols], hue=hue, diag_kind='kde', palette='viridis')
    else:
        g = sns.pairplot(df[columns], diag_kind='kde')
        
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    buf.seek(0)
    return buf
