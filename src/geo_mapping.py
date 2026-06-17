import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster, HeatMap

def generate_plotly_scatter_map(df, lat_col, lon_col, size_col=None, color_col=None, title=None):
    """
    Generate an interactive scatter map using Plotly Mapbox.
    Returns:
        plotly.graph_objects.Figure
    """
    title = title or "Geographic Scatter Map"
    temp_df = df[[lat_col, lon_col]].dropna()
    if temp_df.empty:
        raise ValueError("No latitude/longitude data available.")
        
    # Standard fallback style
    fig = px.scatter_map(
        df,
        lat=lat_col,
        lon=lon_col,
        size=size_col,
        color=color_col,
        zoom=3,
        title=title,
        map_style="open-street-map"
    )
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=18, family="Outfit, sans-serif")
    )
    return fig

def generate_plotly_choropleth_map(df, location_col, color_col, location_mode="USA-states", title=None):
    """
    Generate an interactive Choropleth map using Plotly.
    location_mode: "USA-states", "ISO-3", etc.
    """
    title = title or f"Choropleth Map: {color_col} by {location_col}"
    fig = px.choropleth(
        df,
        locations=location_col,
        color=color_col,
        locationmode=location_mode,
        title=title,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=18, family="Outfit, sans-serif")
    )
    return fig

def generate_folium_marker_map(df, lat_col, lon_col, popup_cols=None, zoom_start=4):
    """
    Generate a Folium map with marker clusters.
    Returns:
        folium.Map object
    """
    temp_df = df.dropna(subset=[lat_col, lon_col])
    if temp_df.empty:
        return None
        
    # Get center coordinates
    center_lat = temp_df[lat_col].mean()
    center_lon = temp_df[lon_col].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    marker_cluster = MarkerCluster().add_to(m)
    
    for _, row in temp_df.iterrows():
        popup_text = ""
        if popup_cols:
            popup_text = "<br>".join([f"<b>{col}:</b> {row[col]}" for col in popup_cols if col in row])
        else:
            popup_text = f"Lat: {row[lat_col]:.4f}, Lon: {row[lon_col]:.4f}"
            
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(marker_cluster)
        
    return m

def generate_folium_heatmap(df, lat_col, lon_col, weight_col=None, zoom_start=4):
    """
    Generate a Folium heat map.
    Returns:
        folium.Map object
    """
    temp_df = df.dropna(subset=[lat_col, lon_col])
    if temp_df.empty:
        return None
        
    center_lat = temp_df[lat_col].mean()
    center_lon = temp_df[lon_col].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    
    heat_data = []
    for _, row in temp_df.iterrows():
        if weight_col and weight_col in row and not pd.isna(row[weight_col]):
            heat_data.append([row[lat_col], row[lon_col], float(row[weight_col])])
        else:
            heat_data.append([row[lat_col], row[lon_col]])
            
    HeatMap(heat_data).add_to(m)
    return m
