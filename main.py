"""Light Source Power Dashboard - Streamlit Application."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime, timedelta

from data_loader import (
    load_all_datasets,
    get_filter_options,
    filter_measurements,
    get_linearity_data,
    get_short_term_stability_data,
    get_long_term_stability_data,
    get_global_metrics,
)

# Page configuration
st.set_page_config(
    page_title="Light Power QC",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for refined styling
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Main Header */
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20; /* Dark Green */
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    
    /* Card Styling using Native Container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); /* Soft shadow */
        border: 1px solid #e0e0e0 !important; /* Force border color */
    }
    
    /* Ensure padding inside the container */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 20px;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        color: #333;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid #ddd;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        font-weight: 600;
        color: #555;
    }
    .stTabs [aria-selected="true"] {
        color: #2E7D32; /* Active tab color */
    }
    
    /* Selectbox Styling */
    .stSelectbox > label {
        color: #1b5e20 !important;
        font-weight: 600;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all datasets with caching."""
    base_path = os.path.join(os.path.dirname(__file__), "light_source_power_datasets")
    return load_all_datasets(base_path)



def get_wavelength_color(wavelength: float) -> str:
    """Get color hex code based on wavelength."""
    try:
        wl = float(wavelength)
        if wl < 400: return '#7B1FA2' # UV - Purple
        if 400 <= wl < 450: return '#6200EA' # Violet
        if 450 <= wl < 495: return '#03A9F4' # Blue
        if 495 <= wl < 570: return '#4CAF50' # Green
        if 570 <= wl < 590: return '#FFEB3B' # Yellow
        if 590 <= wl < 620: return '#FF9800' # Orange
        if 620 <= wl < 750: return '#F44336' # Red
        if wl >= 750: return '#880E4F' # IR - Dark Red
    except (ValueError, TypeError):
        pass
    return '#4CAF50' # Default Theme Green


def create_linearity_graph(df: pd.DataFrame, title: str = "Power Linearity", base_color: str = '#4CAF50'):
    """Create linearity graph with regression line."""
    if df.empty or 'power_set_point' not in df.columns:
        return go.Figure().add_annotation(
            text="No linearity data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
    
    # Aggregate by power set point
    agg_df = df.groupby('power_set_point').agg({
        'power_mw': ['mean', 'std']
    }).reset_index()
    agg_df.columns = ['power_set_point', 'power_mean', 'power_std']
    
    fig = go.Figure()
    
    # Scatter plot with error bars
    fig.add_trace(go.Scatter(
        x=agg_df['power_set_point'],
        y=agg_df['power_mean'],
        error_y=dict(
            type='data',
            array=agg_df['power_std'].fillna(0),
            visible=True,
            color=base_color,
            thickness=1.5
        ),
        mode='markers',
        marker=dict(
            size=10,
            color=base_color,
            line=dict(width=1, color='white')
        ),
        name='Measured Power'
    ))
    
    # Linear regression
    if len(agg_df) >= 2:
        x = agg_df['power_set_point'].values
        y = agg_df['power_mean'].values
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(color=base_color, width=2, dash='dash'),
                name=f'Linear Fit (slope={coeffs[0]:.2f})'
            ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#262730')),
        xaxis_title="Power Set Point",
        yaxis_title="Power (mW)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=60, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
    )
    
    return fig


def create_stability_graph(
    df: pd.DataFrame, 
    title: str = "Power Stability",
    x_label: str = "Time",
    color_col: str = None,
    base_color: str = '#4CAF50'
):
    """Create stability graph showing power over time."""
    if df.empty or 'acquisition_datetime' not in df.columns:
        return go.Figure().add_annotation(
            text="No stability data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
    
    df_sorted = df.sort_values('acquisition_datetime')
    
    # If color_col is provided and exists, treat it as categorical for better coloring
    if color_col and color_col in df_sorted.columns:
        df_sorted[color_col] = df_sorted[color_col].astype(str)
        
        fig = px.line(
            df_sorted,
            x='acquisition_datetime',
            y='power_mw',
            color=color_col,
            # markers=True, -- Removed for cleaner look
            title=title,
            color_discrete_sequence=['#4CAF50', '#2E7D32', '#81C784'] # Palette
        )
        fig.update_traces(line=dict(width=2))
        
        # Update legend title
        fig.update_layout(legend_title_text=color_col.replace('_', ' ').replace('seconds', '(s)').title())
        
    else:
        # Fallback to simple line
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sorted['acquisition_datetime'],
            y=df_sorted['power_mw'],
            mode='lines', # Removed markers
            # marker=dict(size=6, color=base_color),
            line=dict(color=base_color, width=2),
            name='Power (mW)',
            # fill='tozeroy', -- Removed fill for cleaner look
        ))

    # Add mean line (global mean)
    mean_power = df_sorted['power_mw'].mean()
    fig.add_hline(
        y=mean_power,
        line_dash="dash",
        line_color="#E53E3E",
        annotation_text=f"Mean: {mean_power:.2f} mW",
        annotation_position="top right"
    )
    
    # Calculate time range for title
    if not df_sorted.empty and 'acquisition_datetime' in df_sorted.columns:
        start_t = df_sorted['acquisition_datetime'].min().strftime('%H:%M:%S')
        end_t = df_sorted['acquisition_datetime'].max().strftime('%H:%M:%S')
        title = f"{title} ({start_t} - {end_t})"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#262730')),
        xaxis_title=x_label,
        yaxis_title="Power (mW)",
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
        plot_bgcolor='white',
        height=400,
    )
    
    return fig


def create_trend_graph(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    title: str, 
    color_col: str = None
):
    """Create a trend graph for global metrics."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return go.Figure().add_annotation(
            text="No trend data available",
            xref="paper", yref="paper", format="text",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in df.columns else None,
        title=title,
        color_discrete_sequence=['#4CAF50', '#81C784', '#388E3C', '#A5D6A7'], # Theme Greens
        hover_data=['Dataset', 'light_source', 'power_meter']
    )
    
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color='DarkSlateGrey')))
    if color_col:
        fig.update_layout(legend_title_text=color_col.replace('_', ' ').title())
    
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=16, color='#262730'),
        plot_bgcolor='white',
    )
        
    return fig


def display_key_measurements(df: pd.DataFrame, light_source: str, power_meter: str, location: str):
    """Display key measurements as a formatted table."""
    if df.empty:
        st.info("No key measurements available for the selected filters.")
        return
    
    # Filter key measurements
    mask = (
        (df['light_source'] == light_source) &
        (df['power_meter'] == power_meter) &
        (df['measuring_location'] == location)
    )
    filtered = df[mask]
    
    if filtered.empty:
        st.info("No key measurements match the selected filters.")
        return
    
    # Display metrics in columns
    row = filtered.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Power", f"{row.get('power_mean_mw', 'N/A'):.2f} mW" if pd.notna(row.get('power_mean_mw')) else "N/A")
        st.metric("Std Power", f"{row.get('power_std_mw', 'N/A'):.3f} mW" if pd.notna(row.get('power_std_mw')) else "N/A")
    
    with col2:
        st.metric("Min Power", f"{row.get('power_min_mw', 'N/A'):.2f} mW" if pd.notna(row.get('power_min_mw')) else "N/A")
        st.metric("Max Power", f"{row.get('power_max_mw', 'N/A'):.2f} mW" if pd.notna(row.get('power_max_mw')) else "N/A")
    
    with col3:
        r2 = row.get('power_linearity_coefficient_of_determination')
        st.metric("R² (Linearity)", f"{r2:.4f}" if pd.notna(r2) else "N/A")
        slope = row.get('power_linearity_slope')
        st.metric("Slope", f"{slope:.3f}" if pd.notna(slope) else "N/A")
    
    with col4:
        short_stab = row.get('short_term_power_stability')
        st.metric("Short-term Stability", f"{short_stab:.4f}" if pd.notna(short_stab) else "N/A")
        long_stab = row.get('long_term_power_stability')
        st.metric("Long-term Stability", f"{long_stab:.4f}" if pd.notna(long_stab) else "N/A")
    
    # Show full table in expander
    with st.expander("View Full Single Measurements Table"):
        st.dataframe(filtered, use_container_width=True)


def main():
    """Main application."""
    # Header with logo
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "metrics_logo.png")
    
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
    with col_title:
        st.markdown('<h1 class="main-header">Light Power QC Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Quality control and analysis of light source power measurements</p>', unsafe_allow_html=True)
    
    # Load data
    try:
        datasets = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    if not datasets:
        st.warning("No datasets found. Please check the data folder.")
        return
        
    # Global Tab Navigation Removed - simplified view
    # main_tab1, main_tab2 = st.tabs(["Analysis Dashboard", "Metrics History"]) -- DELETED
    
    # with main_tab1: -- DELETED, indented content below shifted out
        
    # Dataset Analysis Layout
    with st.sidebar:
            # Get Metadata for selected items (Logic moved to main area after selection)


            # Display Metadata in Sidebar
            # Dataset selection (Keep dataset in sidebar as it loads the data context)
            dataset_names = sorted(datasets.keys())
            selected_dataset = st.selectbox(
                "Dataset",
                dataset_names,
                index=0
            )
            
            data = datasets[selected_dataset]
            options = get_filter_options(data)

            st.divider()


            # Input Parameters Section
            input_params = data.get('input_parameters', {})
            if input_params:
                with st.expander("Input Parameters"):
                    for key, value in input_params.items():
                        formatted_key = key.replace('_', ' ').title()
                        st.text_input(formatted_key, value=str(value), disabled=True)
        

    # Main content
    pm = data.get('power_measurements', pd.DataFrame())
    key_meas = data.get('key_measurements', pd.DataFrame())
    
    # 1. Key Measurements Container (Placeholder at Top)
    key_meas_container = st.container()

    # 2. Filters Row (Below Key Measurements)
    st.markdown("<h4 style='color: #2E7D32; margin-top: 20px; margin-bottom: 0px;'>Hardware Configuration</h4>", unsafe_allow_html=True)
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        selected_light_source = st.selectbox(
            "Light Source",
            options['light_sources'] if options['light_sources'] else ['No data'],
            index=0
        )
    with col_f2:
        selected_power_meter = st.selectbox(
            "Power Meter",
            options['power_meters'] if options['power_meters'] else ['No data'],
            index=0
        )
    with col_f3:
        selected_location = st.selectbox(
            "Measurement Location",
            options['measuring_locations'] if options['measuring_locations'] else ['No data'],
            index=0
        )

    # Get Metadata for selected items (Logic moved here)
    ls_df = data.get('light_sources', pd.DataFrame())
    pm_df = data.get('power_meters', pd.DataFrame())
    
    current_ls_meta = {}
    if not ls_df.empty and 'name' in ls_df.columns:
         matches = ls_df[ls_df['name'] == selected_light_source]
         if not matches.empty:
             current_ls_meta = matches.iloc[0].to_dict()

    current_pm_meta = {}
    if not pm_df.empty and 'name' in pm_df.columns:
         matches = pm_df[pm_df['name'] == selected_power_meter]
         if not matches.empty:
             current_pm_meta = matches.iloc[0].to_dict()



    # Display Metadata in Sidebar (Optional but useful context)
    if current_ls_meta or current_pm_meta:
        with st.sidebar:
            with st.container(border=True):
                st.markdown("<h3 style='color: #2E7D32; font-size: 1.1rem; margin-bottom: 15px;'>Hardware Details</h3>", unsafe_allow_html=True)
                
                # Light Source Details
                if current_ls_meta:
                    ls_md = f"**Light Source**: {selected_light_source}\n"
                    wl = current_ls_meta.get('wavelength_nm')
                    desc = current_ls_meta.get('description')
                    
                    if wl and pd.notna(wl):
                        ls_md += f"- Wavelength: {wl} nm\n"
                    if desc and pd.notna(desc):
                        if str(desc).startswith('http'):
                            ls_md += f"- Ref: [Link]({desc})\n"
                        else:
                            ls_md += f"- {desc}\n"
                    
                    st.markdown(ls_md)
                    st.divider()

                # Power Meter Details
                if current_pm_meta:
                    pm_md = f"**Power Meter**: {selected_power_meter}\n"
                    manufacturer = current_pm_meta.get('manufacturer')
                    model = current_pm_meta.get('model')
                    desc = current_pm_meta.get('description')
                    
                    if manufacturer and pd.notna(manufacturer):
                         pm_md += f"- Manufacturer: {manufacturer}\n"
                    if model and pd.notna(model):
                         pm_md += f"- Model: {model}\n"
                    if desc and pd.notna(desc):
                        pm_md += f"- {desc}\n"
                        
                    st.markdown(pm_md)

            # Sidebar Footer
            st.markdown("---")
            st.caption("Light Power QC Dashboard")

    
    # Determine Color Scheme
    theme_color = '#2E7D32' # Professional Green (Constant)
    # Wavelength-based coloring disabled for consistency
    # if current_ls_meta:
    #         wl = current_ls_meta.get('wavelength_nm')
    #         if wl:
    #             theme_color = get_wavelength_color(wl)
    
    # Filter data (No date range)
    filtered_pm = filter_measurements(
        pm,
        light_source=selected_light_source,
        power_meter=selected_power_meter,
        measuring_location=selected_location
    )
    
    # Key Measurements Section (Rendered into top container)
    with key_meas_container:
        with st.container(border=True):
            st.markdown(f"<h3 style='color: {theme_color}; margin-bottom: 20px;'>Key Measurements</h3>", unsafe_allow_html=True)
            display_key_measurements(key_meas, selected_light_source, selected_power_meter, selected_location)
    
    # Removed divider for cleaner look
    
    # Create 2 columns for the charts
    col_charts_1, col_charts_2 = st.columns(2)
    
    # --- Column 1: Linearity Section ---
    with col_charts_1:
        st.markdown(f"<h3 style='color: {theme_color}; margin-bottom: 20px;'>Power Linearity</h3>", unsafe_allow_html=True)
        
        lin_tabs = st.tabs(["Linearity Check"])
        with lin_tabs[0]:
            # Linearity data - use classified type if available
            if 'measurement_type' in filtered_pm.columns:
                linearity_data = filtered_pm[filtered_pm['measurement_type'] == 'Linearity']
            else:
                # Fallback if classification failed
                linearity_data = filtered_pm[filtered_pm['power_set_point'] < 1.0] if not filtered_pm.empty else pd.DataFrame()
            
            fig_linearity = create_linearity_graph(
                linearity_data,
                f"Linearity - {selected_light_source}",
                base_color=theme_color
            )
            st.plotly_chart(fig_linearity, use_container_width=True)
                
            # Show date range info
            if not key_meas.empty:
                mask = (
                    (key_meas['light_source'] == selected_light_source) &
                    (key_meas['power_meter'] == selected_power_meter) &
                    (key_meas['measuring_location'] == selected_location)
                )
                if mask.any():
                    row = key_meas[mask].iloc[0]
                    start = row.get('power_linearity_start_datetime', '')
                    end = row.get('power_linearity_end_datetime', '')
                    if start and str(start) != '0001-01-01T00:00:00':
                        st.caption(f"Linearity measurement period: {start} to {end}")

    # --- Column 2: Stability Section ---
    with col_charts_2:
        st.markdown(f"<h3 style='color: {theme_color}; margin-bottom: 20px;'>Power Stability</h3>", unsafe_allow_html=True)
        
        # Stability Overview - use classified type if available
        if 'measurement_type' in filtered_pm.columns:
            stability_data = filtered_pm[filtered_pm['measurement_type'].str.contains('Stability', na=False)].copy()
        else:
            # Fallback
            stability_data = filtered_pm[filtered_pm['power_set_point'] >= 0.95].copy() if not filtered_pm.empty else pd.DataFrame()
        
        if not stability_data.empty:
            # 1. Use existing classification
            type_column = 'measurement_type' if 'measurement_type' in stability_data.columns else 'Measurement Type'

            # 2. Iterate and display separate graphs for each group using Tabs
            if type_column in stability_data.columns:
                groups = sorted(stability_data[type_column].unique())
                
                # Filter out Linearity just in case
                groups = [g for g in groups if 'Stability' in g]
                
                if groups:
                    stab_tabs = st.tabs(groups)
                    
                    for tab, group_name in zip(stab_tabs, groups):
                        with tab:
                            group_data = stability_data[stability_data[type_column] == group_name]
                            
                            # Graph for this specific group
                            fig_stability = create_stability_graph(
                                group_data,
                                f"{group_name}", # Title is just the type
                                "Time",
                                color_col=None,
                                base_color=theme_color
                            )
                            # Reduce height slightly to fit well in column
                            fig_stability.update_layout(height=400)
                            st.plotly_chart(fig_stability, use_container_width=True)
                            
                            # Stats for this group
                            mean_p = group_data['power_mw'].mean()
                            std_p = group_data['power_mw'].std()
                            std_pct = (std_p / mean_p) * 100 if mean_p != 0 else 0
                            start_t = group_data['acquisition_datetime'].min()
                            end_t = group_data['acquisition_datetime'].max()
                            
                            sc1, sc2, sc3 = st.columns(3)
                            sc1.metric("Mean Power", f"{mean_p:.2f} mW")
                            sc2.metric("Std Dev", f"{std_p:.4f} mW")
                            sc3.metric("Variation", f"±{std_pct:.3f}%")
                            st.caption(f"Time Range: {start_t.strftime('%H:%M:%S')} - {end_t.strftime('%H:%M:%S')}")
            else:
                # Fallback single graph
                stab_tabs = st.tabs(["Overview"])
                with stab_tabs[0]:
                    fig_stability = create_stability_graph(
                        stability_data,
                        f"Stability Overview",
                        "Time",
                        base_color=theme_color
                    )
                    st.plotly_chart(fig_stability, use_container_width=True)

        else:
            stab_tabs = st.tabs(["Status"])
            with stab_tabs[0]:
                st.info("No stability measurements found.")
    
    # Raw data view
    st.divider()
    with st.expander("View Raw Power Measurements"):
        if not filtered_pm.empty:
            st.dataframe(filtered_pm, use_container_width=True)
        else:
            st.info("No measurements match the selected filters.")
    
    # Deleted Metrics History Tab logic


    # Footer
    st.divider()
    st.caption("Light Power QC Dashboard | Built with Streamlit")


if __name__ == "__main__":
    main()
