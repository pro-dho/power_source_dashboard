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

# Custom CSS for professional styling (Green Theme)
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #2F855A; /* Dark Green */
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 2rem;
    }
    .stApp {
        background-color: #F7FAFC;
    }
    .metric-card {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stDateInput label,
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {
        color: #2D3748 !important; /* Dark Gray */
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border: none;
        color: #718096;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #2F855A;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2F855A;  /* Professional Green */
        border-bottom: 2px solid #2F855A;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #2F855A;
    }
    /* Button primary color override */
    .stButton > button {
        background-color: #2F855A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all datasets with caching."""
    base_path = os.path.join(os.path.dirname(__file__), "light_source_power_datasets")
    return load_all_datasets(base_path)


def create_linearity_graph(df: pd.DataFrame, title: str = "Power Linearity"):
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
            color='rgba(47, 133, 90, 0.4)' # Green transparency
        ),
        mode='markers',
        marker=dict(
            size=10,
            color='#38A169', # Green 500
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
                line=dict(color='#2F855A', width=2, dash='dash'), # Dark Green
                name=f'Linear Fit (slope={coeffs[0]:.2f})'
            ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#2D3748')),
        xaxis_title="Power Set Point",
        yaxis_title="Power (mW)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=60, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    return fig


def create_stability_graph(
    df: pd.DataFrame, 
    title: str = "Power Stability",
    x_label: str = "Time"
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
    
    fig = go.Figure()
    
    # Line plot with markers
    fig.add_trace(go.Scatter(
        x=df_sorted['acquisition_datetime'],
        y=df_sorted['power_mw'],
        mode='lines+markers',
        marker=dict(size=6, color='#38A169'), # Green 500
        line=dict(color='#38A169', width=2),
        name='Power (mW)',
        fill='tozeroy',
        fillcolor='rgba(56, 161, 105, 0.1)' # Light Green transparency
    ))
    
    # Add mean line
    mean_power = df_sorted['power_mw'].mean()
    fig.add_hline(
        y=mean_power,
        line_dash="dash",
        line_color="#E53E3E", # Red for reference
        annotation_text=f"Mean: {mean_power:.2f} mW",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#2D3748')),
        xaxis_title=x_label,
        yaxis_title="Power (mW)",
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
        plot_bgcolor='white',
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
        color_discrete_sequence=px.colors.sequential.Greens_r, # Professional Green sequence
        hover_data=['Dataset', 'light_source', 'power_meter']
    )
    
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color='DarkSlateGrey')))
    if color_col:
        fig.update_layout(legend_title_text=color_col.replace('_', ' ').title())
    
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=16, color='#2D3748'),
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
    # Header
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
        
    # Global Tab Navigation
    main_tab1, main_tab2 = st.tabs(["Analysis Dashboard", "Metrics History"])
    
    with main_tab1:
        
        # Dataset Analysis Layout
        with st.sidebar:
            st.header("Filters")
            
            # Dataset selection
            dataset_names = sorted(datasets.keys())
            selected_dataset = st.selectbox(
                "Dataset",
                dataset_names,
                index=0
            )
            
            data = datasets[selected_dataset]
            options = get_filter_options(data)
            
            st.divider()
            
            # Light source selection
            selected_light_source = st.selectbox(
                "Light Source",
                options['light_sources'] if options['light_sources'] else ['No data'],
                index=0
            )
            
            # Power meter selection
            selected_power_meter = st.selectbox(
                "Power Meter",
                options['power_meters'] if options['power_meters'] else ['No data'],
                index=0
            )
            
            # Measurement location
            selected_location = st.selectbox(
                "Measurement Location",
                options['measuring_locations'] if options['measuring_locations'] else ['No data'],
                index=0
            )
            
            st.divider()
            
            # Date range for filtering
            st.subheader("Date Range (Raw Data)")
            pm = data.get('power_measurements', pd.DataFrame())
            
            start_date = None
            end_date = None
            
            if not pm.empty and 'acquisition_datetime' in pm.columns:
                min_date = pm['acquisition_datetime'].min()
                max_date = pm['acquisition_datetime'].max()
                
                if pd.notna(min_date) and pd.notna(max_date):
                    start_date = st.date_input(
                        "Start Date",
                        value=min_date.date(),
                        min_value=min_date.date(),
                        max_value=max_date.date()
                    )
                    end_date = st.date_input(
                        "End Date",
                        value=max_date.date(),
                        min_value=min_date.date(),
                        max_value=max_date.date()
                    )
        
        # Main content for Dataset Analysis
        pm = data.get('power_measurements', pd.DataFrame())
        key_meas = data.get('key_measurements', pd.DataFrame())
        
        # Filter data
        filtered_pm = filter_measurements(
            pm,
            light_source=selected_light_source,
            power_meter=selected_power_meter,
            measuring_location=selected_location,
            start_datetime=pd.Timestamp(start_date) if start_date else None,
            end_datetime=pd.Timestamp(end_date) + timedelta(days=1) if end_date else None,
        )
        
        # Key Measurements Section
        st.subheader("Key Measurements")
        display_key_measurements(key_meas, selected_light_source, selected_power_meter, selected_location)
        
        st.divider()
        
        # Graphs
        st.subheader("Analysis Graphs")
        
        # Create tabs for different graph types
        tab1, tab2, tab3 = st.tabs(["Linearity", "Short-term Stability", "Long-term Stability"])
        
        with tab1:
            # Linearity data - measurements with varying power set points
            linearity_data = filtered_pm[filtered_pm['power_set_point'] < 1.0] if not filtered_pm.empty else pd.DataFrame()
            if linearity_data.empty and not filtered_pm.empty:
                linearity_data = filtered_pm
            
            fig_linearity = create_linearity_graph(
                linearity_data,
                f"Power Linearity - {selected_light_source}"
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
        
        with tab2:
            # Short-term stability - short integration time
            short_term_data = get_short_term_stability_data(filtered_pm, integration_time=0.01)
            
            fig_short = create_stability_graph(
                short_term_data,
                f"Short-term Power Stability - {selected_light_source}",
                "Time (seconds)"
            )
            st.plotly_chart(fig_short, use_container_width=True)
            
            if not short_term_data.empty:
                std_pct = (short_term_data['power_mw'].std() / short_term_data['power_mw'].mean()) * 100
                st.caption(f"Variation: ±{std_pct:.2f}% (std/mean)")
        
        with tab3:
            # Long-term stability - longer integration time
            long_term_data = get_long_term_stability_data(filtered_pm, integration_time=0.1)
            
            fig_long = create_stability_graph(
                long_term_data,
                f"Long-term Power Stability - {selected_light_source}",
                "Time (minutes)"
            )
            st.plotly_chart(fig_long, use_container_width=True)
            
            if not long_term_data.empty:
                std_pct = (long_term_data['power_mw'].std() / long_term_data['power_mw'].mean()) * 100
                st.caption(f"Variation: ±{std_pct:.2f}% (std/mean)")
        
        # Raw data view
        st.divider()
        with st.expander("View Raw Power Measurements"):
            if not filtered_pm.empty:
                st.dataframe(filtered_pm, use_container_width=True)
            else:
                st.info("No measurements match the selected filters.")

    with main_tab2:
        st.markdown("### Metrics History")
        st.markdown("Overview of key measurements across all datasets.")
        
        global_metrics = get_global_metrics(datasets)
        
        if not global_metrics.empty:
            # Optional filters for the log
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                log_light_source = st.multiselect(
                    "Filter by Light Source",
                    options=sorted(global_metrics['light_source'].dropna().unique()),
                    default=[]
                )
            with col_f2:
                log_power_meter = st.multiselect(
                    "Filter by Power Meter",
                    options=sorted(global_metrics['power_meter'].dropna().unique()),
                    default=[]
                )
            
            filtered_log = global_metrics.copy()
            if log_light_source:
                filtered_log = filtered_log[filtered_log['light_source'].isin(log_light_source)]
            if log_power_meter:
                filtered_log = filtered_log[filtered_log['power_meter'].isin(log_power_meter)]
                
            # Trend Analysis
            st.subheader("Trends Over Time")
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown("**Stability Trend**")
                # Filter out NaNs for plot
                stab_df = filtered_log.dropna(subset=['short_term_power_stability', 'short_term_power_stability_start_datetime'])
                # Filter out '0001-01-01' dates
                if not stab_df.empty:
                    valid_dates = stab_df['short_term_power_stability_start_datetime'] > pd.Timestamp('1900-01-01')
                    stab_df = stab_df[valid_dates]
                
                if not stab_df.empty:
                    fig_trend_stab = create_trend_graph(
                        stab_df,
                        x_col='short_term_power_stability_start_datetime',
                        y_col='short_term_power_stability',
                        title='Short-term Stability over Time',
                        color_col='light_source'
                    )
                    st.plotly_chart(fig_trend_stab, use_container_width=True)
                else:
                    st.info("Not enough valid date data for stability trend.")

            with col_t2:
                st.markdown("**Linearity R² Trend**")
                lin_df = filtered_log.dropna(subset=['power_linearity_coefficient_of_determination', 'power_linearity_start_datetime'])
                if not lin_df.empty:
                    valid_dates = lin_df['power_linearity_start_datetime'] > pd.Timestamp('1900-01-01')
                    lin_df = lin_df[valid_dates]

                if not lin_df.empty:
                    fig_trend_lin = create_trend_graph(
                        lin_df,
                        x_col='power_linearity_start_datetime',
                        y_col='power_linearity_coefficient_of_determination',
                        title='Linearity R² over Time',
                        color_col='light_source'
                    )
                    st.plotly_chart(fig_trend_lin, use_container_width=True)
                else:
                    st.info("Not enough valid date data for linearity trend.")
            
            st.subheader("Detailed Log")
            # Customize columns for better display
            display_cols = [
                'Dataset', 'light_source', 'power_meter', 'measuring_location',
                'power_mean_mw', 'power_linearity_coefficient_of_determination',
                'short_term_power_stability', 'short_term_power_stability_start_datetime'
            ]
            # Ensure columns exist
            display_cols = [c for c in display_cols if c in filtered_log.columns]
            
            st.dataframe(
                filtered_log[display_cols],
                column_config={
                    "short_term_power_stability": st.column_config.ProgressColumn(
                        "Stability Score",
                        help="Short-term stability (0-1)",
                        format="%.4f",
                        min_value=0,
                        max_value=1,
                    ),
                    "power_linearity_coefficient_of_determination": st.column_config.NumberColumn(
                        "Linearity R²",
                        format="%.4f"
                    ),
                    "short_term_power_stability_start_datetime": st.column_config.DatetimeColumn(
                        "Date",
                        format="D MMM YYYY, HH:mm"
                    )
                },
                use_container_width=True,
                height=400
            )
            
        else:
            st.info("No global metrics available.")

    # Footer
    st.divider()
    st.caption("Light Power QC Dashboard | Built with Streamlit")


if __name__ == "__main__":
    main()
