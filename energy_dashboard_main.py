"""
Energy Analytics & Forecasting Dashboard
CAISO Energy Market Analysis and ML Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Energy Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'forecast_trained' not in st.session_state:
    st.session_state.forecast_trained = False

# Sidebar Navigation
st.sidebar.title("⚡ Energy Analytics")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard Overview", "🦆 Duck Curve Analysis", "🔮 ML Forecasting", 
     "⚡ Generation Mix", "💰 Price Optimization", "🚨 Anomaly Detection"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(datetime.now() - timedelta(days=30), datetime.now()),
    max_value=datetime.now()
)

market = st.sidebar.selectbox(
    "Select Market",
    ["CAISO", "PJM", "ERCOT", "NYISO", "MISO"]
)

# ============================================================================
# DATA GENERATION FUNCTIONS (Replace with actual data sources)
# ============================================================================

@st.cache_data(ttl=3600)
def generate_sample_data(days=30):
    """Generate realistic sample energy data"""
    dates = pd.date_range(end=datetime.now(), periods=days*24*12, freq='5min')
    
    # Simulate realistic load pattern with duck curve characteristics
    hour_of_day = dates.hour + dates.minute/60
    day_of_week = dates.dayofweek
    
    # Base load pattern (duck curve)
    base_load = 25000 + 8000 * np.sin((hour_of_day - 6) * np.pi / 12)
    
    # Solar impact (creates duck curve dip)
    solar_impact = -5000 * np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12)) * (hour_of_day > 8) * (hour_of_day < 18)
    
    # Evening ramp (duck neck)
    evening_ramp = 3000 * np.exp(-((hour_of_day - 19)**2) / 2)
    
    # Weekend adjustment
    weekend_factor = 0.85 if day_of_week.any() >= 5 else 1.0
    
    # Add noise and seasonality
    noise = np.random.normal(0, 500, len(dates))
    seasonal = 2000 * np.sin(2 * np.pi * np.arange(len(dates)) / (365 * 24 * 12))
    
    load = (base_load + solar_impact + evening_ramp + seasonal + noise) * weekend_factor
    
    # Generation mix
    solar = np.maximum(0, 8000 * np.sin((hour_of_day - 6) * np.pi / 12) * (hour_of_day > 6) * (hour_of_day < 19))
    wind = 3000 + 2000 * np.random.randn(len(dates)).cumsum() / 100
    wind = np.clip(wind, 500, 6000)
    
    hydro = 5000 + 1000 * np.sin(2 * np.pi * np.arange(len(dates)) / (365 * 24 * 12))
    natural_gas = load - solar - wind - hydro - 2000
    nuclear = np.ones(len(dates)) * 2000
    
    # Price simulation (correlated with net load)
    net_load = load - solar - wind
    price = 30 + 0.002 * net_load + 10 * np.random.randn(len(dates))
    price = np.clip(price, 10, 200)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'load': load,
        'solar': solar,
        'wind': wind,
        'hydro': hydro,
        'natural_gas': natural_gas,
        'nuclear': nuclear,
        'price': price,
        'net_load': net_load
    })
    
    return df

@st.cache_data
def create_features(df):
    """Feature engineering for ML models"""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Lag features
    for lag in [1, 12, 24*12, 7*24*12]:  # 5min, 1hr, 1day, 1week
        df[f'load_lag_{lag}'] = df['load'].shift(lag)
    
    # Rolling statistics
    df['load_rolling_mean_24h'] = df['load'].rolling(window=24*12, min_periods=1).mean()
    df['load_rolling_std_24h'] = df['load'].rolling(window=24*12, min_periods=1).std()
    
    return df.dropna()

# ============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================================

if page == "📊 Dashboard Overview":
    st.markdown('<h1 class="main-header">⚡ Energy Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading market data...'):
        df = generate_sample_data(days=30)
        st.session_state.data_loaded = True
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_load = df['load'].iloc[-1]
        st.metric("Current Load", f"{current_load:,.0f} MW", 
                 delta=f"{((current_load / df['load'].iloc[-288]) - 1) * 100:.1f}%")
    
    with col2:
        avg_price = df['price'].tail(288).mean()
        st.metric("Avg Price (24h)", f"${avg_price:.2f}/MWh",
                 delta=f"{((avg_price / df['price'].iloc[-288]) - 1) * 100:.1f}%")
    
    with col3:
        renewable_pct = (df[['solar', 'wind']].sum(axis=1) / df['load'] * 100).iloc[-1]
        st.metric("Renewable %", f"{renewable_pct:.1f}%")
    
    with col4:
        peak_load = df['load'].tail(288).max()
        st.metric("Peak Load (24h)", f"{peak_load:,.0f} MW")
    
    with col5:
        net_load = df['net_load'].iloc[-1]
        st.metric("Net Load", f"{net_load:,.0f} MW")
    
    st.markdown("---")
    
    # Recent Load and Price Trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Load Trend (Last 7 Days)")
        recent_data = df.tail(7*24*12)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'], 
            y=recent_data['load'],
            mode='lines',
            name='Load',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'], 
            y=recent_data['net_load'],
            mode='lines',
            name='Net Load',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0),
                         hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Price Trend (Last 7 Days)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'], 
            y=recent_data['price'],
            mode='lines',
            name='Price',
            line=dict(color='#2ca02c', width=2),
            fill='tozeroy',
            fillcolor='rgba(44, 160, 44, 0.1)'
        ))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0),
                         yaxis_title="$/MWh")
        st.plotly_chart(fig, use_container_width=True)
    
    # Generation Mix
    st.subheader("⚡ Current Generation Mix")
    latest = df.iloc[-1]
    gen_data = {
        'Source': ['Solar', 'Wind', 'Hydro', 'Natural Gas', 'Nuclear'],
        'Generation': [latest['solar'], latest['wind'], latest['hydro'], 
                      latest['natural_gas'], latest['nuclear']],
        'Color': ['#FDB462', '#80B1D3', '#2E86AB', '#FB8072', '#B3DE69']
    }
    gen_df = pd.DataFrame(gen_data)
    
    fig = go.Figure(data=[go.Pie(
        labels=gen_df['Source'],
        values=gen_df['Generation'],
        marker=dict(colors=gen_df['Color']),
        hole=0.4,
        textinfo='label+percent',
        textfont_size=14
    )])
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: DUCK CURVE ANALYSIS
# ============================================================================

elif page == "🦆 Duck Curve Analysis":
    st.markdown('<h1 class="main-header">🦆 Duck Curve Analysis</h1>', unsafe_allow_html=True)
    
    df = generate_sample_data(days=30)
    
    # Average daily profile
    df_hourly = df.copy()
    df_hourly['hour'] = df_hourly['timestamp'].dt.hour + df_hourly['timestamp'].dt.minute / 60
    
    daily_profile = df_hourly.groupby('hour').agg({
        'load': 'mean',
        'net_load': 'mean',
        'solar': 'mean',
        'wind': 'mean'
    }).reset_index()
    
    # Duck curve visualization
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_profile['hour'],
        y=daily_profile['load'],
        mode='lines+markers',
        name='Gross Load',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_profile['hour'],
        y=daily_profile['net_load'],
        mode='lines+markers',
        name='Net Load (Duck Curve)',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=6),
        fill='tonexty',
        fillcolor='rgba(255, 127, 14, 0.1)'
    ))
    
    fig.update_layout(
        title="Average Daily Duck Curve Profile",
        xaxis_title="Hour of Day",
        yaxis_title="Load (MW)",
        height=500,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analysis metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        belly_depth = (daily_profile['load'].max() - daily_profile['net_load'].min())
        st.metric("Duck Belly Depth", f"{belly_depth:,.0f} MW",
                 help="Difference between peak load and minimum net load")
    
    with col2:
        ramp_rate = (daily_profile['net_load'].iloc[-5:].mean() - 
                    daily_profile['net_load'].iloc[32:37].mean()) / 5
        st.metric("Evening Ramp Rate", f"{ramp_rate:,.0f} MW/hr",
                 help="Rate of load increase during evening ramp")
    
    with col3:
        solar_penetration = (daily_profile['solar'].max() / daily_profile['load'].max() * 100)
        st.metric("Peak Solar Penetration", f"{solar_penetration:.1f}%")
    
    # Renewable contribution
    st.subheader("☀️ Solar and Wind Contribution")
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=daily_profile['hour'],
        y=daily_profile['solar'],
        mode='lines',
        name='Solar',
        fill='tozeroy',
        fillcolor='rgba(253, 180, 98, 0.6)',
        line=dict(color='#FDB462', width=2)
    ))
    
    fig2.add_trace(go.Scatter(
        x=daily_profile['hour'],
        y=daily_profile['wind'],
        mode='lines',
        name='Wind',
        fill='tozeroy',
        fillcolor='rgba(128, 177, 211, 0.6)',
        line=dict(color='#80B1D3', width=2)
    ))
    
    fig2.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Generation (MW)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# PAGE 3: ML FORECASTING
# ============================================================================

elif page == "🔮 ML Forecasting":
    st.markdown('<h1 class="main-header">🔮 Machine Learning Forecasting</h1>', unsafe_allow_html=True)
    
    st.info("🎯 **Goal: Achieve MAPE ≤ 5-7% for best-in-market predictions**")
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select ML Model",
            ["Ridge Regression", "Hist Gradient Boosting", "Ensemble (Recommended)"]
        )
    
    with col2:
        forecast_horizon = st.slider("Forecast Horizon (hours)", 1, 72, 24)
    
    if st.button("🚀 Train & Forecast", type="primary"):
        with st.spinner('Training models... This may take a minute'):
            # Load and prepare data
            df = generate_sample_data(days=60)
            df_features = create_features(df)
            
            # Split data
            train_size = int(len(df_features) * 0.8)
            train_data = df_features[:train_size]
            test_data = df_features[train_size:]
            
            # Feature columns
            feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend',
                           'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                           'load_lag_1', 'load_lag_12', 'load_lag_288', 'load_lag_2016',
                           'load_rolling_mean_24h', 'load_rolling_std_24h']
            
            X_train = train_data[feature_cols]
            y_train = train_data['load']
            X_test = test_data[feature_cols]
            y_test = test_data['load']
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            models = {}
            predictions = {}
            
            progress_bar = st.progress(0)
            
            if model_type == "Ridge Regression" or model_type == "Ensemble (Recommended)":
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_train_scaled, y_train)
                models['Ridge'] = ridge
                predictions['Ridge'] = ridge.predict(X_test_scaled)
                progress_bar.progress(33)
            
            if model_type == "Hist Gradient Boosting" or model_type == "Ensemble (Recommended)":
                hgb = HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    max_depth=8,
                    random_state=42
                )
                hgb.fit(X_train, y_train)
                models['HGB'] = hgb
                predictions['HGB'] = hgb.predict(X_test)
                progress_bar.progress(66)
            
            if model_type == "Ensemble (Recommended)":
                # Ensemble: average of predictions
                predictions['Ensemble'] = (predictions['Ridge'] + predictions['HGB']) / 2
                progress_bar.progress(100)
            
            # Evaluate models
            st.success("✅ Training complete!")
            
            st.subheader("📊 Model Performance")
            
            results = []
            for name, pred in predictions.items():
                mape = mean_absolute_percentage_error(y_test, pred) * 100
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                r2 = r2_score(y_test, pred)
                results.append({'Model': name, 'MAPE (%)': mape, 'RMSE': rmse, 'R²': r2})
            
            results_df = pd.DataFrame(results)
            
            # Display metrics
            cols = st.columns(len(results))
            for i, (col, result) in enumerate(zip(cols, results)):
                with col:
                    st.metric(
                        f"{result['Model']}",
                        f"{result['MAPE (%)']: .2f}%",
                        delta=f"R²: {result['R²']:.3f}",
                        delta_color="off"
                    )
            
            st.dataframe(results_df.style.highlight_min(subset=['MAPE (%)', 'RMSE'], color='lightgreen'), 
                        use_container_width=True)
            
            # Visualization
            st.subheader("📈 Forecast vs Actual")
            
            plot_data = test_data.head(forecast_horizon * 12).copy()  # 5-min intervals
            plot_data['Actual'] = y_test[:len(plot_data)]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=plot_data['timestamp'],
                y=plot_data['Actual'],
                mode='lines',
                name='Actual Load',
                line=dict(color='#1f77b4', width=3)
            ))
            
            # Add predictions for selected or all models
            colors = ['#ff7f0e', '#2ca02c', '#d62728']
            for idx, (name, pred) in enumerate(predictions.items()):
                fig.add_trace(go.Scatter(
                    x=plot_data['timestamp'],
                    y=pred[:len(plot_data)],
                    mode='lines',
                    name=f'{name} Forecast',
                    line=dict(color=colors[idx % len(colors)], width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f"{forecast_horizon}-Hour Forecast Comparison",
                xaxis_title="Time",
                yaxis_title="Load (MW)",
                height=500,
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (for tree-based models)
            if 'HGB' in models:
                st.subheader("🎯 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': models['HGB'].feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                title="Top 10 Most Important Features")
                fig_imp.update_layout(height=400)
                st.plotly_chart(fig_imp, use_container_width=True)

# ============================================================================
# PAGE 4: GENERATION MIX
# ============================================================================

elif page == "⚡ Generation Mix":
    st.markdown('<h1 class="main-header">⚡ Generation Mix Analysis</h1>', unsafe_allow_html=True)
    
    df = generate_sample_data(days=7)
    
    # Stacked area chart
    st.subheader("📊 Generation by Source (Last 7 Days)")
    
    fig = go.Figure()
    
    sources = [
        ('Nuclear', 'nuclear', '#B3DE69'),
        ('Hydro', 'hydro', '#2E86AB'),
        ('Natural Gas', 'natural_gas', '#FB8072'),
        ('Wind', 'wind', '#80B1D3'),
        ('Solar', 'solar', '#FDB462')
    ]
    
    for name, column, color in sources:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[column],
            mode='lines',
            name=name,
            stackgroup='one',
            fillcolor=color,
            line=dict(width=0.5, color=color)
        ))
    
    fig.update_layout(
        height=500,
        yaxis_title="Generation (MW)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Total Generation by Source")
        total_gen = df[['solar', 'wind', 'hydro', 'natural_gas', 'nuclear']].sum()
        gen_summary = pd.DataFrame({
            'Source': total_gen.index.str.title().str.replace('_', ' '),
            'Total (MWh)': total_gen.values,
            'Percentage': (total_gen.values / total_gen.sum() * 100)
        }).sort_values('Total (MWh)', ascending=False)
        
        st.dataframe(gen_summary.style.format({
            'Total (MWh)': '{:,.0f}',
            'Percentage': '{:.1f}%'
        }), use_container_width=True)
    
    with col2:
        st.subheader("♻️ Renewable vs Non-Renewable")
        renewable = df[['solar', 'wind', 'hydro']].sum().sum()
        non_renewable = df[['natural_gas', 'nuclear']].sum().sum()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Renewable', 'Non-Renewable'],
            values=[renewable, non_renewable],
            marker=dict(colors=['#2ca02c', '#d62728']),
            hole=0.4,
            textinfo='label+percent',
            textfont_size=16
        )])
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================================
# PAGE 5: PRICE OPTIMIZATION
# ============================================================================

elif page == "💰 Price Optimization":
    st.markdown('<h1 class="main-header">💰 Price Optimization & Trading</h1>', unsafe_allow_html=True)
    
    df = generate_sample_data(days=30)
    
    st.subheader("📊 Load vs Price Correlation")
    
    # Scatter plot
    fig = px.scatter(df.tail(7*24*12), x='net_load', y='price', 
                    color='price',
                    color_continuous_scale='RdYlGn_r',
                    title="Net Load vs Price Relationship",
                    labels={'net_load': 'Net Load (MW)', 'price': 'Price ($/MWh)'},
                    trendline="lowess")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Price heatmap by hour and day
    st.subheader("🔥 Price Heatmap")
    
    df_heatmap = df.tail(7*24*12).copy()
    df_heatmap['hour'] = df_heatmap['timestamp'].dt.hour
    df_heatmap['day_name'] = df_heatmap['timestamp'].dt.day_name()
    
    pivot_data = df_heatmap.pivot_table(values='price', index='hour', 
                                        columns='day_name', aggfunc='mean')
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data[[col for col in day_order if col in pivot_data.columns]]
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn_r',
        colorbar=dict(title="$/MWh")
    ))
    
    fig_heat.update_layout(
        title="Average Hourly Prices by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Hour of Day",
        height=500
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Trading opportunities
    st.subheader("💡 Trading Opportunities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        low_price_threshold = df['price'].quantile(0.25)
        low_price_hours = (df['price'] < low_price_threshold).sum() / 12  # Convert to hours
        st.metric("Low Price Hours (Q1)", f"{low_price_hours:.0f} hrs",
                 help="Hours suitable for charging/buying")
    
    with col2:
        high_price_threshold = df['price'].quantile(0.75)
        high_price_hours = (df['price'] > high_price_threshold).sum() / 12
        st.metric("High Price Hours (Q3)", f"{high_price_hours:.0f} hrs",
                 help="Hours suitable for discharging/selling")
    
    with col3:
        price_volatility = df['price'].std()
        st.metric("Price Volatility (σ)", f"${price_volatility:.2f}",
                 help="Standard deviation of prices")

# ============================================================================
# PAGE 6: ANOMALY DETECTION
# ============================================================================

elif page == "🚨 Anomaly Detection":
    st.markdown('<h1 class="main-header">🚨 Anomaly Detection</h1>', unsafe_allow_html=True)
    
    df = generate_sample_data(days=30)
    
    # Calculate rolling statistics
    df['load_rolling_mean'] = df['load'].rolling(window=24*12, center=True).mean()
    df['load_rolling_std'] = df['load'].rolling(window=24*12, center=True).std()
    
    # Detect anomalies (> 3 standard deviations)
    threshold = st.slider("Anomaly Threshold (Standard Deviations)", 1.5, 4.0, 3.0, 0.5)
    
    df['upper_bound'] = df['load_rolling_mean'] + threshold * df['load_rolling_std']
    df['lower_bound'] = df['load_rolling_mean'] - threshold * df['load_rolling_std']
    df['is_anomaly'] = ((df['load'] > df['upper_bound']) | (df['load'] < df['lower_bound']))
    
    anomalies = df[df['is_anomaly']].copy()
    
    # Visualization
    st.subheader(f"📈 Anomaly Detection (Found {len(anomalies)} anomalies)")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['load'],
        mode='lines',
        name='Load',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['upper_bound'],
        mode='lines',
        name='Upper Bound',
        line=dict(color='red', width=1, dash='dash'),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['lower_bound'],
        mode='lines',
        name='Lower Bound',
        line=dict(color='red', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        showlegend=True
    ))
    
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies['timestamp'],
            y=anomalies['load'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    fig.update_layout(
        height=500,
        hovermode='x unified',
        xaxis_title="Time",
        yaxis_title="Load (MW)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly details
    if len(anomalies) > 0:
        st.subheader("🔍 Anomaly Details")
        anomaly_summary = anomalies[['timestamp', 'load', 'load_rolling_mean', 'price']].copy()
        anomaly_summary['deviation'] = (
            (anomaly_summary['load'] - anomaly_summary['load_rolling_mean']) / 
            anomaly_summary['load_rolling_mean'] * 100
        )
        anomaly_summary.columns = ['Timestamp', 'Load (MW)', 'Expected Load (MW)', 
                                   'Price ($/MWh)', 'Deviation (%)']
        
        st.dataframe(
            anomaly_summary.style.format({
                'Load (MW)': '{:,.0f}',
                'Expected Load (MW)': '{:,.0f}',
                'Price ($/MWh)': '${:.2f}',
                'Deviation (%)': '{:+.1f}%'
            }),
            use_container_width=True,
            height=400
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Resources")
st.sidebar.markdown("""
- [CAISO OASIS](http://oasis.caiso.com)
- [EIA Data](https://www.eia.gov/electricity/data.php)
- [Documentation](#)
""")

st.sidebar.markdown("---")
st.sidebar.caption("Energy Analytics Dashboard v1.0 | Built with Streamlit")
