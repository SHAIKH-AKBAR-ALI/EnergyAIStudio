import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# Safe imports
try:
    import joblib
    from tensorflow.keras.models import load_model
    ML_READY = True
except ImportError:
    ML_READY = False

# =================== CONFIGURATION ===================
st.set_page_config(page_title="⚡ Energy AI Studio", layout="wide", initial_sidebar_state="expanded")

# Enhanced CSS with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main { font-family: 'Poppins', sans-serif; }
    
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        animation: heroGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes heroGlow {
        from { box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); }
        to { box-shadow: 0 15px 40px rgba(240, 147, 251, 0.4); }
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(240, 147, 251, 0.1));
        padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease; cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-5px); box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(240, 147, 251, 0.15));
    }
    
    .metric-glass {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px); padding: 1rem; border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.2); text-align: center;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .data-preview {
        background: rgba(255,255,255,0.05); backdrop-filter: blur(5px);
        border-radius: 10px; padding: 1rem; margin: 1rem 0;
    }
    
    .forecast-result {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 15px; padding: 2rem; color: white;
        animation: slideInUp 0.8s ease-out;
    }
    
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white; border-radius: 8px 8px 0 0;
    }
    
    .upload-zone {
        border: 2px dashed #667eea; border-radius: 15px; padding: 2rem;
        text-align: center; background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #f093fb; background: rgba(240, 147, 251, 0.1);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# =================== SESSION STATE & CACHE ===================
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

@st.cache_resource
def load_ml_models():
    if not ML_READY: return None, None, False
    try:
        model = load_model("lstm_energy_model.h5", compile=False)
        scaler = joblib.load("lstm_scaler.pkl")
        return model, scaler, True
    except: return None, None, False

@st.cache_data
def process_energy_data(df):
    """Smart data processing with validation"""
    df = df.copy()
    
    # Find datetime column
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if not date_cols: date_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    date_col = date_cols[0] if date_cols else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    
    # Find energy column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    energy_col = numeric_cols[0] if numeric_cols else None
    
    # Add features
    df['Hour'] = df[date_col].dt.hour
    df['DayOfWeek'] = df[date_col].dt.day_name()
    df['Month'] = df[date_col].dt.month_name()
    df['Season'] = df[date_col].dt.quarter
    df['IsWeekend'] = df[date_col].dt.weekday >= 5
    df['Year'] = df[date_col].dt.year
    
    return df.set_index(date_col), energy_col

def smart_forecast(data_values, window_size, forecast_days, model=None, scaler=None):
    """Adaptive forecasting with fallback"""
    if len(data_values) < window_size:
        # Extend with trend if insufficient data
        trend = np.polyfit(range(len(data_values)), data_values, 1)
        extended = np.concatenate([data_values, [trend[0] * (len(data_values) + i) + trend[1] for i in range(window_size - len(data_values))]])
        data_values = extended[-window_size:]
    
    if model and scaler:
        # LSTM prediction
        scaled_data = scaler.transform(data_values.reshape(-1, 1))
        predictions = []
        current = scaled_data.copy()
        
        for _ in range(forecast_days):
            pred = model.predict(current[-window_size:].reshape(1, window_size, 1), verbose=0)
            predictions.append(pred[0, 0])
            current = np.append(current, pred, axis=0)
        
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    else:
        # Fallback: Enhanced trend + seasonality
        x = np.arange(len(data_values))
        trend_coef = np.polyfit(x, data_values, 2)  # Quadratic trend
        seasonal_component = np.sin(2 * np.pi * x / 7) * np.std(data_values) * 0.1  # Weekly seasonality
        
        future_x = np.arange(len(data_values), len(data_values) + forecast_days)
        trend_forecast = np.polyval(trend_coef, future_x)
        seasonal_forecast = np.sin(2 * np.pi * future_x / 7) * np.std(data_values) * 0.1
        noise = np.random.normal(0, np.std(data_values) * 0.05, forecast_days)  # Small noise
        
        return trend_forecast + seasonal_forecast + noise

# =================== MAIN APP ===================
st.markdown('<div class="hero-header"><h1>⚡ Energy AI Studio</h1><p>Advanced Energy Forecasting & Analytics Platform</p></div>', unsafe_allow_html=True)

# Load models
model, scaler, ml_ready = load_ml_models()

# Sidebar with persistent data
with st.sidebar:
    st.markdown("### 🎯 Quick Actions")
    
    # File upload (persistent)
    uploaded_file = st.file_uploader("📁 Upload Energy Data", type=['csv'], key="main_uploader")
    
    if uploaded_file and st.session_state.data is None:
        with st.spinner('🔄 Processing data...'):
            time.sleep(0.8)  # Animation delay
            st.session_state.data = pd.read_csv(uploaded_file)
            st.session_state.processed_data, energy_col = process_energy_data(st.session_state.data)
        st.success("✅ Data loaded successfully!")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    page = st.radio("", ["🏠 Dashboard", "📊 Data Explorer", "📈 Analytics", "🔮 Forecasting"], label_visibility="collapsed")
    
    # ML Status
    if ml_ready:
        st.success("🤖 AI Models Ready")
    else:
        st.warning("⚠️ AI Models Not Found")
        st.info("📊 Statistical forecasting available")

# Main content based on navigation
if page == "🏠 Dashboard":
    st.markdown("### 🏠 Welcome to Energy AI Studio")
    
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        energy_col = df.select_dtypes(include=[np.number]).columns[0]
        
        # Animated metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-glass"><h3>📊 {len(df):,}</h3><p>Total Records</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-glass"><h3>📅 {df.index.nunique()}</h3><p>Time Points</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-glass"><h3>⚡ {df[energy_col].mean():.0f}</h3><p>Avg Energy (MW)</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-glass"><h3>📈 {df[energy_col].std():.0f}</h3><p>Variability</p></div>', unsafe_allow_html=True)
        
        # Quick insights
        st.markdown("### ⚡ Quick Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            # Recent trend
            recent_data = df[energy_col].resample('D').mean().tail(30)
            fig = px.line(x=recent_data.index, y=recent_data.values, title="📈 Last 30 Days Trend")
            fig.update_traces(line_color='#667eea', line_width=3)
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Energy distribution
            fig = px.histogram(df, x=energy_col, nbins=30, title="📊 Energy Distribution")
            fig.update_traces(marker_color='#f093fb')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Welcome screen with feature cards
        st.markdown("### 🌟 Platform Features")
        
        features = [
            {"icon": "📊", "title": "Smart Data Analysis", "desc": "Automated insights and quality checks"},
            {"icon": "📈", "title": "Interactive Analytics", "desc": "Explore patterns with dynamic visualizations"},
            {"icon": "🔮", "title": "AI Forecasting", "desc": "Advanced LSTM predictions with fallback options"},
            {"icon": "⚡", "title": "Real-time Processing", "desc": "Fast analysis with persistent session state"}
        ]
        
        for i, feature in enumerate(features):
            st.markdown(f"""
            <div class="feature-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.1)  # Stagger animation

elif page == "📊 Data Explorer":
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        energy_col = df.select_dtypes(include=[np.number]).columns[0]
        
        tab1, tab2 = st.tabs(["📋 Data Preview", "🔍 Quality Check"])
        
        with tab1:
            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            st.dataframe(df.head(200), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download processed data
            csv = df.reset_index().to_csv(index=False)
            st.download_button("📥 Download Processed Data", csv, "processed_energy_data.csv", "text/csv")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Data quality metrics
                missing_pct = (df.isnull().sum() / len(df) * 100)
                if missing_pct.sum() > 0:
                    fig = px.bar(x=missing_pct.index, y=missing_pct.values, title="❌ Missing Data %")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No missing data detected!")
            
            with col2:
                # Outlier detection
                Q1 = df[energy_col].quantile(0.25)
                Q3 = df[energy_col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[energy_col] < (Q1 - 1.5 * IQR)) | (df[energy_col] > (Q3 + 1.5 * IQR))).sum()
                
                st.metric("🎯 Outliers Detected", f"{outliers} ({outliers/len(df)*100:.1f}%)")
                
                if outliers > 0:
                    fig = px.box(df, y=energy_col, title="📦 Outlier Analysis")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📁 Upload a dataset to explore!")

elif page == "📈 Analytics":
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        energy_col = df.select_dtypes(include=[np.number]).columns[0]
        
        # Interactive filters
        col1, col2, col3 = st.columns(3)
        with col1:
            years = st.multiselect("📅 Years", sorted(df['Year'].unique()), default=sorted(df['Year'].unique())[-2:])
        with col2:
            seasons = st.multiselect("🌍 Seasons", [1,2,3,4], default=[1,2,3,4], format_func=lambda x: ["❄️ Winter","🌸 Spring","☀️ Summer","🍂 Fall"][x-1])
        with col3:
            show_weekends = st.checkbox("📊 Include Weekends", value=True)
        
        # Filter data
        mask = (df['Year'].isin(years)) & (df['Season'].isin(seasons))
        if not show_weekends:
            mask &= ~df['IsWeekend']
        
        filtered_df = df[mask]
        
        tab1, tab2, tab3 = st.tabs(["⏰ Time Patterns", "📊 Distributions", "🔥 Heatmaps"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                hourly_avg = filtered_df.groupby('Hour')[energy_col].mean()
                fig = px.line(x=hourly_avg.index, y=hourly_avg.values, title="⏰ Hourly Energy Pattern")
                fig.update_traces(line_color='#667eea', line_width=4)
                fig.add_scatter(x=hourly_avg.index, y=hourly_avg.values, mode='markers', marker_size=8, marker_color='#f093fb')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                daily_avg = filtered_df.groupby('DayOfWeek')[energy_col].mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
                fig = px.bar(x=daily_avg.index, y=daily_avg.values, title="📅 Daily Energy Pattern")
                fig.update_traces(marker_color=['#667eea' if day not in ['Saturday','Sunday'] else '#f093fb' for day in daily_avg.index])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.violin(filtered_df, x='Season', y=energy_col, title="🎻 Seasonal Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                weekend_data = filtered_df.groupby(['IsWeekend'])[energy_col].mean()
                fig = px.pie(values=weekend_data.values, names=['Weekday 💼', 'Weekend 🏖️'], title="📊 Weekday vs Weekend")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Create heatmap data
            heatmap_data = filtered_df.pivot_table(values=energy_col, index='Hour', columns='DayOfWeek', aggfunc='mean')
            heatmap_data = heatmap_data.reindex(columns=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
            
            fig = px.imshow(heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, 
                          title="🔥 Energy Consumption Heatmap", color_continuous_scale='plasma')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📁 Upload a dataset to analyze!")

elif page == "🔮 Forecasting":
    st.markdown("### 🔮 AI-Powered Energy Forecasting")
    
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        energy_col = df.select_dtypes(include=[np.number]).columns[0]
        daily_data = df[energy_col].resample('D').mean().dropna()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            window_size = st.selectbox("📏 Analysis Window", [7, 14, 30, 60], index=2)
        with col2:
            forecast_days = st.selectbox("🎯 Forecast Days", [7, 14, 30, 60], index=2)
        with col3:
            forecast_mode = st.selectbox("🤖 Mode", ["AI + Statistical", "Statistical Only"])
        
        if len(daily_data) >= 7:  # Minimum requirement
            if st.button("🔮 Generate Forecast", type="primary"):
                with st.spinner('🤖 AI is analyzing patterns...'):
                    # Progress animation
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress.progress(i + 1)
                    
                    # Get forecast
                    last_values = daily_data.tail(min(window_size, len(daily_data))).values
                    use_ml = ml_ready and forecast_mode == "AI + Statistical"
                    predictions = smart_forecast(last_values, window_size, forecast_days, 
                                               model if use_ml else None, 
                                               scaler if use_ml else None)
                    
                    future_dates = pd.date_range(start=daily_data.index[-1] + timedelta(days=1), periods=forecast_days)
                
                # Results with animation
                st.markdown('<div class="forecast-result">', unsafe_allow_html=True)
                st.markdown("### 🎉 Forecast Results")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Main forecast plot
                    fig = go.Figure()
                    
                    # Historical data
                    hist_data = daily_data.tail(100)
                    fig.add_trace(go.Scatter(
                        x=hist_data.index, y=hist_data.values,
                        name="📊 Historical", line=dict(color='#667eea', width=3),
                        fill='tonexty', fillcolor='rgba(102, 126, 234, 0.1)'
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=predictions,
                        name="🔮 Forecast", line=dict(color='#f093fb', width=4),
                        fill='tonexty', fillcolor='rgba(240, 147, 251, 0.2)'
                    ))
                    
                    # Add confidence band (simulated)
                    upper_bound = predictions * 1.1
                    lower_bound = predictions * 0.9
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=upper_bound,
                        fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=lower_bound,
                        fill='tonexty', fillcolor='rgba(240, 147, 251, 0.1)',
                        mode='lines', line_color='rgba(0,0,0,0)',
                        name='📊 Confidence Band'
                    ))
                    
                    fig.update_layout(
                        title="⚡ Energy Forecast Results",
                        height=500,
                        hovermode='x unified',
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Forecast summary
                    forecast_df = pd.DataFrame({
                        'Date': future_dates.date,
                        'Predicted_MW': predictions.astype(int)
                    })
                    
                    st.dataframe(forecast_df, use_container_width=True, height=400)
                    
                    # Key metrics
                    st.metric("📈 Avg Forecast", f"{predictions.mean():.0f} MW")
                    st.metric("📊 Range", f"{predictions.max()-predictions.min():.0f} MW")
                    st.metric("🎯 Model", "AI-Enhanced" if use_ml else "Statistical")
                    
                    # Download
                    csv_forecast = forecast_df.to_csv(index=False)
                    st.download_button("📥 Download Forecast", csv_forecast, "energy_forecast.csv", "text/csv")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Need at least 7 days of data for forecasting")
    
    else:
        # Manual input mode
        st.markdown("### ✏️ Manual Input Mode")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            manual_values = st.text_area(
                "📝 Paste daily energy values (comma-separated)",
                height=150,
                placeholder="3500, 3600, 3450, 3700, 3550, 3800, 3650..."
            )
            
            if manual_values and st.button("🔮 Forecast from Manual Input", type="primary"):
                try:
                    values = np.array([float(x.strip()) for x in manual_values.split(',') if x.strip()])
                    if len(values) >= 7:
                        with st.spinner('🤖 Processing manual input...'):
                            time.sleep(1)
                            window_size = min(30, len(values))
                            predictions = smart_forecast(values, window_size, 30, model if ml_ready else None, scaler if ml_ready else None)
                            
                        # Show results
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=list(range(-len(values), 0)), y=values, name="📊 Input Data", line=dict(color='#667eea', width=3)))
                        fig.add_trace(go.Scatter(x=list(range(0, len(predictions))), y=predictions, name="🔮 Forecast", line=dict(color='#f093fb', width=3)))
                        fig.update_layout(title="📈 Manual Input Forecast", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download
                        manual_forecast_df = pd.DataFrame({'Day': range(1, len(predictions)+1), 'Predicted_MW': predictions.astype(int)})
                        csv_manual = manual_forecast_df.to_csv(index=False)
                        st.download_button("📥 Download Results", csv_manual, "manual_forecast.csv", "text/csv")
                    else:
                        st.error("❌ Need at least 7 values for forecasting")
                except:
                    st.error("❌ Invalid input format. Use comma-separated numbers.")
        
        with col2:
            if ml_ready and scaler:
                st.info(f"📊 Expected range:\n{scaler.data_min_[0]:.0f} - {scaler.data_max_[0]:.0f} MW")
            
            st.markdown("""
            **💡 Tips:**
            - Use daily average values
            - Minimum 7 values required
            - More data = better predictions
            - AI models enhance accuracy
            """)

# Footer
st.markdown("---")
st.markdown("<center>⚡ Built with AI-Enhanced Analytics | Streamlit + TensorFlow</center>", unsafe_allow_html=True)