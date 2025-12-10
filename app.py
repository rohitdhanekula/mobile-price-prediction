import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from price_mapper import MobilePriceMapper

# Page configuration
st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
.prediction-card {
    background-color: #e8f4fd;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 2px solid #1f77b4;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📱 Mobile Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown('Predict mobile phone price ranges using machine learning')

# Load Model
@st.cache_data
def load_model():
    try:
        with open('mobilepricemodel.pkl', 'rb') as model_file:
            model_data = pickle.load(model_file)
        return model_data
    except FileNotFoundError:
        st.error('Model file not found. Please run python model.py to train and save the model first.')
        st.stop()

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
feature_columns = model_data['feature_columns']
accuracy = model_data['accuracy']
cv_scores = model_data['cv_scores']

# Initialize price mapper
price_mapper = MobilePriceMapper()

# Price range mapping
price_range_mapping = {
    0: 'Low Cost',
    1: 'Medium Cost', 
    2: 'High Cost',
    3: 'Very High Cost'
}

price_colors = {
    0: '🟢', 1: '🔵', 2: '🟠', 3: '🔴'
}

# User Input Function
def user_input_features():
    st.sidebar.subheader('Battery Performance')
    battery_power = st.sidebar.slider('Battery Power (mAh)', 3000, 6000, 4500, help='Modern battery capacity 3000-6000 mAh')
    clock_speed = st.sidebar.slider('Clock Speed (GHz)', 1.5, 3.5, 2.5, help='Modern processor clock speed')
    n_cores = st.sidebar.slider('Number of Cores', 4, 8, 6, help='Modern processor cores 4-8 cores')
    
    st.sidebar.subheader('Display & Camera')
    pc = st.sidebar.slider('Primary Camera (MP)', 12, 108, 50, help='Modern primary camera 12-108 MP')
    fc = st.sidebar.slider('Front Camera (MP)', 8, 32, 16, help='Modern front camera 8-32 MP')
    px_height = st.sidebar.slider('Pixel Resolution Height', 720, 2160, 1440, help='Modern screen resolution height')
    px_width = st.sidebar.slider('Pixel Resolution Width', 1280, 3840, 2560, help='Modern screen resolution width')
    sc_h = st.sidebar.slider('Screen Height (cm)', 12, 18, 15, help='Modern screen height')
    sc_w = st.sidebar.slider('Screen Width (cm)', 6, 9, 7, help='Modern screen width')
    touch_screen = st.sidebar.selectbox('Touch Screen', ['No', 'Yes'], help='Touch screen capability')
    
    st.sidebar.subheader('Memory & Storage')
    ram = st.sidebar.slider('RAM (MB)', 4096, 16384, 8192, help='Modern RAM capacity 4-16 GB')
    int_memory = st.sidebar.slider('Internal Memory (GB)', 64, 1024, 256, help='Modern storage capacity 64GB-1TB')
    
    st.sidebar.subheader('Connectivity')
    blue = st.sidebar.selectbox('Bluetooth', ['No', 'Yes'], help='Bluetooth connectivity')
    wifi = st.sidebar.selectbox('WiFi', ['No', 'Yes'], help='WiFi connectivity')
    three_g = st.sidebar.selectbox('3G', ['No', 'Yes'], help='3G network support')
    four_g = st.sidebar.selectbox('4G', ['No', 'Yes'], help='4G network support')
    dual_sim = st.sidebar.selectbox('Dual SIM', ['No', 'Yes'], help='Dual SIM card support')
    
    st.sidebar.subheader('Physical Properties')
    mobile_wt = st.sidebar.slider('Mobile Weight (g)', 150, 250, 200, help='Modern phone weight 150-250g')
    m_dep = st.sidebar.slider('Mobile Depth (cm)', 0.7, 1.2, 0.9, help='Modern phone thickness')
    talk_time = st.sidebar.slider('Talk Time (hours)', 8, 24, 16, help='Modern battery talk time')
    
    # Convert categorical to numerical
    blue = 1 if blue == 'Yes' else 0
    dual_sim = 1 if dual_sim == 'Yes' else 0
    four_g = 1 if four_g == 'Yes' else 0
    three_g = 1 if three_g == 'Yes' else 0
    touch_screen = 1 if touch_screen == 'Yes' else 0
    wifi = 1 if wifi == 'Yes' else 0
    
    data = {
        'battery_power': battery_power,
        'blue': blue,
        'clock_speed': clock_speed,
        'dual_sim': dual_sim,
        'fc': fc,
        'four_g': four_g,
        'int_memory': int_memory,
        'm_dep': m_dep,
        'mobile_wt': mobile_wt,
        'n_cores': n_cores,
        'pc': pc,
        'px_height': px_height,
        'px_width': px_width,
        'ram': ram,
        'sc_h': sc_h,
        'sc_w': sc_w,
        'talk_time': talk_time,
        'three_g': three_g,
        'touch_screen': touch_screen,
        'wifi': wifi
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('Model Performance Metrics')
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric('Accuracy', f'{accuracy:.1%}', help='Model accuracy on test data')
    with metric_col2:
        st.metric('CV Score', f'{np.mean(cv_scores):.1%}', help='Cross-validation score')
    with metric_col3:
        st.metric('CV Std', f'{np.std(cv_scores):.1%}', help='Cross-validation standard deviation')
    with metric_col4:
        st.metric('Features', len(feature_columns), help='Number of input features')

with col2:
    st.subheader('Your Mobile Specs')
    st.dataframe(input_df.T, use_container_width=True)

st.markdown('---')

st.subheader('Price Prediction')
if st.button('Predict Price Range', type='primary', use_container_width=True):
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_columns)
    
    prediction = model.predict(input_scaled_df)
    prediction_proba = model.predict_proba(input_scaled_df)
    predicted_price_range = price_range_mapping[prediction[0]]
    confidence = prediction_proba[0][prediction[0]] * 100
    
    features_dict = input_df.iloc[0].to_dict()
    price_info = price_mapper.get_price_estimate(features_dict, prediction[0])
    
    st.markdown(f"""
    <div class="prediction-card">
    <h2>{price_colors[prediction[0]]} {predicted_price_range}</h2>
    <p>Confidence: {confidence:.1f}%</p>
    <hr>
    <h3>Real Price Prediction</h3>
    <p><strong>Price Range:</strong> {price_info['price_range']}</p>
    <p><strong>Estimated Price:</strong> {price_info['estimated_price']}</p>
    <p><strong>Category:</strong> {price_info['category']} Phone</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader('Prediction Probabilities')
    prob_df = pd.DataFrame({
        'Price Range': [price_range_mapping[i] for i in range(4)],
        'Probability': prediction_proba[0] * 100
    })
    fig = px.bar(prob_df, x='Price Range', y='Probability', title='Prediction Probability Distribution', color='Probability', color_continuous_scale='Blues')
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
