import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="ML Model Deployment with Streamlit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        padding: 1rem;
        background-color: #e8f5e8;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Synthetic data generation function
@st.cache_data
def generate_house_data(n_samples=1000):
    """Generate synthetic house price data for training"""
    np.random.seed(42)
    
    # Generate features
    size = np.random.normal(1800, 600, n_samples)
    size = np.clip(size, 500, 5000)  # Realistic size limits
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.25, 0.05])
    bathrooms = np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.4, 0.3, 0.1])
    age = np.random.randint(0, 50, n_samples)
    
    # Generate price with realistic relationships
    price = (size * 120 + 
             bedrooms * 15000 + 
             bathrooms * 10000 - 
             age * 500 + 
             np.random.normal(0, 20000, n_samples))
    
    price = np.clip(price, 50000, 1000000)  # Realistic price limits
    
    return pd.DataFrame({
        'size_sqft': size,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age,
        'price': price
    })

# Model training function
@st.cache_data
def train_models():
    """Train multiple ML models and return the best one"""
    df = generate_house_data()
    
    # Feature preparation
    X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years']]
    y = df['price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train different models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    model_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        model_results[name] = {
            'model': model,
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    # Select best model based on R² score
    best_model_name = max(model_results.keys(), 
                         key=lambda x: model_results[x]['r2_score'])
    
    return model_results, best_model_name, df

# Load or train models
@st.cache_resource
def get_models():
    """Load or train models with caching"""
    return train_models()

# Main application function
def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Advanced House Price Predictor</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar for model selection and information
    st.sidebar.title("Model Information")
    
    # Load models
    model_results, best_model_name, df = get_models()
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model:",
        options=list(model_results.keys()),
        index=list(model_results.keys()).index(best_model_name)
    )
    
    # Display model metrics
    st.sidebar.markdown("### Model Performance")
    st.sidebar.metric(
        "R² Score", 
        f"{model_results[selected_model]['r2_score']:.4f}"
    )
    st.sidebar.metric(
        "RMSE", 
        f"${model_results[selected_model]['rmse']:,.0f}"
    )
    
    # Best model indicator
    if selected_model == best_model_name:
        st.sidebar.success("✅ Best performing model")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🏡 House Details")
        
        # Input fields
        size = st.number_input(
            'House Size (sq ft)',
            min_value=500,
            max_value=5000,
            value=1800,
            step=50,
            help="Enter the house size in square feet"
        )
        
        bedrooms = st.selectbox(
            'Number of Bedrooms',
            options=[1, 2, 3, 4, 5],
            index=2,
            help="Select the number of bedrooms"
        )
        
        bathrooms = st.selectbox(
            'Number of Bathrooms',
            options=[1, 2, 3, 4],
            index=1,
            help="Select the number of bathrooms"
        )
        
        age = st.slider(
            'House Age (years)',
            min_value=0,
            max_value=50,
            value=10,
            help="Select the age of the house"
        )
        
        # Prediction button
        if st.button('🔮 Predict Price', type="primary"):
            model = model_results[selected_model]['model']
            
            # Make prediction
            features = np.array([[size, bedrooms, bathrooms, age]])
            prediction = model.predict(features)[0]
            
            # Display result
            st.markdown(
                f'<div class="prediction-result">Estimated Price: ${prediction:,.2f}</div>',
                unsafe_allow_html=True
            )
            
            # Confidence interval (approximation)
            rmse = model_results[selected_model]['rmse']
            confidence_lower = prediction - rmse
            confidence_upper = prediction + rmse
            
            st.info(f"**Prediction Range:** ${confidence_lower:,.0f} - ${confidence_upper:,.0f}")
            
            # Store prediction in session state for visualization
            st.session_state.prediction = {
                'size': size,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'age': age,
                'price': prediction
            }
    
    with col2:
        st.markdown("### 📊 Data Analysis")
        
        # Feature importance for Random Forest
        if selected_model == 'Random Forest':
            model = model_results[selected_model]['model']
            feature_names = ['Size (sq ft)', 'Bedrooms', 'Bathrooms', 'Age (years)']
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                title='Feature Importance',
                orientation='h',
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig_importance.update_layout(height=300)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Dataset statistics
        st.markdown("### 📈 Dataset Statistics")
        stats_df = df.describe().round(2)
        st.dataframe(stats_df, use_container_width=True)
    
    # Visualization section
    st.markdown("### 🎯 Model Visualization")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Price vs Size", "Model Comparison", "Prediction Context"])
    
    with tab1:
        # Scatter plot of size vs price
        fig_scatter = px.scatter(
            df, 
            x='size_sqft', 
            y='price',
            title='House Size vs Price Relationship',
            labels={'size_sqft': 'Size (sq ft)', 'price': 'Price ($)'},
            opacity=0.6
        )
        
        # Add prediction point if available
        if 'prediction' in st.session_state:
            pred = st.session_state.prediction
            fig_scatter.add_scatter(
                x=[pred['size']], 
                y=[pred['price']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Your Prediction',
                showlegend=True
            )
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        # Model comparison
        comparison_data = []
        for name, result in model_results.items():
            comparison_data.append({
                'Model': name,
                'R² Score': result['r2_score'],
                'RMSE': result['rmse']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_r2 = px.bar(
                comparison_df, 
                x='Model', 
                y='R² Score',
                title='Model R² Score Comparison',
                color='R² Score',
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            fig_rmse = px.bar(
                comparison_df, 
                x='Model', 
                y='RMSE',
                title='Model RMSE Comparison',
                color='RMSE',
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
    
    with tab3:
        # Distribution plots
        if 'prediction' in st.session_state:
            pred = st.session_state.prediction
            
            # Price distribution with prediction
            fig_dist = px.histogram(
                df, 
                x='price',
                nbins=50,
                title='Price Distribution with Your Prediction',
                labels={'price': 'Price ($)'}
            )
            
            # Add vertical line for prediction
            fig_dist.add_vline(
                x=pred['price'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Your Prediction: ${pred['price']:,.0f}"
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Make a prediction to see how it compares to the dataset!")
    
    # Additional information
    st.markdown("### ℹ️ About This Application")
    
    with st.expander("How to use this app"):
        st.write("""
        1. **Enter house details** in the left column
        2. **Select a model** from the sidebar
        3. **Click 'Predict Price'** to get an estimate
        4. **Explore visualizations** to understand the model's behavior
        5. **Compare models** using the performance metrics
        """)
    
    with st.expander("Technical Details"):
        st.write(f"""
        - **Dataset Size:** {len(df):,} synthetic house records
        - **Features Used:** Size, Bedrooms, Bathrooms, Age
        - **Best Model:** {best_model_name}
        - **Best R² Score:** {model_results[best_model_name]['r2_score']:.4f}
        - **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🚀 **Built with Streamlit** | 🤖 **Machine Learning Model Deployment Demo**"
    )

if __name__ == '__main__':
    main()
