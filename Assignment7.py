import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Function to generate synthetic house data
@st.cache_data
def generate_house_data(n_samples=100):
    np.random.seed(42)
    size = np.random.normal(1500, 500, n_samples)
    price = size * 100 + np.random.normal(0, 10000, n_samples)
    return pd.DataFrame({'size_sqft': size, 'price': price})

# Function to train the linear regression model
@st.cache_resource
def train_model():
    df = generate_house_data()
    X = df[['size_sqft']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Streamlit app UI and logic
def main():
    st.title('🏠 Simple House Price Predictor')
    st.write('Enter the house size (in square feet) to predict its sale price.')

    # Train the model (cached)
    model = train_model()

    # User input
    size = st.number_input(
        'House size (square feet)', min_value=500, max_value=5000, value=1500, step=10
    )

    if st.button('Predict price'):
        prediction = model.predict(np.array([[size]]))
        st.success(f'Estimated price: ${prediction[0]:,.2f}')

        # Visualization
        df = generate_house_data()
        fig = px.scatter(df, x='size_sqft', y='price', title='Size vs Price Relationship')
        fig.add_scatter(
            x=[size], y=[prediction[0]],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Prediction'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
