import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r"C:\Users\mohap\vscodeproject\House price prediction\House_Price_Prediction.pkl", 'rb'))

# Set the title of the Streamlit app
st.title("House Price Prediction App")

# Add a brief description
st.write("This app predicts the house range based on square footage using a simple linear regression model.")

# Add input widget for user to enter years of experience
Sqft_living = st.number_input("Enter Sq Footage:", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0)

# When the button is clicked, make predictions
if st.button("Houe Price"):
    # Make a prediction using the trained model
    Sqft_input = np.array([[Sqft_living]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(Sqft_input)
   
    # Display the result
    st.success(f"The predicted price of house for {Sqft_living} Square feet is: ${prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model was trained using a dataset of prices of house and square footage model by prakash senapati")