import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
from streamlit_option_menu import option_menu
import joblib

# Define unique values for select boxes
flat_model_options =['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
					'Model A-Maisonette', 'Apartment', 'Maisonette', 'Terrace',
					'2-room', 'Improved-Maisonette', 'Multi Generation',
					'Premium Apartment', 'Adjoined flat', 'Premium Maisonette',
					'Model A2', 'DBSS', 'Type S1', 'Type S2', 'Premium Apartment Loft',
					'3Gen']
flat_type_options = ['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', 'MULTI GENERATION']
town_options = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT TIMAH',
                'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
                'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE',
                'QUEENSTOWN', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS',
                'YISHUN', 'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS', 'PUNGGOL']
storey_range_options = ['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15', '19 TO 21',
                        '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30', '31 TO 33', '40 TO 42',
                        '37 TO 39', '34 TO 36', '46 TO 48', '43 TO 45', '49 TO 51']

#Loading the Saved Model
model_filename = r'C:/Users/mathe\Documents/Vscode/capstone/resale_price_prediction_decision_tree.joblib'
pipeline = joblib.load(model_filename)

# Creating the StreamLit Page Configuration  
st.set_page_config(
    page_title="Singapore Flat resale prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded", )

# Creating the Option Menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home","Singapore Flat resale prediction"], 
                icons=['house','ui-checks-grid'], menu_icon="menu-button-fill", default_index=0)
    
# Creating the Home menu
if selected == "Home":
    st.title(":blue[Singapore] Flat resale prediction 🏢")
    st.subheader('',divider='blue')
    st.write("""**This Application is used for predicting resale prices for flats in Singapore using Machine learning. By leveraging historical transaction data and key features, it aims to provide accurate estimates for resale prices, aiding potential buyers and sellers in making informed decisions**""")
    st.write("""**We are using :blue[Random Forest Regression] model to predict the resale price.This Model was with the lowest Mean Absolute Error (MAE) and Mean Squared Error (MSE), and the highest R2 score of :blue[0.975539], it's the most accurate model among the various regression models tested and evaluated. Random Forest also mitigates the overfitting problem that Decision Trees might have by averaging multiple trees.**""")
    st.subheader('',divider='blue')

# Creating the Prediction page
elif selected == 'Singapore Flat resale prediction':
    st.title("Flat Resale Prediction")
    town = st.selectbox("Town", options=town_options)
    flat_type = st.selectbox("Flat Type", options=flat_type_options)
    flat_model = st.selectbox("Flat Model", options=flat_model_options)
    storey_range = st.selectbox("Storey Range", options=storey_range_options)
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=0.0, max_value=500.0, value=100.0)
    current_remaining_lease = st.number_input("Current Remaining Lease", min_value=0.0, max_value=99.0, value=20.0)
    year = 2024
    lease_commence_date = current_remaining_lease + year - 99
    years_holding = 99 - current_remaining_lease

    # Create a button to trigger the prediction
    if st.button("Predict Resale Price"):

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'town': [town],
            'flat_type': [flat_type],
            'flat_model': [flat_model],
            'storey_range': [storey_range],
            'floor_area_sqm': [floor_area_sqm],
            'current_remaining_lease': [current_remaining_lease],
            'lease_commence_date': [lease_commence_date],
            'years_holding': [years_holding],
            'remaining_lease': [current_remaining_lease],
            'year': [year]
        })

        # Make a prediction using the model
        prediction = pipeline.predict(input_data)

        # Display the prediction
        st.title(f"Predicted Resale Price: :blue{prediction}")
