import streamlit as st
import sklearn
import pandas as pd

st.title('Tips Prediction Interface')
df = pd.read_csv("tip (1).csv")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['day'] = le.fit_transform(df['day'])
df['time'] = le.fit_transform(df['time'])
Mango = df.drop(columns = 'tip') 
Tomato = df['tip']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Mango,Tomato,test_size=0.15,random_state=5)
from sklearn.tree import DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor()
tree_regressor.fit(X_train,y_train)
st.header('Enter Restaurant Bill Details:')
total_bill = st.number_input('Total Bill Amount:', min_value=0.0, format="%.2f")
size = st.number_input('Party Size:', min_value=1, step=1)
sex = st.selectbox('Sex of Payer:', ['Male', 'Female'])
smoker = st.selectbox('Smoker?', ['Yes', 'No'])
day = st.selectbox('Day of the Week:', ['Thur', 'Fri', 'Sat', 'Sun'])
time = st.selectbox('Time of Day:', ['Lunch', 'Dinner'])
sex_encoded = 1 if sex == 'Male' else 0  # Assuming 1=Male, 0=Female
smoker_encoded = 1 if smoker == 'Yes' else 0  # Assuming 1=Yes, 0=No

day_mapping = {'Thur': 0, 'Fri': 1, 'Sat': 2, 'Sun': 3} # Map string to numerical values
day_encoded = day_mapping[day]

time_mapping = {'Lunch': 0, 'Dinner': 1} # Map string to numerical values
time_encoded = time_mapping[time]
# Prediction
input_data = pd.DataFrame({
    'total_bill': [total_bill],
    'size': [size],
    'sex': [sex_encoded],
    'smoker': [smoker_encoded],
    'day': [day_encoded],
    'time': [time_encoded]
})

if st.button('Predict Tip'):
    predicted_tip = tree_regressor.predict(input_data)
    st.success(f'Predicted Tip Amount: ${predicted_tip[0]:.2f}') # Format to two decimal places

    
    









 