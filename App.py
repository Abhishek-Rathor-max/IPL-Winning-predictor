import streamlit as st
import pickle
import pandas as pd

# Updated teams and cities lists (kept from your snippet)
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load the model
pipe = pickle.load(open('pipe_IPL.pkl', 'rb'))

st.title('IPL Win Predictor')

# FIX 1: Changed st.beta_columns to st.columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', step=1)

# FIX 2: Changed st.beta_columns to st.columns
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', step=1)
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets_out = st.number_input('Wickets out', step=1, max_value=10)

if st.button('Predict Probability'):
    # Logic Calculations
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets_out

    # FIX 3: Prevent division by zero if overs or balls_left is 0
    if overs > 0:
        crr = score / overs
    else:
        crr = 0

    if balls_left > 0:
        rrr = (runs_left * 6) / balls_left
    else:
        rrr = 0

    # Creating the Input DataFrame
    # Note: Column names MUST match exactly what you used during model training
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Prediction
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Displaying Results
    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")
