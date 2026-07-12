import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="AI Health-Risk & Lifestyle Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI Health-Risk & Lifestyle Predictor")
st.markdown("---")

# ==========================================
# 2. DATA GENERATION & ML MODEL TRAINING
# ==========================================
# Creating a realistic synthetic dataset for training the ML model
@st.cache_resource
def train_lifestyle_model():
    np.random.seed(42)
    num_samples = 1000
    
    # Generate realistic ranges for 1000 people
    sleep = np.random.uniform(4, 10, num_samples)          # 4 to 10 hours
    screen_time = np.random.uniform(1, 12, num_samples)    # 1 to 12 hours
    water = np.random.uniform(1, 5, num_samples)            # 1 to 5 liters
    exercise = np.random.uniform(0, 90, num_samples)        # 0 to 90 minutes
    junk_food = np.random.randint(0, 8, num_samples)        # 0 to 7 meals/week
    
    # Mathematical logic: Good habits increase score, bad habits decrease it
    # Max theoretical score approx 100
    score = (sleep * 5) - (screen_time * 3) + (water * 6) + (exercise * 0.4) - (junk_food * 4) + 50
    
    # Normalize score to be strictly between 0 and 100
    score = np.clip(score, 0, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Sleep_Hours': sleep,
        'Screen_Time': screen_time,
        'Water_Intake': water,
        'Exercise_Min': exercise,
        'Junk_Food_Freq': junk_food,
        'Health_Score': score
    })
    
    # Train a Random Forest Regressor
    X = df[['Sleep_Hours', 'Screen_Time', 'Water_Intake', 'Exercise_Min', 'Junk_Food_Freq']]
    y = df['Health_Score']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, df

model, dataset = train_lifestyle_model()

# ==========================================
# 3. USER INTERFACE (SIDEBAR INPUTS)
# ==========================================
st.sidebar.header("📊 Enter Daily Habits")
st.sidebar.markdown("Adjust the metrics below to reflect a typical day:")

user_sleep = st.sidebar.slider("Sleep Hours", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
user_screen = st.sidebar.slider("Screen Time (Hours)", min_value=0.0, max_value=16.0, value=5.0, step=0.5)
user_water = st.sidebar.slider("Water Intake (Liters)", min_value=0.5, max_value=6.0, value=2.5, step=0.1)
user_exercise = st.sidebar.slider("Physical Exercise (Minutes)", min_value=0, max_value=120, value=30, step=5)
user_junk = st.sidebar.slider("Junk Food Frequency (Meals/Week)", min_value=0, max_value=14, value=2, step=1)

# ==========================================
# 4. PREDICTION LOGIC & VISUALIZATION
# ==========================================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🔮 Your AI Health Assessment")
    
    # Predict based on user input
    user_features = np.array([[user_sleep, user_screen, user_water, user_exercise, user_junk]])
    predicted_score = model.predict(user_features)[0]
    
    # Display the score prominently
    st.metric(label="Predicted Lifestyle Health Score", value=f"{round(predicted_score, 1)} / 100")
    
    # Risk categorization and feedback
    if predicted_score >= 75:
        st.success("✅ **Category: Healthy & Low Risk**")
        st.write("Fantastic job! Your habits are highly sustainable. Keep maintaining this balance to prevent long-term metabolic health issues.")
    elif 45 <= predicted_score < 75:
        st.warning("⚠️ **Category: Moderate Risk / Sub-optimal Lifestyle**")
        st.write("Your habits are decent, but there is noticeable room for optimization. Small tweaks to your screen time, sleep sync, or daily hydration could greatly boost your energy levels.")
    else:
        st.error("🚨 **Category: High Lifestyle Risk**")
        st.write("Your metrics suggest high vulnerability to fatigue, strain, and poor recovery. Prioritize increasing your sleep and cutting back on screen time immediately.")

    # Actionable tailored insights
    st.markdown("### 💡 AI Recommendations")
    if user_sleep < 7:
        st.write("- **Sleep Alert:** Try getting at least 7-8 hours of sleep to improve mental recovery.")
    if user_screen > 6:
        st.write("- **Digital Fatigue:** Your screen time is elevated. Implement the 20-20-20 rule to reduce digital eye strain.")
    if user_water < 2.5:
        st.write("- **Hydration Deficit:** Increase water intake closer to 3 liters to boost metabolic performance.")
    if user_junk > 3:
        st.write("- **Dietary Impact:** High processed food frequency drains energy stability. Try replacing a few meals with whole whole foods.")

with col2:
    st.subheader("Your Dynamic Habit Breakdown")
    
    # Create a dictionary of your current slider variables
    # Replace these names with the actual slider variables you defined on the left!
    user_habits = {
        "sleep": user_sleep,
        "screen time": user_screen,
        "water intake": user_water,
        "exercise min": user_exercise
    }
    
    # Display it as a bar chart
    st.bar_chart(user_habits)