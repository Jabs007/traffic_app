import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime
from streamlit_lottie import st_lottie
import sqlite3
import requests
import shap
import streamlit_authenticator as stauth
from passlib.hash import sha256_crypt

# Set Streamlit page configuration at the very top
st.set_page_config(
    page_title="Smart Traffic Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- User Authentication ---
def authenticate():
    # Store hashed password only once to avoid re-hashing on every rerun
    if "hashed_password" not in st.session_state:
        st.session_state["hashed_password"] = sha256_crypt.hash("admin123")
    credentials = {
        "usernames": {
            "admin": {
                "name": "Admin",
                "password": st.session_state["hashed_password"]
            }
        }
    }
    authenticator = stauth.Authenticate(
        credentials,
        "traffic_app",
        "abcdef",
        cookie_expiry_days=1
    )
    name, authentication_status, _ = authenticator.login("Login", location="main")  # Added location parameter
    if authentication_status is False:
        st.error("Username/password is incorrect")
        st.stop()
    elif authentication_status is None:
        st.warning("Please enter your username and password")
        st.stop()
    st.session_state["user"] = name
    return name

# --- Theme Toggle ---
def theme_toggle():
    st.sidebar.markdown("### 🌓 Theme")
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "light" else 1)
    st.session_state.theme = theme.lower()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {"#0e1117" if st.session_state.theme == "dark" else "#ffffff"};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Main App Class and UI Logic ---
class TrafficPredictorApp:
    def __init__(self):
        self.model = None
        self.label_encoders = None
        self.feature_columns = None
        self.df = None
        self.load_model_and_data()
        self.create_database()

    def load_model_and_data(self):
        try:
            self.model = joblib.load('traffic_model.pkl')
            self.label_encoders = joblib.load('label_encoders.pkl')
            self.feature_columns = joblib.load('feature_columns.pkl')
            self.df = pd.read_csv('mobility_with_new_features.csv')
        except Exception as e:
            st.error(f"Data loading error: {e}")
            self.model = None
            self.label_encoders = None
            self.feature_columns = None
            self.df = pd.DataFrame()

    def create_database(self):
        try:
            conn = sqlite3.connect('prediction_history.db')
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT,
                            Vehicle_Count INTEGER,
                            Traffic_Speed_kmh REAL,
                            "Road_Occupancy_%" REAL,
                            Traffic_Light_State TEXT,
                            Weather_Condition TEXT,
                            Accident_Report INTEGER,
                            Hour INTEGER,
                            DayOfWeek TEXT,
                            Prediction TEXT
                        )''')
            conn.commit()
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
        finally:
            conn.close()

    def save_to_db(self, input_data, prediction_label):
        try:
            traffic_light_str = self.label_encoders['Traffic_Light_State'].inverse_transform([input_data['Traffic_Light_State']])[0]
            weather_str = self.label_encoders['Weather_Condition'].inverse_transform([input_data['Weather_Condition']])[0]
            day_str = self.label_encoders['DayOfWeek'].inverse_transform([input_data['DayOfWeek']])[0]
            conn = sqlite3.connect('prediction_history.db')
            c = conn.cursor()
            c.execute('''INSERT INTO history (timestamp, Vehicle_Count, Traffic_Speed_kmh, "Road_Occupancy_%", 
                                              Traffic_Light_State, Weather_Condition, Accident_Report,
                                              Hour, DayOfWeek, Prediction)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       input_data['Vehicle_Count'],
                       input_data['Traffic_Speed_kmh'],
                       input_data['Road_Occupancy_%'],
                       traffic_light_str,
                       weather_str,
                       input_data['Accident_Report'],
                       input_data['Hour'],
                       day_str,
                       prediction_label))
            conn.commit()
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
        except KeyError as e:
            st.error(f"KeyError: Missing key {e} in input_data. Please check the input fields.")
        finally:
            conn.close()

    def load_history_from_db(self, limit=10):
        try:
            conn = sqlite3.connect('prediction_history.db')
            df = pd.read_sql_query(f"SELECT * FROM history ORDER BY id DESC LIMIT {limit}", conn)
            conn.close()
            return df
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            return pd.DataFrame()

    def load_lottie_url(self, url: str):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading animation: {e}")
            return None

    def display_home_page(self):
        st.title("🚦 Smart Traffic Prediction System")
        lottie_animation = self.load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_Q7WY7CfUco.json")
        if lottie_animation:
            st_lottie(lottie_animation, speed=1, loop=True, quality="high", height=300)
        st.markdown("""
            Welcome to the interactive dashboard for traffic prediction using smart mobility data.  
            Navigate using the sidebar to explore insights or predict traffic conditions.
        """)
        st.markdown("### 🕒 Recent Predictions")
        history_df = self.load_history_from_db(limit=5)
        if not history_df.empty:
            st.dataframe(history_df.drop(columns=["id"]), use_container_width=True)
        else:
            st.info("No recent predictions found.")

    def display_traffic_prediction(self):
        st.title("🧠 Predict Traffic Condition")
        with st.form("prediction_form"):
            vehicle_count = st.number_input("Vehicle Count", min_value=0, help="Number of vehicles detected")
            traffic_speed = st.slider("Traffic Speed (km/h)", 0, 150, 50, help="Average speed of vehicles")
            road_occupancy = st.slider("Road Occupancy (%)", 0, 100, 30, help="Percentage of road occupied")
            traffic_light = st.selectbox("Traffic Light State", self.label_encoders['Traffic_Light_State'].classes_, help="Current state of the traffic light")
            weather = st.selectbox("Weather Condition", self.label_encoders['Weather_Condition'].classes_, help="Current weather condition")
            accident = st.radio("Accident Reported?", ['No', 'Yes'], help="Is there an accident reported?")
            hour = st.slider("Hour of Day", 0, 23, help="Hour of the day (0-23)")
            day_of_week = st.selectbox("Day of Week", self.label_encoders['DayOfWeek'].classes_, help="Day of the week")
            submit = st.form_submit_button("Predict")

        if submit:
            errors = self.validate_inputs(vehicle_count, traffic_speed, road_occupancy)
            if errors:
                for err in errors:
                    st.error(err)
            else:
                with st.spinner("Predicting..."):
                    self.make_prediction(vehicle_count, traffic_speed, road_occupancy, traffic_light,
                                        weather, accident, hour, day_of_week)

        # Clear history button
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            if st.button("🗑️ Clear Prediction History"):
                st.session_state.prediction_history = []
                st.success("Prediction history cleared.")

    def validate_inputs(self, vehicle_count, traffic_speed, road_occupancy):
        errors = []
        if vehicle_count <= 0:
            errors.append("🚫 Vehicle Count must be greater than 0.")
        if traffic_speed <= 0:
            errors.append("🚫 Traffic Speed must be greater than 0 km/h.")
        if road_occupancy <= 0:
            errors.append("🚫 Road Occupancy must be greater than 0%.")
        return errors

    def make_prediction(self, vehicle_count, traffic_speed, road_occupancy, traffic_light, weather,
                        accident, hour, day_of_week):
        input_dict = {
            'Vehicle_Count': vehicle_count,
            'Traffic_Speed_kmh': traffic_speed,
            'Road_Occupancy_%': road_occupancy,
            'Traffic_Light_State': self.label_encoders['Traffic_Light_State'].transform([traffic_light])[0],
            'Weather_Condition': self.label_encoders['Weather_Condition'].transform([weather])[0],
            'Accident_Report': 1 if accident == 'Yes' else 0,
            'Hour': hour,
            'DayOfWeek': self.label_encoders['DayOfWeek'].transform([day_of_week])[0],
        }

        try:
            input_df = pd.DataFrame([input_dict])[self.feature_columns]
            prediction = self.model.predict(input_df)[0]
            prediction_label = self.label_encoders['Traffic_Condition'].inverse_transform([prediction])[0]
            st.success(f"🚦 Predicted Traffic Condition: **{prediction_label}**")
            self.save_to_db(input_dict, prediction_label)

            # SHAP Explainability
            st.markdown("#### 🔍 Model Explanation (SHAP)")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(input_df)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.initjs()
            st.pyplot(shap.force_plot(explainer.expected_value[0], shap_values[0][0], input_df.iloc[0], matplotlib=True, show=False))

        except KeyError:
            st.error("Prediction Error: Missing expected data for prediction. Check your inputs.")
            return
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            return

        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []

        st.session_state.prediction_history.append({
            'Vehicle_Count': vehicle_count,
            'Traffic_Speed_kmh': traffic_speed,
            'Road_Occupancy_%': road_occupancy,
            'Traffic_Light_State': traffic_light,
            'Weather_Condition': weather,
            'Accident_Report': accident,
            'Hour': hour,
            'DayOfWeek': day_of_week,
            'Predicted_Condition': prediction_label
        })

        st.markdown("### 🕘 Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df[::-1], use_container_width=True)

        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name='traffic_prediction_history.csv',
            mime='text/csv'
        )

    def display_eda_dashboard(self):
        st.title("📊 Exploratory Traffic Analysis Dashboard")

        with st.sidebar:
            st.subheader("📋 Filter Data")
            if not self.df.empty and 'Date' in self.df.columns:
                date_range = st.date_input("Date Range", [pd.to_datetime(self.df['Date']).min(), pd.to_datetime(self.df['Date']).max()])
            else:
                date_range = None
            hour_range = st.slider("Hour Range", 0, 23, (0, 23))
            weather_filter = st.multiselect("Weather Condition", self.df['Weather_Condition'].unique() if not self.df.empty else [], self.df['Weather_Condition'].unique() if not self.df.empty else [])
            traffic_filter = st.multiselect("Traffic Condition", self.df['Traffic_Condition'].unique() if not self.df.empty else [], self.df['Traffic_Condition'].unique() if not self.df.empty else [])

        # Apply filters
        if not self.df.empty and date_range:
            filtered_df = self.df[
                (pd.to_datetime(self.df['Date']).between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))) &
                (self.df['Hour'].between(hour_range[0], hour_range[1])) &
                (self.df['Weather_Condition'].isin(weather_filter)) &
                (self.df['Traffic_Condition'].isin(traffic_filter))
            ]
        else:
            filtered_df = pd.DataFrame()

        if not filtered_df.empty:
            st.markdown("### 🚘 Key Metrics")
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Avg Speed (km/h)", f"{filtered_df['Traffic_Speed_kmh'].mean():.2f}")
            kpi2.metric("Avg Occupancy (%)", f"{filtered_df['Road_Occupancy_%'].mean():.2f}")
            kpi3.metric("Total Vehicles", f"{filtered_df['Vehicle_Count'].sum()}")

            st.markdown("### ⏰ Busiest Hours")
            hour_df = filtered_df.groupby('Hour').size().reset_index(name='Count')
            fig_hour = px.line(hour_df, x='Hour', y='Count', markers=True,
                               title="Traffic Volume by Hour",
                               labels={"Count": "Traffic Count", "Hour": "Hour of the Day"},
                               template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                               line_shape='spline')
            fig_hour.update_traces(line=dict(color='royalblue'))
            fig_hour.update_layout(hovermode="x unified")
            st.plotly_chart(fig_hour, use_container_width=True)

            st.markdown("### 📅 Busiest Day of Week")
            day_df = filtered_df.groupby('DayOfWeek').size().reset_index(name='Count')
            fig_day = px.bar(day_df, x='DayOfWeek', y='Count', color='DayOfWeek',
                             title="Traffic Volume by Day",
                             color_continuous_scale="Viridis",
                             labels={"DayOfWeek": "Day of the Week", "Count": "Traffic Count"},
                             template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white")
            fig_day.update_layout(hovermode="x unified", showlegend=False)
            st.plotly_chart(fig_day, use_container_width=True)

            st.markdown("### 🚨 Accident Reports")
            acc_df = filtered_df['Accident_Report'].value_counts().reset_index()
            acc_df.columns = ['Accident_Reported', 'Count']
            acc_df['Accident_Reported'] = acc_df['Accident_Reported'].map({0: 'No', 1: 'Yes'})
            fig_acc = px.pie(acc_df, names='Accident_Reported', values='Count',
                             title="Accident Distribution",
                             color='Accident_Reported',
                             color_discrete_map={'Yes': 'red', 'No': 'green'})
            fig_acc.update_traces(textinfo='percent+label', pull=[0.1, 0])
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.error("No data available for the selected filters.")

        st.markdown("### 🚦 Average Speed by Traffic Condition")
        if not filtered_df.empty:
            avg_speed_df = filtered_df.groupby("Traffic_Condition")["Traffic_Speed_kmh"].mean().reset_index()
            fig_avg_speed = px.bar(avg_speed_df, x="Traffic_Condition", y="Traffic_Speed_kmh",
                                color="Traffic_Condition",
                                title="Average Speed by Traffic Condition",
                                labels={"Traffic_Speed_kmh": "Avg Speed (km/h)"},
                                template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                                text_auto='.2f')
            fig_avg_speed.update_layout(showlegend=False)
            st.plotly_chart(fig_avg_speed, use_container_width=True)

    def display_about_page(self):
        st.title("🔍 About This App")
        st.markdown("""
            **Smart Traffic Prediction System** is an interactive platform designed to predict traffic conditions
            based on real-time data collected from smart mobility sources. By using advanced machine learning 
            models, this system can help manage traffic flow and predict congestion based on several factors, 
            such as traffic light state, road occupancy, and weather conditions.

            ### Key Features:
            - Traffic Condition Prediction using machine learning
            - Real-time data collection and logging
            - Interactive EDA Dashboard with key metrics and visualizations
            - Historical data and traffic prediction logging
            - Model explainability with SHAP

            ### Developed By:
            - Analytics Nexus
            - Traffic Data Science Team
        """)
        st.image("https://via.placeholder.com/400x200.png?text=Traffic+Prediction+App", caption="Traffic Prediction")
        st.markdown("[📄 Documentation](https://github.com/your-repo/traffic-prediction-docs)")

def sidebar_logo():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/854/854878.png", width=80)
    st.sidebar.markdown("<h3 style='text-align: center;'>Smart Traffic</h3>", unsafe_allow_html=True)

def footer():
    st.markdown("""
        <hr>
        <div style='text-align: center; color: gray; font-size: 0.9em;'>
            &copy; 2024 Analytics Nexus | Powered by Streamlit 🚦
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    sidebar_logo()
    theme_toggle()
    authenticate()
    app = TrafficPredictorApp()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Home",
            "🧠 Traffic Prediction",
            "📊 EDA Dashboard",
            "ℹ️ About"
        ],
        format_func=lambda x: x.split(" ", 1)[-1]
    )

    if "Home" in page:
        app.display_home_page()
    elif "Traffic Prediction" in page:
        app.display_traffic_prediction()
    elif "EDA Dashboard" in page:
        app.display_eda_dashboard()
    elif "About" in page:
        app.display_about_page()

    footer()
st.write("This is the end of the app")