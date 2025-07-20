import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title
st.title("🏢 Employee Salary Prediction Dashboard")

# Load dataset
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

uploaded_file = st.file_uploader("Upload Employee_Salary.csv file", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.subheader("📊 Original Dataset")
    st.dataframe(df)
    st.text(df.info())

    # Data Insights
    st.subheader("🔍 Missing Values")
    st.write(df.isna().sum())

    st.subheader("📌 Job Title Distribution")
    st.bar_chart(df['Job_Title'].value_counts())

    st.subheader("🏢 Department Distribution")
    st.bar_chart(df['Department'].value_counts())

    st.subheader("🎓 Education Level Distribution")
    st.bar_chart(df['Education_Level'].value_counts())

    # Drop unnecessary columns
    df.drop(['Hire_Date', 'Employee_ID'], inplace=True, axis=1)

    # Boxplots
    st.subheader("📦 Boxplot - Monthly Salary")
    fig, ax = plt.subplots()
    sns.boxplot(x=df['Monthly_Salary'], ax=ax)
    st.pyplot(fig)

    st.subheader("📦 Boxplot - Age")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df['Age'], ax=ax2)
    st.pyplot(fig2)

    # Encode categorical columns
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Education_Level'] = le.fit_transform(df['Education_Level'])

    dummies_job = pd.get_dummies(df['Job_Title'], prefix='JobTitle').astype(int)
    dummies_dept = pd.get_dummies(df['Department'], prefix='Department').astype(int)

    df.drop(['Job_Title', 'Department'], axis=1, inplace=True)
    df_encoded = pd.concat([df, dummies_job, dummies_dept], axis=1)

    st.subheader("🔠 Encoded Dataset")
    st.dataframe(df_encoded)

    # Scaling
    scaler = MinMaxScaler()
    scaled_cols = ['Age', 'Work_Hours_Per_Week', 'Years_At_Company', 'Education_Level',
                   'Performance_Score', 'Projects_Handled', 'Overtime_Hours', 'Sick_Days',
                   'Remote_Work_Frequency', 'Team_Size', 'Training_Hours', 'Promotions',
                   'Employee_Satisfaction_Score', 'Resigned']

    df_scaled = df_encoded.copy()
    df_scaled[scaled_cols] = scaler.fit_transform(df_scaled[scaled_cols])

    st.subheader("📐 Scaled Dataset")
    st.dataframe(df_scaled)

    # Train Model
    X = df_scaled.drop(columns=['Monthly_Salary'])
    y = df_scaled['Monthly_Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # Evaluation
    st.subheader("📈 Random Forest Evaluation Metrics")

    def evaluate_model(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2

    mae, mse, rmse, r2 = evaluate_model(y_test, y_pred)

    st.write(f"*Mean Absolute Error (MAE):* {mae:.4f}")
    st.write(f"*Mean Squared Error (MSE):* {mse:.4f}")
    st.write(f"*Root Mean Squared Error (RMSE):* {rmse:.4f}")
    st.write(f"*R2 Score:* {r2:.4f}")

    # Predictions table
    st.subheader("🧾 Sample Predictions")
    predictions_df = pd.DataFrame({'Actual': y_test[:10].values, 'Predicted': y_pred[:10]})
    st.dataframe(predictions_df)

else:
    st.warning("📁 Please upload the Employee_Salary.csv file to begin.")
