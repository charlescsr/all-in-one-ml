import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle

def main():
    st.title("Streamlit Tutorial")

    # Sidebar in streamlit with the following choices: Preprocessing, Pickling.
    choice = st.sidebar.selectbox("Select a page", ["Preprocessing", "Pickling"])

    if choice == "Preprocessing":
        st.subheader("Preprocessing Page")
        st.write("This is the page to preprocess the data.")

        # File input
        file = st.file_uploader("Upload a file", type=["csv"])
        if file is not None:
            df = pd.read_csv(file)
            st.dataframe(df)

            # Column list
            col_list = st.multiselect("Select columns to clean", df.columns)

            # Preprocessing options
            preprocess_choice = st.selectbox("Select preprocessing", ["", "MinMaxScaler", "LabelEncoder", "OneHotEncoder"])

            if preprocess_choice == "MinMaxScaler":
                scaler = MinMaxScaler()
                scaler.fit(df[col_list])
                df[col_list] = scaler.transform(df[col_list])
                st.dataframe(df)

            elif preprocess_choice == "LabelEncoder":
                le = LabelEncoder()
                df[col_list] = le.fit_transform(df[col_list])
                st.dataframe(df)

            elif preprocess_choice == "OneHotEncoder":
                ohe = OneHotEncoder()
                df[col_list] = ohe.fit_transform(df[col_list])
                st.dataframe(df)

            # Download button to get preprocessed data
            st.download_button("Download preprocessed data", data=df.to_csv(index=False), file_name="preprocessed.csv")

    elif choice == "Pickling":
        st.subheader("Pickling Page")
        st.write("This is the page to pickle the model.")

        file = st.file_uploader("Upload a file", type=["csv"])

        if file is not None:
            model = st.selectbox("Select model", ["", "Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Logistic Regression", "SVM", "KNN", "Decision Tree Classifier", "Random Forest Classifier"])

            if model == "Linear Regression":
                st.write("Linear Regression")
                st.write("This is the page to pickle the model.")

                df = pd.read_csv(file)
                st.dataframe(df)

                # Target column
                target_column = st.selectbox("Select target column", df.columns)

                # Model
                X = df.drop(target_column, axis=1)
                y = df[target_column]

                # Train test split slider
                train_test_split_slider = st.slider("Select train test split", 0.1, 0.9, 0.2)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_slider, random_state=42)

                # Model
                regressor = LinearRegression()
                regressor.fit(X_train, y_train)

                # Prediction
                y_pred = regressor.predict(X_test)
                st.write("Prediction: ", y_pred)

                # Accuracy
                st.write("Accuracy: ", regressor.score(X_test, y_test))

                # MSE
                st.write("MSE: ", mean_squared_error(y_test, y_pred))

                # R2
                st.write("R2: ", r2_score(y_test, y_pred))

                # Pickle
                pickle.dump(regressor, open("model.pkl", "wb"))
                st.download_button("Download pickled model", data=open("model.pkl", "rb"), file_name="model.pkl")

                


if __name__ == "__main__":
    main()