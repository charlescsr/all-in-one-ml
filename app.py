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
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score, davies_bouldin_score, calinski_harabasz_score, mean_squared_error, r2_score
import pickle

def main():
    st.title("All in one ML")
    st.text("This app will help in Preprocessing and Model Building")

    # Sidebar in streamlit with the following choices: Preprocessing, Pickling.
    choice = st.sidebar.selectbox("Select a page", ["Preprocessing", "Pickling", "Cross Validation"])

    if choice == "Preprocessing":
        st.subheader("Preprocessing Page")
        st.write("This is the page to preprocess the data.")

        # File input
        file = st.file_uploader("Upload a file", type=["csv"])
        if file is not None:
            df = pd.read_csv(file)

            if any(df.columns.str.contains('^Unnamed')):
                # Remove unnamed columns
                df.drop(df.columns[df.columns.str.contains('^Unnamed')], axis=1, inplace=True)

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
            df = pd.read_csv(file)

            if any(df.columns.str.contains('^Unnamed')):
                df.drop(df.columns[df.columns.str.contains('^Unnamed')], axis=1, inplace=True)

            st.dataframe(df)

            model = st.selectbox("Select model", ["", "Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Logistic Regression", "SVM", "KNN", "Decision Tree Classifier", "Random Forest Classifier", "K Means Clustering"])

            if model == "Linear Regression":
                st.write("Linear Regression")
                st.write("This is the page to pickle the model.")

                # Target column
                target_column = st.selectbox("Select target column", df.columns)

                # Model
                X = df.drop(target_column, axis=1)
                y = df[target_column]

                # Train test split slider
                train_test_split_slider = st.slider("Select train test split", 0.1, 0.9, 0.1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_slider, random_state=42)

                # Options to customize Linear Regressor Object in Sklearn
                # Defaults: fit_intercept=True, copy_X=True, n_jobs=None, positive=False
                fit_intercept_checkbox = st.checkbox("Fit intercept")
                copy_X_checkbox = st.checkbox("Copy X")
                n_jobs_slider = st.slider("Select number of jobs", 1, 10, 1)
                positive_checkbox = st.checkbox("Positive")

                fit_intercept = True
                copy_X = True
                n_jobs = None
                positive = False


                if not fit_intercept_checkbox:
                    fit_intercept = False

                if not copy_X_checkbox:
                    copy_X = True

                if positive_checkbox:
                    positive = True

                if n_jobs_slider == 1:
                    n_jobs = None

                else:
                    n_jobs = n_jobs_slider

                # Model
                regressor = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)
                
                if st.button("Train"):
                    regressor.fit(X_train, y_train)

                    # Prediction
                    y_pred = regressor.predict(X_test)
                    st.write("Prediction: ", y_pred)

                    # Accuracy
                    st.write("Accuracy: ", regressor.score(X_test, y_test) * 100)

                    # MSE
                    st.write("MSE: ", mean_squared_error(y_test, y_pred))

                    # RMSE
                    st.write("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

                    # R2
                    st.write("R2: ", r2_score(y_test, y_pred))

                    # Pickle
                    pickle.dump(regressor, open("model.pkl", "wb"))
                    st.download_button("Download pickled model", data=open("model.pkl", "rb"), file_name="model.pkl")

            elif model == "Decision Tree Regressor":
                st.write("Decision Tree Regressor")
                st.write("This is the page to pickle the model.")

                # Target column
                target_column = st.selectbox("Select target column", df.columns)

                # Model
                X = df.drop(target_column, axis=1)
                y = df[target_column]

                # Train test split slider
                train_test_split_slider = st.slider("Select train test split", 0.1, 0.9, 0.1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_slider, random_state=42)

                # Options to customize Decision Tree Regressor Object in Sklearn
                # Defaults: criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1
                criterion_selectbox = st.selectbox("Select criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"])
                
                splitter_selectbox = st.selectbox("Select splitter", ["best", "random"])

                max_depth_slider = st.slider("Select max depth", 1, 10, 1)

                min_samples_split_slider = st.slider("Select min samples split", 1, 10, 1)

                min_samples_leaf_slider = st.slider("Select min samples leaf", 1, 10, 1)
                

                # Model
                regressor = DecisionTreeRegressor(criterion=criterion_selectbox, splitter=splitter_selectbox, max_depth=max_depth_slider, min_samples_split=min_samples_split_slider, min_samples_leaf=min_samples_leaf_slider)
                
                if st.button("Train"):
                    regressor.fit(X_train, y_train)

                    # Prediction
                    y_pred = regressor.predict(X_test)
                    st.write("Prediction: ", y_pred)

                    # Accuracy
                    st.write("Accuracy: ", regressor.score(X_test, y_test) * 100)

                    # MSE
                    st.write("MSE: ", mean_squared_error(y_test, y_pred))

                    # RMSE
                    st.write("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

                    # R2
                    st.write("R2: ", r2_score(y_test, y_pred))

                    # Pickle
                    pickle.dump(regressor, open("model.pkl", "wb"))
                    st.download_button("Download pickled model", data=open("model.pkl", "rb"), file_name="model.pkl")

            elif model == "Random Forest Regressor":
                st.write("Random Forest Regressor")
                st.write("This is the page to pickle the model.")

                # Target column
                target_column = st.selectbox("Select target column", df.columns)

                # Model
                X = df.drop(target_column, axis=1)
                y = df[target_column]

                # Train test split slider
                train_test_split_slider = st.slider("Select train test split", 0.1, 0.9, 0.1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_slider, random_state=42)

                # Model
                regressor = RandomForestRegressor()
                
                if st.button("Train"):
                    regressor.fit(X_train, y_train)

                    # Prediction
                    y_pred = regressor.predict(X_test)
                    st.write("Prediction: ", y_pred)

                    # Accuracy
                    st.write("Accuracy: ", regressor.score(X_test, y_test) * 100)

                    # MSE
                    st.write("MSE: ", mean_squared_error(y_test, y_pred))

                    # RMSE
                    st.write("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

                    # R2
                    st.write("R2: ", r2_score(y_test, y_pred))

                    # Pickle
                    pickle.dump(regressor, open("model.pkl", "wb"))
                    st.download_button("Download pickled model", data=open("model.pkl", "rb"), file_name="model.pkl")

            elif model == "Logistic Regression":
                st.write("Logistic Regression")
                st.write("This is the page to pickle the model.")

                # Target column
                target_column = st.selectbox("Select target column", df.columns)

                # Model
                X = df.drop(target_column, axis=1)
                y = df[target_column]

                # Train test split slider
                train_test_split_slider = st.slider("Select train test split", 0.1, 0.9, 0.1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_slider, random_state=42)

                # Options to customize Logistic Regression Object in Sklearn
                # Defaults: penalty='l2', C=1.0, fit_intercept=True, solver='liblinear', max_iter=100
                penalty_selectbox = st.selectbox("Select penalty", ["l1", "l2"])

                C_slider = st.slider("Select C", 1, 10, 1)
                C_slider = float(C_slider)

                fit_intercept_checkbox = st.checkbox("Select fit intercept")
                fit_intercept_checkbox = bool(fit_intercept_checkbox)

                solver_selectbox = st.selectbox("Select solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"])

                max_iter_slider = st.slider("Select max iter", 1, 100, 1)


                # Model
                if st.button("Train"):
                    regressor = LogisticRegression(penalty=penalty_selectbox, C=C_slider, fit_intercept=fit_intercept_checkbox, solver=solver_selectbox, max_iter=max_iter_slider)
                    regressor.fit(X_train, y_train)

                    # Prediction
                    y_pred = regressor.predict(X_test)
                    st.write("Prediction: ", y_pred)

                    # Accuracy score
                    st.write("Accuracy: ", regressor.score(X_test, y_test) * 100)

                    # Confusion matrix
                    st.write("Confusion matrix: ", confusion_matrix(y_test, y_pred))

                    # Classification report
                    classification_report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
                    st.dataframe(classification_report_df)

                    # Pickle
                    pickle.dump(regressor, open("model.pkl", "wb"))
                    st.download_button("Download pickled model", data=open("model.pkl", "rb"), file_name="model.pkl")

            elif model == "SVM":
                st.write("SVM")
                st.write("This is the page to pickle the model.")

                # Target column
                target_column = st.selectbox("Select target column", df.columns)

                # Model
                X = df.drop(target_column, axis=1)
                y = df[target_column]

                # Train test split slider
                train_test_split_slider = st.slider("Select train test split", 0.1, 0.9, 0.1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_slider, random_state=42)

                # Options to customize SVM Object in Sklearn
                # Defaults: C=1.0, kernel='rbf', degree=3, gamma='auto', probability=False, tol=0.001, max_iter=-1
                C_slider = st.slider("Select C", 1, 10, 1)
                C_slider = float(C_slider)

                kernel_selectbox = st.selectbox("Select kernel", ["linear", "poly", "rbf", "sigmoid"])

                degree_slider = st.slider("Select degree", 1, 10, 1)
                degree_slider = int(degree_slider)

                gamma_selectbox = st.selectbox("Select gamma", ["auto", "scale"])

                probability_checkbox = st.checkbox("Select probability")
                probability_checkbox = bool(probability_checkbox)

                tol_slider = st.slider("Select tol", 0.001, 0.01, 0.001)
                tol_slider = float(tol_slider)

                max_iter_slider = st.slider("Select max iter", 1, 100, 1)
                max_iter_slider = int(max_iter_slider)

                # Model
                regressor = SVC(C=C_slider, kernel=kernel_selectbox, degree=degree_slider, gamma=gamma_selectbox, probability=probability_checkbox, tol=tol_slider, max_iter=max_iter_slider)
                
                if st.button("Train"):
                    regressor.fit(X_train, y_train)

                    # Prediction
                    y_pred = regressor.predict(X_test)
                    st.write("Prediction: ", y_pred)

                    # Accuracy
                    st.write("Accuracy: ", regressor.score(X_test, y_test) * 100)

                    # Confusion matrix
                    st.write("Confusion matrix: ", confusion_matrix(y_test, y_pred))

                    # Classification report
                    classification_report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
                    st.dataframe(classification_report_df)

                    # Pickle
                    pickle.dump(regressor, open("model.pkl", "wb"))
                    st.download_button("Download pickled model", data=open("model.pkl", "rb"), file_name="model.pkl")

            elif model == "KNN":
                st.write("KNN")
                st.write("This is the page to pickle the model.")

                # Target column
                target_column = st.selectbox("Select target column", df.columns)

                # Model
                X = df.drop(target_column, axis=1)
                y = df[target_column]

                # Train test split slider
                train_test_split_slider = st.slider("Select train test split", 0.1, 0.9, 0.1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_slider, random_state=42)

                # Options to customize KNN Object in Sklearn
                # Defaults: n_neighbors=5, algorithm='auto', metric='minkowski'
                n_neighbors_slider = st.slider("Select n_neighbors", 1, 10, 1)
                n_neighbors_slider = int(n_neighbors_slider)

                algorithm_selectbox = st.selectbox("Select algorithm", ["auto", "ball_tree", "kd_tree", "brute"])

                metric_selectbox = st.selectbox("Select metric", ["minkowski", "euclidean", "manhattan"])
                
                # Model
                classifier = KNeighborsClassifier(n_neighbors=n_neighbors_slider, algorithm=algorithm_selectbox, metric=metric_selectbox)
                
                if st.button("Train"):
                    classifier.fit(X_train, y_train)

                    # Prediction
                    y_pred = classifier.predict(X_test)
                    st.write("Prediction: ", y_pred)

                    # Accuracy
                    st.write("Accuracy: ", classifier.score(X_test, y_test))

                    # Confusion matrix
                    st.write("Confusion matrix: ", confusion_matrix(y_test, y_pred))

                    # Classification report
                    classification_report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
                    st.dataframe(classification_report_df)

                    # Pickle
                    pickle.dump(classifier, open("model.pkl", "wb"))
                    st.download_button("Download pickled model", data=open("model.pkl", "rb"), file_name="model.pkl")

            elif model == "Decision Tree Classifier":
                st.write("Decision Tree Classifier")
                st.write("This is the page to pickle the model.")

                # Target column
                target_column = st.selectbox("Select target column", df.columns)

                # Model
                X = df.drop(target_column, axis=1)
                y = df[target_column]

                # Train test split slider
                train_test_split_slider = st.slider("Select train test split", 0.1, 0.9, 0.1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_slider, random_state=42)

                # Options to customize Decision Tree Classifier Object in Sklearn
                # Defaults: criterion='gini', splitter='best', max_depth=None
                criterion_selectbox = st.selectbox("Select criterion", ["gini", "entropy"])

                splitter_selectbox = st.selectbox("Select splitter", ["best", "random"])

                max_depth_slider = st.slider("Select max depth", 1, 10, 1)
                max_depth_slider = int(max_depth_slider)
                
                # Model
                classifier = DecisionTreeClassifier(criterion=criterion_selectbox, splitter=splitter_selectbox, max_depth=max_depth_slider)
                
                if st.button("Train"):
                    classifier.fit(X_train, y_train)

                    # Prediction
                    y_pred = classifier.predict(X_test)
                    st.write("Prediction: ", y_pred)

                    # Accuracy
                    st.write("Accuracy: ", classifier.score(X_test, y_test))

                    # Confusion matrix
                    st.write("Confusion matrix: ", confusion_matrix(y_test, y_pred))

                    # Classification report
                    classification_report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
                    st.dataframe(classification_report_df)

                    # Pickle
                    pickle.dump(classifier, open("model.pkl", "wb"))
                    st.download_button("Download pickled model", data=open("model.pkl", "rb"), file_name="model.pkl")

            elif model == "Random Forest Classifier":
                st.write("Random Forest Classifier")
                st.write("This is the page to pickle the model.")

                # Target column
                target_column = st.selectbox("Select target column", df.columns)

                # Model
                X = df.drop(target_column, axis=1)
                y = df[target_column]

                # Train test split slider
                train_test_split_slider = st.slider("Select train test split", 0.1, 0.9, 0.1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_slider, random_state=42)

                # Options to customize Random Forest Classifier Object in Sklearn
                # Defaults: n_estimators=100, max_features='sqrt', criterion='gini', max_depth=None
                n_estimators_slider = st.slider("Select n_estimators", 1, 100, 1)
                n_estimators_slider = int(n_estimators_slider)

                max_features_selectbox = st.selectbox("Select max features", ["sqrt", "log2", "auto"])

                criterion_selectbox = st.selectbox("Select criterion", ["gini", "entropy"])

                max_depth_slider = st.slider("Select max depth", 1, 10, 1)
                max_depth_slider = int(max_depth_slider) 
                
                # Model
                classifier = RandomForestClassifier(n_estimators=n_estimators_slider, max_features=max_features_selectbox, criterion=criterion_selectbox, max_depth=max_depth_slider)
                
                if st.button("Train"):
                    classifier.fit(X_train, y_train)

                    # Prediction
                    y_pred = classifier.predict(X_test)
                    st.write("Prediction: ", y_pred)

                    # Accuracy
                    st.write("Accuracy: ", classifier.score(X_test, y_test))

                    # Confusion matrix
                    st.write("Confusion matrix: ", confusion_matrix(y_test, y_pred))

                    # Classification report
                    classification_report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))
                    st.dataframe(classification_report_df)

                    # Pickle
                    pickle.dump(classifier, open("model.pkl", "wb"))
                    st.download_button("Download pickled model", data=open("model.pkl", "rb"), file_name="model.pkl")

            elif model == "K Means Clustering":
                st.write("K Means Clustering")
                st.write("This is the page to pickle the model.")

                # Target column
                target_column = st.selectbox("Select target column", df.columns)

                # Model
                X = df.drop(target_column, axis=1)
                y = df[target_column]

                # Train test split slider
                train_test_split_slider = st.slider("Select train test split", 0.1, 0.9, 0.1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_slider, random_state=42)

                # Options to customize K Means Clustering Object in Sklearn
                # Defaults: n_clusters=3, init='k-means++', n_init=10, max_iter=300, algorithm='lloyd'
                n_clusters_slider = st.slider("Select n_clusters", 1, 10, 1)
                n_clusters_slider = int(n_clusters_slider)

                init_selectbox = st.selectbox("Select init", ["k-means++", "random"])

                n_init_slider = st.slider("Select n_init", 1, 10, 1)
                n_init_slider = int(n_init_slider)

                max_iter_slider = st.slider("Select max_iter", 1, 1000, 1)
                max_iter_slider = int(max_iter_slider)

                algorithm_selectbox = st.selectbox("Select algorithm", ["lloyd", "elkan", "auto", "full"])

                # Model
                ml = KMeans(n_clusters=n_clusters_slider, init=init_selectbox, n_init=n_init_slider, max_iter=max_iter_slider, algorithm=algorithm_selectbox)
                
                if st.button("Train"):
                    ml.fit(X_train)

                    # Prediction
                    y_pred = ml.predict(X_test)
                    st.write("Prediction: ", y_pred)

                    # Silhouette score
                    st.write("Silhouette score: ", silhouette_score(X_train, ml.labels_))

                    # Calinski-Harabasz score
                    st.write("Calinski-Harabasz score: ", calinski_harabasz_score(X_train, ml.labels_))

                    # Davies-Bouldin score
                    st.write("Davies-Bouldin score: ", davies_bouldin_score(X_train, ml.labels_))

                    # Pickle
                    pickle.dump(ml, open("model.pkl", "wb"))
                    st.download_button("Download pickled model", data=open("model.pkl", "rb"), file_name="model.pkl")
    
    elif choice == "Cross Validation":
        # 2 fields: One to upload model and one to upload data
        st.write("Cross Validation")
        st.write("This is the page to perform cross validation of the given model.")

        # Upload model
        uploaded_file = st.file_uploader("Upload pickled model", type=["pickle", "pkl"])

        if uploaded_file is not None:
            pkl = uploaded_file.getvalue()
            model = pickle.loads(pkl)

        # Upload data
        uploaded_file = st.file_uploader("Upload data", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            if any(df.columns.str.contains('^Unnamed')):
                df.drop(df.columns[df.columns.str.contains('^Unnamed')], axis=1, inplace=True)
            
            st.write(df)

            # Target column
            target_column = st.selectbox("Select target column", df.columns)

            # Model
            X = df.drop(target_column, axis=1)
            y = df[target_column]

            # Cross validation
            # Choice of Fold method
            fold_method_selectbox = st.selectbox("Select fold method", ["", "KFold", "StratifiedKFold"])

            # Choice of K
            k_slider = st.slider("Select k", 1, 10, 1)

            # Choice of shuffle
            shuffle_selectbox = st.selectbox("Select shuffle", ["True", "False"])
            shuffle_selectbox = bool(shuffle_selectbox)

            # Choice of random state
            random_state_slider = st.number_input("Select random state", value=42)

            # Cross validation
            if fold_method_selectbox == "KFold":
                cv = KFold(n_splits=k_slider, shuffle=shuffle_selectbox, random_state=random_state_slider)

            elif fold_method_selectbox == "StratifiedKFold":
                cv = StratifiedKFold(n_splits=k_slider, shuffle=shuffle_selectbox, random_state=random_state_slider)

            # Cross val score
            if st.button("Cross Validate"):
                cross = cross_val_score(model, X, y, cv=cv)

                # Display max accuracy, min and avg accuracy
                st.write("Max accuracy: ", cross.max() * 100)
                st.write("Min accuracy: ", cross.min() * 100)
                st.write("Avg accuracy: ", cross.mean() * 100)


if __name__ == "__main__":
    main()