from cmath import nan
import inspect
from sklearn import svm
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.iloc[-3][0] = nan
    df_classification_report.iloc[-3][1] = nan
    df_classification_report.iloc[-3][3] = df_classification_report.iloc[-2][3]
    return df_classification_report

def get_result(model, X, X_train, X_test, y_test, feature_columns):
    y_pred = model.predict(X_test)

    # Display the results
    st.markdown("### Classification Results ###")
    st.markdown("Number of Cases: {}".format(X_train.shape[0]))

    # Create two columns
    col1, col2 = st.columns(2)

    # Display the classification report in the first column
    with col1:
        st.markdown("##### Classification report #####")
        st.markdown("<div style='margin-top: 60px'></div>", unsafe_allow_html=True)
        st.dataframe(get_classification_report(y_test, y_pred))

    # Display the confusion matrix in the second column
    with col2:
        st.markdown("##### Confusion matrix #####")
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.tolist()
        labels = np.unique(y_test)
        fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels, y=labels, text_auto=True)
        st.plotly_chart(fig)

    # Get the column names from the data
    columns = feature_columns

    # Create a dictionary to store the entered values
    inputs = {}

    # Create a list of columns
    cols = st.columns(len(columns))

    # Create a text field for each column in the corresponding column
    for i, column in enumerate(columns):
        inputs[column] = cols[i].text_input(f'Value for {column}')

    # Create a dataframe from the entered values
    input_data = pd.DataFrame([inputs])
    for column in input_data.columns:
        input_data[column] = pd.to_numeric(input_data[column], errors='ignore')

    if input_data.isnull().values.any():
        st.warning('Warning: Vui lòng điền đầy đủ các giá trị.')
    else:
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        # Make a prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.write('The predicted label is: ', prediction[0])

def pre_train(data):
        data_copy = data.copy()

        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_copy.columns)

        # Select target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_copy.columns)

        if not feature_columns:
            st.warning("Chon cot tinh nang")

            return None, None, None
        else:
            # Split the data into training and testing sets
            standardScaler = StandardScaler()
            minMaxScaler = MinMaxScaler()
            X = data_copy[feature_columns]
            X = pd.get_dummies(X)

            scaler_type = st.selectbox('Chọn kiểu scale',('None', 'Standard Scaler', 'Min-max Scaler'))
            if scaler_type == 'None':
                X = X
            elif scaler_type == 'Standard Scaler':
                X = standardScaler.fit_transform(X)
            elif scaler_type == 'Min-max Scaler':
                X = minMaxScaler.fit_transform(X)
            y = data_copy[target_column]

            return X, y, feature_columns

def train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


class Classification:
    def knn_classification(data):
        # Create a copy of the data
            X, y, feature_columns = pre_train(data)

            if X is not None and y is not None and feature_columns is not None:

                X_train, X_test, y_train, y_test = train_test(X, y)

                with st.popover("Choose parameter", use_container_width=True):
                    n_neighbors = st.slider("Choose n_neighbors", 1, 50, value=5)
                    weights = st.selectbox("Choose weights", ('uniform', 'distance'))
                    metric = st.selectbox("Choose metric", ('minkowski', 'euclidean', 'manhattan', 'chebyshev', 'cosine', 'canberra'))
                    algorithm = st.selectbox("Choose algorithm", ('auto', 'ball_tree', 'kd_tree', 'brute'))
                    leaf_size = st.number_input("Choose leaf_size", value=30)

                params = {
                    'n_neighbors': n_neighbors,
                    'weights': weights,
                    'metric': metric,
                    'algorithm': algorithm,
                    'leaf_size': leaf_size
                }

                # Create and train the KNN classifier
                knn = KNeighborsClassifier()
                knn.set_params(**params)
                knn.fit(X_train, y_train)

                get_result(knn, X, X_train, X_test, y_test, feature_columns)

    def lgreg_classification(data):

        X, y, feature_columns = pre_train(data)

        if X is not None and y is not None and feature_columns is not None:

            X_train, X_test, y_train, y_test = train_test(X, y)

            with st.popover("Choose parameter", use_container_width=True):
                penalty = st.selectbox("Choose penalty", ('l1', 'l2', 'elasticnet', None), index=1)
                dual = st.selectbox("Choose dual", (True, False), index=1)
                tol = float(st.text_input("Choose tolerance", value=0.0001))
                C_param = st.number_input("Choose C", value=1.0)
                fit_intercept = st.selectbox("Choose fit_intercept", (True, False))
                intercept_scaling = st.number_input("Choose intercept_scaling", value=1)
                class_weight = st.selectbox("Choose class_weight", ('balanced', None), index=1)
                random_state = st.number_input("Choose random_state", value=None)
                solver = st.selectbox("Choose solver", ('lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga', 'newton-cholesky'))
                max_iter = st.number_input("Choose max_iter", value=100)
                verbose = st.number_input("Choose verbose", value=0)
                warm_start = st.selectbox("Choose warm_start", (True, False), index=1)
                n_jobs = st.number_input("Choose n_jobs", value=None)
                l1_ratio = st.number_input("Choose l1_ratio", value=None)

            params = {
                'penalty': penalty,
                'dual': dual,
                'tol': tol,
                'C': C_param,
                'fit_intercept': fit_intercept,
                'intercept_scaling': intercept_scaling,
                'class_weight': class_weight,
                'random_state': random_state,
                'solver': solver,
                'max_iter': max_iter,
                'verbose': verbose,
                'warm_start': warm_start,
                'n_jobs': n_jobs,
                'l1_ratio': l1_ratio
            }

            # Create and train the KNN classifier
            lgreg = LogisticRegression()
            lgreg.set_params(**params)
            lgreg.fit(X_train, y_train)

            get_result(lgreg, X, X_train, X_test, y_test, feature_columns)

    def randomfor_classification(data):

        X, y, feature_columns = pre_train(data)

        if X is not None and y is not None and feature_columns is not None:

            X_train, X_test, y_train, y_test = train_test(X, y)

            with st.popover("Choose parameter", use_container_width=True):
                n_estimators = st.number_input("Choose n_estimators", value=100)
                criterion = st.selectbox("Choose criterion", ('gini', 'entropy', 'log_loss'))
                max_depth = st.number_input("Choose max_depth", value=None)
                min_samples_split = st.number_input("Choose min_samples_split", value=2)
                min_samples_leaf = st.number_input("Choose min_samples_leaf", value=1)
                min_weight_fraction_leaf = st.number_input("Choose min_weight_fraction_leaf", value=0.0)
                max_features = st.selectbox("Choose max_features", ('sqrt', 'log2', None))
                max_leaf_nodes = st.number_input("Choose max_leaf_nodes", value=None)
                min_impurity_decrease = st.number_input("Choose min_impurity_decrease", value=0.0)
                bootstrap = st.selectbox("Choose bootstrap", (True, False), index=0)
                oob_score = st.selectbox("Choose oob_score", (True, False), index=1)
                n_jobs = st.number_input("Choose n_jobs", value=None)
                random_state = st.number_input("Choose random_state", value=None)
                verbose = st.number_input("Choose verbose", value=0)
                warm_start = st.selectbox("Choose warm_start", (True, False), index=1)
                ccp_alpha = st.number_input("Choose ccp_alpha", value=0.0)
                max_samples = st.number_input("Choose max_samples", value=None)

            params = {
                'n_estimators': n_estimators,
                'criterion': criterion,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'min_weight_fraction_leaf': min_weight_fraction_leaf,
                'max_features': max_features,
                'max_leaf_nodes': max_leaf_nodes,
                'min_impurity_decrease': min_impurity_decrease,
                'bootstrap': bootstrap,
                'oob_score': oob_score,
                'n_jobs': n_jobs,
                'random_state': random_state,
                'verbose': verbose,
                'warm_start': warm_start,
                'ccp_alpha': ccp_alpha,
                'max_samples': max_samples
            }

            rf = RandomForestClassifier()
            rf.set_params(**params)
            rf.fit(X_train, y_train)

            get_result(rf, X, X_train, X_test, y_test, feature_columns)

    def naivebayes_classification(data):

        X, y, feature_columns = pre_train(data)

        if X is not None and y is not None and feature_columns is not None:

            X_train, X_test, y_train, y_test = train_test(X, y)

            gnb = GaussianNB()
            gnb.fit(X_train, y_train)

            get_result(gnb, X, X_train, X_test, y_test, feature_columns)

    def svm_classification(data):
        X, y, feature_columns = pre_train(data)

        if X is not None and y is not None and feature_columns is not None:

            X_train, X_test, y_train, y_test = train_test(X, y)

            svc = SVC()
            svc.fit(X_train, y_train)

            get_result(svc, X, X_train, X_test, y_test, feature_columns)

    def run(data):
        st.write(" # Phân lớp # ")
        st.write("#### Dữ liệu ####")
        st.write("Data")
        with st.expander("See data", expanded=True):
            edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
        st.markdown("---")
        class_type = st.selectbox("", ["KNN", 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM'])
        if class_type == 'KNN':
            Classification.knn_classification(edit_data)
        if class_type == 'Logistic Regression':
            Classification.lgreg_classification(edit_data)
        if class_type == 'Random Forest':
            Classification.randomfor_classification(edit_data)
        if class_type == 'Naive Bayes':
            Classification.naivebayes_classification(edit_data)
        if class_type == 'SVM':
            Classification.svm_classification(edit_data)
