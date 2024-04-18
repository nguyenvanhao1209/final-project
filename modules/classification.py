from cmath import nan
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


class Classification:
    def knn_classification(data):
        # Create a copy of the data
        data_copy = data.copy()

        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_copy.columns)

        # Select target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_copy.columns)

        if not feature_columns:
            st.warning("Chon cot tinh nang")
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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Get the number of neighbors from the user
            k = st.slider("Chọn số lượng hàng xóm k", 1, 50)

            # Create and train the KNN classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = knn.predict(X_test)

            # Display the results
            st.markdown("### K-Nearest Neighbors Classification Results ###")
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
                prediction = knn.predict(input_data)

                # Display the prediction
                st.write('The predicted label is: ', prediction[0])

    def lgreg_classification(data):
        # Create a copy of the data
        data_copy = data.copy()

        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_copy.columns)

        # Select target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_copy.columns)

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Get the number of neighbors from the user
        k = st.slider("Chọn số lần lặp tối đa k", 0, 500, step=50)

        # Create and train the KNN classifier
        knn = LogisticRegression(max_iter=k)
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(X_test)

        # Display the results
        st.markdown("### Logistic Regression Classification Results ###")
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
            prediction = knn.predict(input_data)

            # Display the prediction
            st.write('The predicted label is: ', prediction[0])

    def randomfor_classification(data):
        # Create a copy of the data
        data_copy = data.copy()

        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_copy.columns)

        # Select target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_copy.columns)

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Get the number of neighbors from the user
        k = st.slider("Chọn số cây k", 10, 200, step=10)

        # Create and train the KNN classifier
        knn = RandomForestClassifier(n_estimators=k)
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(X_test)

        # Display the results
        st.markdown("### Random Forest Classification Results ###")
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
            prediction = knn.predict(input_data)

            # Display the prediction
            st.write('The predicted label is: ', prediction[0])

    def naivebayes_classification(data):
        # Create a copy of the data
        data_copy = data.copy()

        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_copy.columns)

        # Select target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_copy.columns)

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the KNN classifier
        knn = GaussianNB()
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(X_test)

        # Display the results
        st.markdown("### Naive-Bayes Classification Results ###")
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
            prediction = knn.predict(input_data)

            # Display the prediction
            st.write('The predicted label is: ', prediction[0])

    def svm_classification(data):
        # Create a copy of the data
        data_copy = data.copy()

        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_copy.columns)

        # Select target variable
        target_column = st.selectbox("Chọn biến mục tiêu", data_copy.columns)

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)
        # Create and train the KNN classifier
        knn = SVC()
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(X_test)


        # Display the results
        st.markdown("### SVM Classification Results ###")
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
            prediction = knn.predict(input_data)

            # Display the prediction
            st.write('The predicted label is: ', prediction[0])
