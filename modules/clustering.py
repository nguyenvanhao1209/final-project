import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.graph_objects as go


def get_result(model, X, X_scaled, feature_columns):
    # Add cluster labels to the data
    X["cluster"] = model.labels_
    X["cluster"] = X["cluster"].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Clustering Visualize #####")
        if len(feature_columns) <= 2:
            fig = px.scatter(X, x=X.iloc[:, 0], y=X.iloc[:, 1], color="cluster")
            # st.markdown("Number of Clusters: {}".format(n_clusters))
            st.plotly_chart(fig, use_container_width=True)
            # cluster_labels = X["cluster"]
            # cluster_counts = cluster_labels.value_counts()
            # st.write(cluster_counts)
        else:
            pass

    with col2:

        st.markdown("##### Clustering Result Visualize #####")
        silhouette_avg = silhouette_score(X_scaled, model.labels_)
        st.markdown(f"Silhouette Score: {silhouette_avg:.4f}")
        st.dataframe(X, use_container_width=True)


def pre_train(data):
    data_copy = data.copy()

    data_number_columns = data_copy.select_dtypes(include=["int", "float"]).columns
    feature_columns = st.multiselect("Chọn biến tính năng", data_number_columns)
    X = data_copy[feature_columns]

    if not feature_columns:
        st.warning("Chon cot tinh nang")
        return None, None, None
    else:
        scaler_type = st.selectbox("Chọn kiểu scale", ("None", "Standard Scaler", "Min-max Scaler"))
        if scaler_type == "None":
            X_scaled = X
        if scaler_type == "Standard Scale":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        if scaler_type == "Min-max Scale":
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
        return X_scaled, X, feature_columns


class Clustering:
    def kmeans_clustering(data):
        # Create a copy of the data
        X_scaled, X, feature_columns = pre_train(data)
        # Select the number of clusters

        # Perform K-Means clustering
        if X_scaled is not None and feature_columns is not None:
            n_clusters = st.slider("Chọn số lượng cụm", 2, 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_scaled)

            get_result(kmeans, X, X_scaled, feature_columns)

    def dbscan_clustering(data):

        # Create a copy of the data
        X_scaled, X, feature_columns = pre_train(data)

        if X_scaled is not None and feature_columns is not None:

            # Set DBSCAN hyperparameters
            col1, col2 = st.columns(2)
            with col1:
                eps = st.number_input("Nhập epsilon", value=0.5)
            with col2:
                min_samples = st.number_input(
                    "Nhập số lượng điểm tối thiểu", value=10, step=1
                )  # Adjust default value as needed

            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            dbscan.fit(X_scaled)

            # Get clustering result
            get_result(dbscan, X, X_scaled, feature_columns)

    def optics_clustering(data):
        # Create a copy of the data
        X_scaled, X, feature_columns = pre_train(data)

        if X_scaled is not None and feature_columns is not None:
            # Set OPTICS hyperparameters
            # Adjust default value as needed
            min_samples = st.number_input(
                "Nhập số lượng điểm tối thiểu", value=10, step=1
            )  # Adjust default value as needed

            # Perform OPTICS clustering
            optics = OPTICS(eps=0.5, min_samples=min_samples, metric="euclidean")
            optics.fit(X_scaled)

            # Extract DBSCAN-like clusters from the OPTICS output
            # Add cluster labels to the data
            get_result(optics, X, X_scaled, feature_columns)
            
    def meanshift(data):
        # Create a copy of the data
        X_scaled, X, feature_columns = pre_train(data)

        if X_scaled is not None and feature_columns is not None:
            # Set OPTICS hyperparameters
            # Adjust default value as needed
            col1, col2 = st.columns(2)
            with col1:
                quantile = st.number_input("Nhập epsilon", value=0.5)
            with col2:
                n_samples = st.number_input(
                    "Nhập số lượng điểm tối thiểu", value=10, step=1
                )  # Adjust default value as needed

            # Perform OPTICS clustering
            bandwidth = estimate_bandwidth(X_scaled, quantile=quantile, n_samples=n_samples)
            meanshift = MeanShift(bandwidth=bandwidth)
            meanshift.fit(X_scaled)

            # Extract DBSCAN-like clusters from the OPTICS output
            # Add cluster labels to the data
            get_result(meanshift, X, X_scaled, feature_columns)
            
    
    def run(data):
        st.write(" # Phân cụm # ")
        st.write("#### Dữ liệu ####")
        st.write("Data")
        with st.expander("See data", expanded=True):
            edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
        st.markdown("---")
        class_type = st.selectbox("", ["K Means", 'DBSCAN', 'OPTICS','Mean Shift'])
        if class_type == 'K Means':
            Clustering.kmeans_clustering(edit_data)
        if class_type == 'DBSCAN':
            Clustering.dbscan_clustering(edit_data)
        if class_type == 'OPTICS':
            Clustering.optics_clustering(edit_data)
        if class_type == 'Mean Shift':
            Clustering.meanshift(edit_data)
