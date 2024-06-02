import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.graph_objects as go

def get_result(model, X, X_scaled, feature_columns, n_clusters):
    # Add cluster labels to the data
    X["cluster"] = model.labels_
    X["cluster"] = X["cluster"].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Clustering Visualize #####")
        if len(feature_columns) <= 2:
            fig = px.scatter(X, x=X.iloc[:, 0], y=X.iloc[:, 1], color="cluster")
            st.markdown("Number of Clusters: {}".format(n_clusters))
            st.plotly_chart(fig, use_container_width=True)
            cluster_labels = X["cluster"]
            cluster_counts = cluster_labels.value_counts()
            st.write(cluster_counts)
        else:
            pass

    with col2:

        st.markdown("##### Clustering Result Visualize #####")
        silhouette_avg = silhouette_score(X_scaled, model.labels_)
        st.markdown(f"Silhouette Score: {silhouette_avg:.4f}")
        st.dataframe(X, use_container_width=True)

class Clustering:
    def kmeans_clustering(data):
        # Create a copy of the data
        data_copy = data.copy()

        data_number_colum = data_copy.select_dtypes(include=["int", "float"]).columns
        # Select feature variables
        feature_columns = st.multiselect("Chọn biến tính năng", data_number_colum)
        X = data_copy[feature_columns]
        # Select the number of clusters
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Chọn số lượng cụm", 2, 10)
        with col2:
            option = st.selectbox(
                "Chọn kiểu scale",
                ("None", "Standard Scale", "Min-max Scale"),
            )
        # Get the data for clustering

        # Standardize the data before clustering

        # Perform K-Means clustering
        if not feature_columns:
            st.warning("Chon cot tinh nang")
        else:
            if option == "None":
                X_scaled = X
            if option == "Standard Scale":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            if option == "Min-max Scale":
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_scaled)

            get_result(kmeans, X, X_scaled, feature_columns, n_clusters)

    def dbscan_clustering(data):

        # Create a copy of the data
        data_copy = data.copy()

        # Select numerical feature columns
        data_number_colum = data_copy.select_dtypes(include=["int", "float"]).columns
        feature_columns = st.multiselect("Chọn biến tính năng", data_number_colum)

        # Get the data for clustering
        X = data_copy[feature_columns]

        option = st.selectbox(
            "Chọn kiểu scale",
            ("None", "Standard Scale", "Min-max Scale"),
        )

        # Standardize the data before clustering
        if not feature_columns:
            st.warning("Chon cot tinh nang")
        else:
            if option == "None":
                X_scaled = X
            if option == "Standard Scale":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            if option == "Min-max Scale":
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)

            # Set DBSCAN hyperparameters
            col1, col2 = st.columns(2)
            with col1:
                eps = st.number_input('Nhập epsilon', value=0.5)
            with col2:
                min_samples = st.number_input('Nhập số lượng điểm tối thiểu', value=10, step=1)  # Adjust default value as needed

            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            dbscan.fit(X_scaled)

            # Add cluster labels to the data
            X["cluster"] = dbscan.labels_
            X["cluster"] = X["cluster"].astype(str)

            # Visualize the clusters
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Clustering Visualize #####")
                if len(feature_columns) <= 2:
                    fig = px.scatter(X, x=X.iloc[:, 0], y=X.iloc[:, 1], color="cluster")
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.write("Hello")

            with col2:
                st.markdown("##### Clustering Result Visualize #####")
                silhouette_avg = silhouette_score(X_scaled, dbscan.labels_)
                st.markdown(f"Silhouette Score: {silhouette_avg:.4f}")
                st.dataframe(X, use_container_width=True)

    def optics_clustering(data):
        # Create a copy of the data
        data_copy = data.copy()

        # Select numerical feature columns
        data_number_colum = data_copy.select_dtypes(include=["int", "float"]).columns
        feature_columns = st.multiselect("Chọn biến tính năng", data_number_colum)

        # Get the data for clustering
        X = data_copy[feature_columns]

        option = st.selectbox(
            "Chọn kiểu scale",
            ("None", "Standard Scale", "Min-max Scale"),
        )

        if not feature_columns:
            st.warning("Chon cot tinh nang")
        else:
            if option == "None":
                X_scaled = X
            if option == "Standard Scale":
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            if option == "Min-max Scale":
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)

            # Set OPTICS hyperparameters
            # Adjust default value as needed
            min_samples = st.number_input('Nhập số lượng điểm tối thiểu', value=10, step=1) # Adjust default value as needed

            # Perform OPTICS clustering
            optics = OPTICS(eps=0.5, min_samples=min_samples, metric='euclidean')
            optics.fit(X_scaled)

            # Extract DBSCAN-like clusters from the OPTICS output

            # Add cluster labels to the data
            X["cluster"] = optics.labels_
            X["cluster"] = X["cluster"].astype(str)

            # Visualize the clusters
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Clustering Visualize #####")
                if len(feature_columns) <= 2:
                    fig = px.scatter(X, x=X.iloc[:, 0], y=X.iloc[:, 1], color="cluster")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Hello")

            with col2:
                st.markdown("##### Clustering Result Visualize #####")
                silhouette_avg = silhouette_score(X_scaled, optics.labels_)
                st.markdown(f"Silhouette Score: {silhouette_avg:.4f}")
                st.dataframe(X, use_container_width=True)
