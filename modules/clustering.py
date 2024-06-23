import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.graph_objects as go
from PIL import Image


def get_result(model, X, X_scaled, feature_columns):
    # Add cluster labels to the data
    try:
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
    except:
        st.toast("Number of labels is 1", icon = "🐞")


def pre_train(data):
    data_copy = data.copy()

    data_number_columns = data_copy.select_dtypes(include=["int", "float"]).columns
    feature_columns = st.multiselect("Select feature columns", data_number_columns)
    

    if not feature_columns:
        st.container().warning("Please select feature columns")
        return None, None, None
    else:
        scaler_type = st.selectbox('Select scale type',('None', 'Standard Scaler', 'Min-max Scaler'))
        # Split the data into training and testing sets
        X = data_copy[feature_columns]
        X_scaled = X.copy()
        standardScaler = StandardScaler()
        minMaxScaler = MinMaxScaler()
        if scaler_type == 'None':
            X_scaled = X
        elif scaler_type == 'Standard Scaler':
            X_scaled = standardScaler.fit_transform(X)
        elif scaler_type == 'Min-max Scaler':
            X_scaled = minMaxScaler.fit_transform(X)
        return X_scaled, X, feature_columns


class Clustering:
    def kmeans_clustering(data):
        # Create a copy of the data
        X_scaled, X, feature_columns = pre_train(data)
        # Select the number of clusters

        # Perform K-Means clustering
        if X_scaled is not None and feature_columns is not None:
            with st.popover("Choose parameter", use_container_width=True):
                text_n_cluster = """
                ### n_clusters: int, default=8
                The number of clusters to form as well as the number of centroids to generate.
                """
                n_clusters = st.slider("Choose number of cluster", 2, 10, help=text_n_cluster)
                text_init = """
                ### init: {‘k-means++’, ‘random’}, callable or array-like of shape (n_clusters, n_features), default=’k-means++’
                Method for initialization:
                - ‘k-means++’ : selects initial cluster centroids using sampling based on an empirical probability distribution of the points’ contribution to the overall inertia. This technique speeds up convergence. The algorithm implemented is “greedy k-means++”. It differs from the vanilla k-means++ by making several trials at each sampling step and choosing the best centroid among them.
                - ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.
                """
                init = st.selectbox("Choose init", ("k-means++", "random"), help=text_init)
                text_max_iter = """
                ### max_iter: int, default=300
                Maximum number of iterations of the k-means algorithm for a single run.
                """
                max_iter = st.number_input("Choose max_iter", value=300, help=text_max_iter)
                text_random_state = """
                ### random_state: int, default=40
                Determines random number generation for centroid initialization. Use an int to make the randomness deterministic
                """
                random_state = st.number_input("Choose random state", value=40, help=text_random_state)
                text_algorithm = """
                ### algorithm: {“lloyd”, “elkan”}, default=”lloyd”
                K-means algorithm to use. The classical EM-style algorithm is "lloyd". The "elkan" variation can be more efficient on some datasets with well-defined clusters, by using the triangle inequality. However it’s more memory intensive due to the allocation of an extra array of shape
                """
                algorithm = st.selectbox("Choose algorithm", ("lloyd", "elkan"), help=text_algorithm)

            params = {
                "n_clusters": n_clusters,
                "init": init,
                "max_iter": max_iter,
                "random_state": random_state,
                "algorithm": algorithm,
            }
          
            kmeans = KMeans()
            kmeans.set_params(**params)
            kmeans.fit(X_scaled)

            get_result(kmeans, X, X_scaled, feature_columns)

    def dbscan_clustering(data):

        # Create a copy of the data
        X_scaled, X, feature_columns = pre_train(data)

        if X_scaled is not None and feature_columns is not None:

            with st.popover("Choose parameter", use_container_width=True):
                text_eps = """
                ### eps: float, default=0.5
                The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
                """
                eps = st.number_input("Choose eps", value=0.5, help=text_eps)
                text_min_samples = """
                ### min_samples: int, default=5
                The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. If min_samples is set to a higher value, DBSCAN will find denser clusters, whereas if it is set to a lower value, the found clusters will be more sparse.
                """
                min_samples = st.number_input("Choose min_samples", value=5, step=1, help=text_min_samples)
                text_metric = """
                ### metric: str, or callable, default=’euclidean’
                Maximum number of iterations of the k-means algorithm for a single run.
                """
                metric = st.selectbox(
                    "Choose metric", ("euclidean", "manhattan", "chebyshev", "cosine", "canberra"), help=text_metric
                )
                text_algorithm = """
                ### algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors. See NearestNeighbors module documentation for details.
                """
                algorithm = st.selectbox(
                    "Choose algorithm", ("auto", "ball_tree", "kd_tree", "brute"), help=text_algorithm
                )
                text_leaf_size = """
                ### leaf_size: int, default=30
                Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
                """
                leaf_size = st.number_input("Choose leaf_size", value=30, step=1, help=text_leaf_size)

            params = {
                "eps": eps,
                "min_samples": min_samples,
                "metric": metric,
                "leaf_size": leaf_size,
                "algorithm": algorithm,
            }
            # Perform DBSCAN clustering
            dbscan = DBSCAN()
            dbscan.set_params(**params)
            dbscan.fit(X_scaled)

            # Get clustering result
            get_result(dbscan, X, X_scaled, feature_columns)

    def optics_clustering(data):
        # Create a copy of the data
        X_scaled, X, feature_columns = pre_train(data)

        if X_scaled is not None and feature_columns is not None:
            with st.popover("Choose parameter", use_container_width=True):
                text_min_samples = """
                ### min_samples: int, default=5
                The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. If min_samples is set to a higher value, DBSCAN will find denser clusters, whereas if it is set to a lower value, the found clusters will be more sparse.
                """
                min_samples = st.number_input("Choose min_samples", value=5, step=1, help=text_min_samples)
                text_max_eps = """
                ### max_eps: float, default=np.inf
                The maximum distance between two samples for one to be considered as in the neighborhood of the other. Default value of np.inf will identify clusters across all scales; reducing max_eps will result in shorter run times.
                """
                max_eps = st.number_input("Choose eps", value=100000000000, help=text_max_eps)
                text_metric = """
                ### metric: str or callable, default=’minkowski’
                Metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.

                If metric is a callable function, it is called on each pair of instances (rows) and the resulting value recorded. The callable should take two arrays as input and return one value indicating the distance between them. This works for Scipy’s metrics, but is less efficient than passing the metric name as a string. If metric is “precomputed”, X is assumed to be a distance matrix and must be square.
                """
                metric = st.selectbox(
                    "Choose metric", ('minkowski',"euclidean", "manhattan", "chebyshev", "cosine", "canberra"), help=text_metric
                )
                
                text_cluster_method = """
                ### cluster_method: str, default=’xi’
                The extraction method used to extract clusters using the calculated reachability and ordering. Possible values are “xi” and “dbscan”.
                """
                cluster_method = st.selectbox(
                    "Choose cluster method", ("xi", "dbscan"), help=text_cluster_method
                )
                
                text_algorithm = """
                ### algorithm: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors. See NearestNeighbors module documentation for details.
                """
                algorithm = st.selectbox(
                    "Choose algorithm", ("auto", "ball_tree", "kd_tree", "brute"), help=text_algorithm
                )
                text_leaf_size = """
                ### leaf_size: int, default=30
                Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
                """
                leaf_size = st.number_input("Choose leaf_size", value=30, step=1, help=text_leaf_size)

            params = {
                "min_samples": min_samples,
                "max_eps": max_eps,
                "metric": metric,
                "cluster_method": cluster_method,
                "leaf_size": leaf_size,
                "algorithm": algorithm,
            }

            # Perform OPTICS clustering
            optics = OPTICS()
            optics.set_params(**params)
            optics.fit(X_scaled)

            # Extract DBSCAN-like clusters from the OPTICS output
            # Add cluster labels to the data
            get_result(optics, X, X_scaled, feature_columns)

    def meanshift_clustering(data):
        # Create a copy of the data
        X_scaled, X, feature_columns = pre_train(data)

        if X_scaled is not None and feature_columns is not None:
            with st.popover("Choose parameter", use_container_width=True):
                text_quantile = """
                ### quantile: float, default=0.3
                Should be between [0, 1] 0.5 means that the median of all pairwise distances is used.
                """
                quantile = st.number_input("Choose quantile", value=0.3, help=text_quantile)
                text_n_samples = """
                ### n_samples: int, default=None
                The number of samples to use. If not given, all samples are used.
                """
                n_samples = st.number_input(
                    "Choose n samples", value=None, step=1, help=text_n_samples
                )
                text_cluster_all = """
                ### cluster_all: bool, default=True
                If true, then all points are clustered, even those orphans that are not within any kernel. Orphans are assigned to the nearest kernel. If false, then orphans are given cluster label -1.
                """
                cluster_all = st.selectbox("Choose cluster_all", (True, False), index=0, help=text_cluster_all)
                text_max_iter = """
                ### max_iter: int, default=300
                Maximum number of iterations, per seed point before the clustering operation terminates (for that seed point), if has not converged yet.
                """
                max_iter = st.number_input("Choose max_iter", value=300, help=text_max_iter)
                

            
            bandwidth = estimate_bandwidth(X_scaled, quantile=quantile, n_samples=n_samples)
            
            params = {
                "bandwidth": bandwidth,
                "cluster_all": cluster_all,
                "max_iter": max_iter    
            }
            
            meanshift = MeanShift()
            meanshift.set_params(**params)
            meanshift.fit(X_scaled)

            
            get_result(meanshift, X, X_scaled, feature_columns)

    def run(data):
        content_image = Image.open("image/content6.png")
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(" # Clustering # ")
            st.markdown("""<p>Provide popular unsupervised clustering algorithms.</p>""", unsafe_allow_html=True)
        with col2:
            st.image(content_image)
        st.markdown("---")
        st.write("#### Your data ####")
        with st.expander("See data", expanded=True):
            edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
        st.markdown("---")
        class_type = st.selectbox("Select clustering algorithm", ["K Means", "DBSCAN", "OPTICS", "Mean Shift"])
        if class_type == "K Means":
            Clustering.kmeans_clustering(edit_data)
        if class_type == "DBSCAN":
            Clustering.dbscan_clustering(edit_data)
        if class_type == "OPTICS":
            Clustering.optics_clustering(edit_data)
        if class_type == "Mean Shift":
            Clustering.meanshift_clustering(edit_data)
