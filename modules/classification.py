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
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PIL import Image


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
    with st.popover("Train and test subsets", use_container_width=True):
        text_test_size = """
        ### test_size: float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
        """
        test_size = st.slider("Choose test size", min_value=0.0, max_value=1.0, step=0.01, value=0.25, help=text_test_size)
        text_random_state = """
        ### random_state: int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
        """
        random_state = st.number_input("Choose random state", value=None, help=text_random_state)
        text_stratify = """
        ### stratify: array-like, default=None
        If not None, data is split in a stratified fashion, using this as the class labels.
        """
        stratify_box = st.checkbox("Use stratify", value=False, help=text_stratify)
        if stratify_box:
            stratify = y
        else:
            stratify = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    return X_train, X_test, y_train, y_test


class Classification:
    def knn_classification(data):
        # Create a copy of the data
            X, y, feature_columns = pre_train(data)

            if X is not None and y is not None and feature_columns is not None:

                X_train, X_test, y_train, y_test = train_test(X, y)

                with st.popover("Choose parameter", use_container_width=True):
                    text_n_neighbors = """
                    ### **n_neighbors**: ***int***, **default=5**.
                    Number of neighbors to use by default for kneighbors queries."""
                    n_neighbors = st.slider("Choose n_neighbors", 1, 50, value=5, help=text_n_neighbors)
                    text_weights = """
                    ### weights: {‘uniform’, ‘distance’}, callable or None, default=’uniform’
                    Weight function used in prediction. Possible values:
                    - ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                    - ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
                    - [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.
                    """
                    weights = st.selectbox("Choose weights", ('uniform', 'distance'), help=text_weights)
                    text_metrix = """
                    ### metric: string or callable, default=’minkowski’
                    Metric to use for distance computation. Default is “minkowski”, which results in the standard Euclidean distance when p = 2. See the documentation of scipy.spatial.distance and the metrics listed in distance_metrics for valid metric values.
                    """
                    metric = st.selectbox("Choose metric", ('minkowski', 'euclidean', 'manhattan', 'chebyshev', 'cosine', 'canberra'), help=text_metrix)
                    text_algorithm = """
                    ### algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                    Algorithm used to compute the nearest neighbors:
                    - ‘ball_tree’ will use BallTree
                    - ‘kd_tree’ will use KDTree
                    - ‘brute’ will use a brute-force search.
                    - ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
                    #### Note: fitting on sparse input will override the setting of this parameter, using brute force.
                    """
                    algorithm = st.selectbox("Choose algorithm", ('auto', 'ball_tree', 'kd_tree', 'brute'), help=text_algorithm)
                    text_leaf_size = """
                    ### leaf_size: int, default=30
                    Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
                    """
                    leaf_size = st.number_input("Choose leaf_size", value=30, help=text_leaf_size)

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
                text_penalty = """
                ### penalty: {‘l1’, ‘l2’, ‘elasticnet’, None}, default=’l2’
                Specify the norm of the penalty:
                - None: no penalty is added;
                - 'l2': add a L2 penalty term and it is the default choice;
                - 'l1': add a L1 penalty term;
                - 'elasticnet': both L1 and L2 penalty terms are added.
                """
                penalty = st.selectbox("Choose penalty", ('l1', 'l2', 'elasticnet', None), index=1, help=text_penalty)
                text_dual = """
                ### dual: bool, default=False
                Dual (constrained) or primal (regularized, see also this equation) formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
                """
                dual = st.selectbox("Choose dual", (True, False), index=1, help=text_dual)
                text_tol = """
                ### tol: float, default=1e-4
                Tolerance for stopping criteria.
                """
                tol = float(st.text_input("Choose tolerance", value=0.0001, help=text_tol))
                text_C = """
                ### C: float, default=1.0
                Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
                """
                C_param = st.number_input("Choose C", value=1.0, help=text_C)
                text_fit_intercept = """
                ### fit_intercept: bool, default=True
                Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
                """
                fit_intercept = st.selectbox("Choose fit_intercept", (True, False), help=text_fit_intercept)
                text_intercept_scaling = """
                ### intercept_scaling: float, default=1
                When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic feature weight.
                """
                intercept_scaling = st.number_input("Choose intercept_scaling", value=1, help=text_intercept_scaling)
                text_class_weight = """
                ### class_weight: dict or ‘balanced’, default=None
                Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies
                """
                class_weight = st.selectbox("Choose class_weight", ('balanced', None), index=1, help=text_class_weight)
                text_random_state = """
                ### random_state: int, RandomState instance or None, default=None
                Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data. See Glossary for details.
                """
                random_state = st.number_input("Choose random_state", value=None, help=text_random_state)
                text_solver = """
                ### solver: {‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
                Algorithm to use in the optimization problem. Default is ‘lbfgs’. To choose a solver, you might want to consider the following aspects:
                - For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;
                - For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
                - ‘liblinear’ and ‘newton-cholesky’ can only handle binary classification by default. To apply a one-versus-rest scheme for the multiclass setting one can wrapt it with the OneVsRestClassifier.
                - ‘newton-cholesky’ is a good choice for n_samples >> n_features, especially with one-hot encoded categorical features with rare categories. Be aware that the memory usage of this solver has a quadratic dependency on n_features because it explicitly computes the Hessian matrix.
                """
                solver = st.selectbox("Choose solver", ('lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga', 'newton-cholesky'), help=text_solver)
                text_max_iter = """
                ### max_iter: int, default=100
                Maximum number of iterations taken for the solvers to converge.
                """
                max_iter = st.number_input("Choose max_iter", value=100, help=text_max_iter)
                text_verbose = """
                ### verbose: int, default=0
                For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
                """
                verbose = st.number_input("Choose verbose", value=0, help=text_verbose)
                text_warm_start = """
                ### warm_start: bool, default=False
                When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
                """
                warm_start = st.selectbox("Choose warm_start", (True, False), index=1, help=text_warm_start)
                text_n_jobs = """
                ### n_jobs: int, default=None
                Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”. This parameter is ignored when the solver is set to ‘liblinear’ regardless of whether ‘multi_class’ is specified or not. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
                """
                n_jobs = st.number_input("Choose n_jobs", value=None, help=text_n_jobs)
                text_l1_ratio = """
                ### l1_ratio: float, default=None
                The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.
                """
                l1_ratio = st.number_input("Choose l1_ratio", value=None, help=text_l1_ratio)

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
                text_n_estimators = """
                ### n_estimators: int, default=100
                The number of trees in the forest.
                """
                n_estimators = st.number_input("Choose n_estimators", value=100, help=text_n_estimators)
                text_criterion = """
                ### criterion: {“gini”, “entropy”, “log_loss”}, default=”gini”
                The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain, see Mathematical formulation. Note: This parameter is tree-specific.
                """
                criterion = st.selectbox("Choose criterion", ('gini', 'entropy', 'log_loss'), help=text_criterion)
                text_max_depth = """
                ### max_depth: int, default=None
                The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                """
                max_depth = st.number_input("Choose max_depth", value=None, help=text_max_depth)
                text_min_samples_split = """
                ### min_samples_split: int or float, default=2
                The minimum number of samples required to split an internal node:
                - If int, then consider min_samples_split as the minimum number.
                - If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
                """
                min_samples_split = st.number_input("Choose min_samples_split", value=2, help=text_min_samples_split)
                text_min_samples_leaf = """
                ### min_samples_leaf: int or float, default=1
                The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
                - If int, then consider min_samples_leaf as the minimum number.
                - If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
                """
                min_samples_leaf = st.number_input("Choose min_samples_leaf", value=1, help=text_min_samples_leaf)
                text_min_weight_fraction_leaf = """
                ### min_weight_fraction_leaf: float, default=0.0
                The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
                """
                min_weight_fraction_leaf = st.number_input("Choose min_weight_fraction_leaf", value=0.0, help=text_min_weight_fraction_leaf)
                text_max_features = """
                ### max_features: {“auto”, “sqrt”, “log2”}, int or float, default=”sqrt”
                The number of features to consider when looking for the best split:
                - If int, then consider max_features features at each split.
                - If float, then max_features is a fraction and max(1, int(max_features * n_features_in_)) features are considered at each split.
                - If “sqrt”, then max_features=sqrt(n_features).
                - If “log2”, then max_features=log2(n_features).
                - If None, then max_features=n_features.
                """
                max_features = st.selectbox("Choose max_features", ('sqrt', 'log2', None), help=text_max_features)
                text_max_leaf_nodes = """
                ### max_leaf_nodes: int, default=None
                Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
                """
                max_leaf_nodes = st.number_input("Choose max_leaf_nodes", value=None, help=text_max_leaf_nodes)
                text_min_impurity_decrease = """
                ### min_impurity_decrease: float, default=0.0
                A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
                """
                min_impurity_decrease = st.number_input("Choose min_impurity_decrease", value=0.0, help=text_min_impurity_decrease)
                text_bootstrap = """
                ### bootstrap: bool, default=True
                Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
                """
                bootstrap = st.selectbox("Choose bootstrap", (True, False), index=0, help=text_bootstrap)
                text_oob_score = """
                ### oob_score: bool or callable, default=False
                Whether to use out-of-bag samples to estimate the generalization score. By default, accuracy_score is used. Provide a callable with signature metric(y_true, y_pred) to use a custom metric. Only available if bootstrap=True.
                """
                oob_score = st.selectbox("Choose oob_score", (True, False), index=1, help=text_oob_score)
                text_n_jobs = """
                ### n_jobs: int, default=None
                The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
                """
                n_jobs = st.number_input("Choose n_jobs", value=None, help=text_n_jobs)
                text_random_state = """
                ### random_state: int, RandomState instance or None, default=None
                Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features).
                """
                random_state = st.number_input("Choose random_state", value=None, help=text_random_state)
                text_verbose = """
                ### verbose: int, default=0
                Controls the verbosity when fitting and predicting.
                """
                verbose = st.number_input("Choose verbose", value=0, help=text_verbose)
                text_warm_start = """
                ### warm_start: bool, default=False
                When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
                """
                warm_start = st.selectbox("Choose warm_start", (True, False), index=1, help=text_warm_start)
                text_ccp_alpha = """
                ### ccp_alpha: non-negative float, default=0.0
                Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed.
                """
                ccp_alpha = st.number_input("Choose ccp_alpha", value=0.0, help=text_ccp_alpha)
                text_max_samples = """
                ### max_samples: int or float, default=None
                If bootstrap is True, the number of samples to draw from X to train each base estimator.
                - If None (default), then draw X.shape[0] samples.
                - If int, then draw max_samples samples.
                - If float, then draw max(round(n_samples * max_samples), 1) samples. Thus, max_samples should be in the interval (0.0, 1.0].
                """
                max_samples = st.number_input("Choose max_samples", value=None, help=text_max_samples)

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

    def gaussian_naivebayes_classification(data):

        X, y, feature_columns = pre_train(data)

        if X is not None and y is not None and feature_columns is not None:

            X_train, X_test, y_train, y_test = train_test(X, y)

            gnb = GaussianNB()
            gnb.fit(X_train, y_train)

            get_result(gnb, X, X_train, X_test, y_test, feature_columns)

    def bernoulli_naivebayes_classification(data):

        X, y, feature_columns = pre_train(data)

        if X is not None and y is not None and feature_columns is not None:

            X_train, X_test, y_train, y_test = train_test(X, y)

            with st.popover("Choose parameter", use_container_width=True):
                text_alpha = """
                ### alpha: float, default=1.0
                Additive (Laplace/Lidstone) smoothing parameter (set alpha=0 and force_alpha=True, for no smoothing).
                """
                alpha = st.number_input("Choose alpha", value=1.0, help=text_alpha)
                text_force_alpha = """
                ### force_alpha: bool, default=True
                If False and alpha is less than 1e-10, it will set alpha to 1e-10. If True, alpha will remain unchanged. This may cause numerical errors if alpha is too close to 0.
                """
                force_alpha = st.selectbox("Choose force_alpha", (True, False), index=0, help=text_force_alpha)
                text_binarize = """
                ### binarize: float or None, default=None
                Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors.
                """
                binarize = st.number_input("Choose binarize", value=None, help=text_binarize)
                text_fit_prior = """
                ### fit_prior: bool, default=True
                Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
                """
                fit_prior = st.selectbox("Choose fit_prior", (True, False), index=1, help=text_fit_prior)
                
            params = {
                'alpha': alpha,
                'binarize': binarize,
                'fit_prior': fit_prior,
                'force_alpha': force_alpha
            }

            bnb = BernoulliNB()
            bnb.set_params(**params)
            bnb.fit(X_train, y_train)

            get_result(bnb, X, X_train, X_test, y_test, feature_columns)

    def multinomial_naivebayes_classification(data):

        X, y, feature_columns = pre_train(data)

        if X is not None and y is not None and feature_columns is not None:

            X_train, X_test, y_train, y_test = train_test(X, y)

            with st.popover("Choose parameter", use_container_width=True):
                text_alpha = """
                ### alpha: float, default=1.0
                Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
                """
                alpha = st.number_input("Choose alpha", value=1.0, help=text_alpha)
                text_force_alpha = """
                ### force_alpha: bool, default=True
                If False and alpha is less than 1e-10, it will set alpha to 1e-10. If True, alpha will remain unchanged. This may cause numerical errors if alpha is too close to 0.
                """
                force_alpha = st.selectbox("Choose force_alpha", (True, False), index=0, help=text_force_alpha)
                text_fit_prior = """
                ### fit_prior: bool, default=True
                Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
                """
                fit_prior = st.selectbox("Choose fit_prior", (True, False), index=1, help=text_fit_prior)
            
            params = {
                'alpha': alpha,
                'fit_prior': fit_prior,
                'force_alpha': force_alpha
            }

            mnb = MultinomialNB()
            mnb.set_params(**params)
            mnb.fit(X_train, y_train)

            get_result(mnb, X, X_train, X_test, y_test, feature_columns)

    def svm_classification(data):
        X, y, feature_columns = pre_train(data)

        if X is not None and y is not None and feature_columns is not None:

            X_train, X_test, y_train, y_test = train_test(X, y)

            with st.popover("Choose parameter", use_container_width=True):
                text_C = """
                ### C: float, default=1.0
                Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
                """
                C = st.number_input("Choose C", value=1.0, help=text_C)
                text_kernel = """
                ### kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
                Specifies the kernel type to be used in the algorithm. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
                """
                kernel = st.selectbox("Choose kernel", ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'), index=2, help=text_kernel)
                text_degree = """
                ### degree: int, default=3
                Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
                """
                degree = st.number_input("Choose degree", value=3, help=text_degree)
                text_gamma = """
                ### gamma: {‘scale’, ‘auto’} or float, default=’scale’
                Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                - if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
                - if ‘auto’, uses 1 / n_features
                - if float, must be non-negative.
                """
                gamma = st.selectbox("Choose gamma", ('scale', 'auto'), help=text_gamma)
                text_coef0 = """
                ### coef0: float, default=0.0
                Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
                """
                coef0 = st.number_input("Choose coef0", value=0.0, help=text_coef0)
                text_shrinking = """
                ### shrinking: bool, default=True
                Whether to use the shrinking heuristic.
                """
                shrinking = st.selectbox("Choose shrinking", (True, False), index=0, help=text_shrinking)
                text_probability = """
                ### probability: bool, default=False
                Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.
                """
                probability = st.selectbox("Choose probability", (True, False), index=1, help=text_probability)
                text_tol = """
                ### tol: float, default=1e-3
                Tolerance for stopping criterion.
                """
                tol = float(st.text_input("Choose tol", value=0.001, help=text_tol))
                text_cache_size = """
                ### cache_size: float, default=200
                Specify the size of the kernel cache (in MB).
                """
                cache_size = st.number_input("Choose cache_size", value=200, help=text_cache_size)
                text_class_weight = """
                ### class_weight: dict or ‘balanced’, default=None
                Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one.
                """
                class_weight = st.selectbox("Choose class_weight", ('balanced', None), index=1, help=text_class_weight)
                text_verbose = """
                ### verbose: bool, default=False
                Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.
                """
                verbose = st.selectbox("Choose verbose", (True, False), index=1, help=text_verbose)
                text_max_iter = """
                ### max_iter: int, default=-1
                Hard limit on iterations within solver, or -1 for no limit.
                """
                max_iter = st.number_input("Choose max_iter", value=-1, help=text_max_iter)
                text_decision_function_shape = """
                ### decision_function_shape: {‘ovo’, ‘ovr’}, default=’ovr’
                Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, note that internally, one-vs-one (‘ovo’) is always used as a multi-class strategy to train models; an ovr matrix is only constructed from the ovo matrix. The parameter is ignored for binary classification.
                """
                decision_function_shape = st.selectbox("Choose decision_function_shape", ('ovo', 'ovr'), index=1, help=text_decision_function_shape)
                text_break_ties = """
                ### break_ties: bool, default=False
                If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties according to confidence values of decision_function; otherwise the first class among the tied classes is returned. Please note that breaking ties comes at a relatively high computational cost compared to a simple predict.
                """
                break_ties = st.selectbox("Choose break_ties", (True, False), index=1, help=text_break_ties)
                text_random_state = """
                ### random_state: int, RandomState instance or None, default=None
                Controls the pseudo random number generation for shuffling the data for probability estimates.
                """
                random_state = st.number_input("Choose random_state", value=None, help=text_random_state)

            params = {
                'C': C,
                'kernel': kernel,
                'degree': degree,
                'gamma': gamma,
                'coef0': coef0,
                'shrinking': shrinking,
                'probability': probability,
                'tol': tol,
                'cache_size': cache_size,
                'class_weight': class_weight,
                'verbose': verbose,
                'max_iter': max_iter,
                'decision_function_shape': decision_function_shape,
                'break_ties': break_ties,
                'random_state': random_state
            }
            svc = SVC()
            svc.set_params(**params)
            svc.fit(X_train, y_train)

            get_result(svc, X, X_train, X_test, y_test, feature_columns)

    def run(data):
        content_image = Image.open("image/content4.png")
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(" # Classification # ")
            st.markdown("""<p>Provide many different classification methods</p>""", unsafe_allow_html=True)
        with col2:
            st.image(content_image)
        st.markdown("---")
        st.write("#### Your data ####")
        with st.expander("See data", expanded=True):
            edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
        st.markdown("---")
        class_type = st.selectbox("", ["KNN", 'Logistic Regression', 'Random Forest', 'Gaussian Naive Bayes', 'Bernoulli Naive Bayes', 'Multinomial Naive Bayes','SVM'])
        if class_type == 'KNN':
            Classification.knn_classification(edit_data)
        if class_type == 'Logistic Regression':
            Classification.lgreg_classification(edit_data)
        if class_type == 'Random Forest':
            Classification.randomfor_classification(edit_data)
        if class_type == 'Gaussian Naive Bayes':
            Classification.gaussian_naivebayes_classification(edit_data)
        if class_type == 'Bernoulli Naive Bayes':
            Classification.bernoulli_naivebayes_classification(edit_data)
        if class_type == 'Multinomial Naive Bayes':
            Classification.multinomial_naivebayes_classification(edit_data)
        if class_type == 'SVM':
            Classification.svm_classification(edit_data)
