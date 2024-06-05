import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import feature_selection
from statsmodels.stats.anova import anova_lm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PIL import Image


def get_result(model, X, X_train, X_test, y_train, y_test, feature_columns, target_column):
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = ((y_pred - y_test) ** 2).mean()
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Display the results
    st.markdown("## Regression Results ##")
    st.markdown("Number of Cases: {}".format(X_train.shape[0]))

    col1, col2, col3 = st.columns(3)

    # Coefficients
    with col1:
        st.markdown("Coefficients")
        st.dataframe(pd.DataFrame({"Coefficients": [model.intercept_, model.coef_[0]]}))

    # Residuals Statistics\
    with col2:
        # ANOVA
        st.markdown("ANOVA")
        f_values, p_values = feature_selection.f_classif(X_train, y_train)
        p_values_round = []
        for i in p_values:
            p_values_round.append("{:.3e}".format(i))
        st.dataframe(
            pd.DataFrame({"Columns": feature_columns, "F_value": f_values, "p_values": p_values_round}).set_index(
                "Columns"))
    with col3:
        st.markdown("Model Summary")
        st.dataframe(pd.DataFrame({"MSE": [mse], "RMSE": [rmse], "R-squared": [r2]}))

    # Plot the predicted values vs. the actual values
    if len(feature_columns) == 1:
        fig = px.scatter(x=X_train.squeeze(), y=y_train, trendline="ols", labels={
            "x": feature_columns[0],
            "y": target_column
        }, trendline_color_override="red")
        st.plotly_chart(fig, theme=None, use_container_width=True)

    # Print the equation of the linear regression model
    equation = r"y = {:.2f}".format(model.intercept_)
    for i, coef in enumerate(model.coef_):
        equation += r" + {:.2f}x_{}".format(coef, i)
    st.markdown(f"The equation is : <span style='color: green'></span>", unsafe_allow_html=True)
    st.latex(equation)

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
        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        # Make a prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.write('The predicted label is: ', prediction[0])

def pre_train(data):
    data_copy = data.copy()
    data_number_colum = data_copy.select_dtypes(include=["int", "float"]).columns

    # Select the target variable
    target_column = st.selectbox("Chọn biến mục tiêu", data_number_colum)

    # Select feature variables
    feature_columns = st.multiselect("Chọn biến tính năng", data_number_colum, default=[target_column])

    if len(feature_columns) == 0:
        st.error("Please select at least one feature variable.")
        return None, None, None, None
    
    scaler_type = st.selectbox('Chọn kiểu scale',('None', 'Standard Scaler', 'Min-max Scaler'))
    # Split the data into training and testing sets
    X = data_copy[feature_columns]
    standardScaler = StandardScaler()
    minMaxScaler = MinMaxScaler()
    if scaler_type == 'None':
            X = X
    elif scaler_type == 'Standard Scaler':
            X = standardScaler.fit_transform(X)
    elif scaler_type == 'Min-max Scaler':
            X = minMaxScaler.fit_transform(X)
    y = data_copy[target_column]

    return X, y, feature_columns, target_column

def train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
class Regression:
    def simple_linear_regresstion(data):
        X, y, feature_columns, target_column = pre_train(data)

        if X is not None and y is not None:
            X_train, X_test, y_train, y_test = train_test(X, y)

            with st.popover("Choose parameter", use_container_width=True):
                text_fit_intercept = """
                ### fit_intercept: bool, default=True
                Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
                """
                fit_intercept = st.selectbox("Choose fit_intercept", (True, False), index=0, help=text_fit_intercept)
                text_copy_X = """
                ### copy_X: bool, default=True
                If True, X will be copied; else, it may be overwritten.
                """
                copy_X = st.selectbox("Choose copy_X", (True, False), index=0, help=text_copy_X)
                text_n_jobs = """
                ### n_jobs: int, default=None
                The number of jobs to use for the computation. This will only provide speedup in case of sufficiently large problems, that is if firstly n_targets > 1 and secondly X is sparse or if positive is set to True. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
                """
                n_jobs = st.number_input("Choose n_jobs", value=None, help=text_n_jobs)
                text_positive = """
                ### positive: bool, default=False
                When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.
                """
                positive = st.selectbox("Choose positive", (True, False), index=1, help=text_positive)

            # Create and train the linear regression model
            params = {
                "fit_intercept": fit_intercept,
                "copy_X": copy_X,
                "n_jobs": n_jobs,
                "positive": positive
            }
            model = LinearRegression()
            model.set_params(**params)
            model.fit(X_train, y_train)

            get_result(model, X, X_train, X_test, y_train, y_test, feature_columns, target_column)

    def ridge_regression(data):
        X, y, feature_columns, target_column = pre_train(data)

        if X is not None and y is not None:
            X_train, X_test, y_train, y_test = train_test(X, y)

            with st.popover("Choose parameter", use_container_width=True):
                text_alpha = """
                ### alpha: float, default=1.0
                Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization.
                """
                alpha = st.number_input("Choose alpha", value=1.0, help=text_alpha)
                text_fit_intercept = """
                ### fit_intercept: bool, default=True
                Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
                """
                fit_intercept = st.selectbox("Choose fit_intercept", (True, False), index=0, help=text_fit_intercept)
                text_copy_X = """
                ### copy_X: bool, default=True
                If True, X will be copied; else, it may be overwritten.
                """
                copy_X = st.selectbox("Choose copy_X", (True, False), index=0, help=text_copy_X)
                text_max_iter = """
                ### max_iter: int, default=None
                Maximum number of iterations for conjugate gradient solver. For 'sparse_cg' and 'lsqr' solvers, the default value is determined by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.
                """
                max_iter = st.number_input("Choose max_iter", value=None, help=text_max_iter)
                text_tol = """
                ### tol: float, default=1e-4
                The precision of the solution (coef_) is determined by tol which specifies a different convergence criterion for each solver:
                - ‘svd’: tol has no impact.
                - ‘cholesky’: tol has no impact.
                - ‘sparse_cg’: norm of residuals smaller than tol.
                - ‘lsqr’: tol is set as atol and btol of scipy.sparse.linalg.lsqr, which control the norm of the residual vector in terms of the norms of matrix and coefficients.
                - ‘sag’ and ‘saga’: relative change of coef smaller than tol.
                - ‘lbfgs’: maximum of the absolute (projected) gradient=max|residuals| smaller than tol.
                """
                tol = float(st.text_input("Choose tolerance", value=0.0001, help=text_tol))
                text_solver = """
                ### solver: {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’, ‘lbfgs’}, default=’auto’
                Solver to use in the computational routines:
                - ‘auto’ chooses the solver automatically based on the type of data.
                - ‘svd’ uses a Singular Value Decomposition of X to compute the Ridge coefficients. More stable for singular matrices than ‘cholesky’.
                - ‘cholesky’ uses the standard scipy.linalg.solve function to obtain a closed-form solution.
                - ‘sparse_cg’ uses the conjugate gradient solver as found in scipy.sparse.linalg.cg. As an iterative algorithm, this solver is more appropriate than ‘cholesky’ for large-scale data (possibility to set tol and max_iter).
                - ‘lsqr’ uses the dedicated regularized least-squares routine scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative procedure.
                - ‘sag’ uses a Stochastic Average Gradient descent, and ‘saga’ uses its improved, unbiased version named SAGA. Both methods also use an iterative procedure, and are often faster than other solvers when both n_samples and n_features are large. Note, however, that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.
                - ‘lbfgs’ uses L-BFGS-B optimization. It is quite efficient for small problems but not for large problems.
                """
                solver = st.selectbox("Choose solver", ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'), index=0, help=text_solver)
                text_positive = """
                ### positive: bool, default=False
                When set to True, forces the coefficients to be positive. Only ‘lbfgs’ solver is supported in this case.
                """
                positive = st.selectbox("Choose positive", (True, False), index=1, help=text_positive)
                text_random_state = """
                ### random_state: int, RandomState instance or None, default=None
                Used when solver == ‘sag’ or ‘saga’ to shuffle the data.
                """
                random_state = st.number_input("Choose random_state", value=None, help=text_random_state)

            params = {
                "alpha": alpha,
                "fit_intercept": fit_intercept,
                "copy_X": copy_X,
                "max_iter": max_iter,
                "tol": tol,
                "solver": solver,
                "random_state": random_state,
                "positive": positive
            }
            model = Ridge()
            model.set_params(**params)
            model.fit(X_train, y_train)

            get_result(model, X, X_train, X_test, y_train, y_test, feature_columns, target_column)

    def lasso_regression(data):
        X, y, feature_columns, target_column = pre_train(data)

        if X is not None and y is not None:
            X_train, X_test, y_train, y_test = train_test(X, y)

            with st.popover("Choose parameter", use_container_width=True):
                text_alpha = """
                ### alpha: float, default=1.0
                Constant that multiplies the L1 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).
                """
                alpha = st.number_input("Choose alpha", value=1.0, help=text_alpha)
                text_fit_intercept = """
                ### fit_intercept: bool, default=True
                Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
                """
                fit_intercept = st.selectbox("Choose fit_intercept", (True, False), index=0, help=text_fit_intercept)
                text_precompute = """
                ### precompute: bool or array-like of shape (n_features, n_features), default=False
                Whether to use a precomputed Gram matrix to speed up calculations. The Gram matrix can also be passed as argument. For sparse input this option is always True to preserve sparsity.
                """
                precompute = st.selectbox("Choose precompute", (True, False), index=1, help=text_precompute)
                text_copy_X = """
                ### copy_X: bool, default=True
                If True, X will be copied; else, it may be overwritten.
                """
                copy_X = st.selectbox("Choose copy_X", (True, False), index=0, help=text_copy_X)
                text_max_iter = """
                ### max_iter: int, default=1000
                The maximum number of iterations.
                """
                max_iter = st.number_input("Choose max_iter", value=1000, help=text_max_iter)
                text_tol = """
                ### tol: float, default=1e-4
                The tolerance for the optimization: if the updates are smaller than tol, the optimization code checks the dual gap for optimality and continues until it is smaller than tol.
                """
                tol = float(st.text_input("Choose tolerance", value=0.0001, help=text_tol))
                text_warm_start = """
                ### warm_start: bool, default=False
                When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
                """
                warm_start = st.selectbox("Choose warm_start", (True, False), index=0, help=text_warm_start)
                text_positive = """
                ### positive: bool, default=False
                When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.
                """
                positive = st.selectbox("Choose positive", (True, False), index=1, help=text_positive)
                text_random_state = """
                ### random_state: int, RandomState instance or None, default=None
                Used when selection == ‘random’ to shuffle the data. See Glossary for details.
                """
                random_state = st.number_input("Choose random_state", value=None, help=text_random_state)
                text_selection = """
                ### selection: {‘cyclic’, ‘random’}, default=’cyclic’
                If set to ‘random’, a random coefficient is updated every iteration rather than looping over features sequentially by default. This (setting to ‘random’) often leads to significantly faster convergence especially when tol is higher than 1e-4.
                """
                selection = st.selectbox("Choose selection", ('cyclic', 'random'), index=0, help=text_selection)

            params = {
                "alpha": alpha,
                "fit_intercept": fit_intercept,
                "precompute": precompute,
                "copy_X": copy_X,
                "max_iter": max_iter,
                "tol": tol,
                "warm_start": warm_start,
                "positive": positive,
                "random_state": random_state,
                "selection": selection
            }
            model = Lasso()
            model.set_params(**params)
            model.fit(X_train, y_train)

            get_result(model, X, X_train, X_test, y_train, y_test, feature_columns, target_column)

    def run(data):
        content_image = Image.open("image/content3.png")
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(" # Regression # ")
            st.markdown("""<p>Provide a variety of linear models for regresstion</p>""", unsafe_allow_html=True)
        with col2:
            st.image(content_image)
        st.markdown("---")
        st.write("#### Your data ####")
        with st.expander("See data", expanded=True):
            edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
        st.markdown("---")
        regression_type = st.selectbox("", ["OLS Linear Regression", 'Ridge', 'Lasso'])
        if regression_type == "OLS Linear Regression":
            Regression.simple_linear_regresstion(data)
        if regression_type == "Ridge":
            Regression.ridge_regression(data)
        if regression_type == "Lasso":
            Regression.lasso_regression(data)