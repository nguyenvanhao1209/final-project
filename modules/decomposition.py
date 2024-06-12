import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA, NMF, FastICA
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

class Decomposition:
    def pca_decomposition(data: pd.DataFrame):
        
        number_columns = data.select_dtypes(include=[float, int])
        with st.popover("Choose parameter", use_container_width=True):
            text_n_components = """
                ### n_components: int, float or ‘mle’, default=None
                Number of components to keep."""
            n_components = st.slider("Number of PCA components", 1, number_columns.shape[1], help=text_n_components)
            text_svd_solver = """
                ### svd_solver: {‘auto’, ‘full’, ‘covariance_eigh’, ‘arpack’, ‘randomized’}, default=’auto’
            """
            svd_solver = st.selectbox("Choose svd solver", ("auto", "full","arpack","randomized"), help=text_svd_solver)
        
        target_columns = st.multiselect("Select target columns", data.select_dtypes(exclude=['number']).columns)
        params = {
            "n_components": n_components,
            "svd_solver": svd_solver,
        }
        pca = PCA()
        pca.set_params(**params)
        components = pca.fit_transform(number_columns)
        explained_variance = pca.explained_variance_ratio_
        col1, col2, col3 = st.columns([2,1,1])
        coef_matrix = number_columns.corr()
        eigenvalues, _ = np.linalg.eig(coef_matrix)
        with col1:
            fig = go.Figure(data=go.Heatmap(
                   z=coef_matrix,
                   x=number_columns.columns,  # Labels for x-axis
                   y=number_columns.columns,  # Labels for y-axis
                   colorscale='burgyl',
                   ))

            # Add title and labels
            fig.update_layout(
                title='Correlation Matrix Heatmap',
                xaxis_nticks=36,
            )
            
            st.plotly_chart(fig,theme = None, use_container_width=True)
            eigen_df = pd.DataFrame(data=eigenvalues, columns=['Eigenvalue'])
            explained_df = pd.DataFrame(data=explained_variance, index=[f'PC{i+1} Explain' for i in range(n_components)], columns=['Explained variance ratio'])
            
            
            
        
        # Plotting the PCA result
        with col3:
            pca_df = pd.DataFrame(data=components, columns=[f'PC{i+1}' for i in range(n_components)])
            for i in range(len(target_columns)):
                pca_df[f'{target_columns[i]}'] = data[f'{target_columns[i]}']    
            st.dataframe(pca_df, use_container_width=True)
            
        
        with col2:
            st.dataframe(eigen_df, use_container_width=True)
            st.dataframe(explained_df, use_container_width=True)
            
            
    def nmf_decomposition(data: pd.DataFrame):
        number_columns = data.select_dtypes(include=[float, int])
        with st.popover("Choose parameter", use_container_width=True):
            text_n_components = """
                ### n_components: int, float or ‘mle’, default=None
                Number of components to keep."""
            n_components = st.slider("Number of NMF components", 1, number_columns.shape[1], help=text_n_components)
        nmf = NMF(n_components=n_components, init='random', random_state=0)
        target_columns = st.multiselect("Select target columns", data.select_dtypes(exclude=['number']).columns)
        components = nmf.fit_transform(number_columns)
        H = nmf.components_
        nmf_df = pd.DataFrame(data=components, columns=[f'C{i+1}' for i in range(n_components)])
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            fig = go.Figure(data=go.Heatmap(
                   z=H,
                   x=number_columns.columns,  # Labels for x-axis
                   y=number_columns.columns,  # Labels for y-axis
                   colorscale='burgyl',
                   ))

            # Add title and labels
            fig.update_layout(
                title='Correlation Matrix Heatmap',
                xaxis_nticks=36,
            )
            st.plotly_chart(fig,theme = None, use_container_width=True)
            
        
        with col2:
            reconstructed_data = np.dot(components, H)
            reconstructed_df = pd.DataFrame(reconstructed_data, columns=number_columns.columns)
            
            for i in range(len(target_columns)):
                reconstructed_df[f'{target_columns[i]}'] = data[f'{target_columns[i]}']
                
            st.dataframe(reconstructed_df, use_container_width=True)
            reconstruction_error = nmf.reconstruction_err_
            st.write(f"Reconstruction Error: {reconstruction_error:.4f}")
        
        with col3:
            for i in range(len(target_columns)):
                nmf_df[f'{target_columns[i]}'] = data[f'{target_columns[i]}']      
            st.dataframe(nmf_df, use_container_width=True)
        
    def ica_decomposition(data: pd.DataFrame):
        number_columns = data.select_dtypes(include=[float, int])
        with st.popover("Choose parameter", use_container_width=True):
            n_components = st.slider("Number of ICA components", 1, number_columns.shape[1])
        ica = FastICA(n_components=n_components, random_state=0)
        target_columns = st.multiselect("Select target columns", data.select_dtypes(exclude=['number']).columns)
        components = ica.fit_transform(number_columns)
        ica_df = pd.DataFrame(data=components, columns=[f'IC{i+1}' for i in range(n_components)])
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            fig = go.Figure(data=go.Heatmap(
                   z=ica.mixing_,
                   x=number_columns.columns,  # Labels for x-axis
                   y=number_columns.columns,  # Labels for y-axis
                   colorscale='burgyl',
                   ))

            # Add title and labels
            fig.update_layout(
                title='Correlation Matrix Heatmap',
                xaxis_nticks=36,
            )
            st.plotly_chart(fig,theme = None, use_container_width=True)
            
        
        with col2:
            reconstructed_data = np.dot(components, ica.mixing_.T) + ica.mean_
            reconstructed_df = pd.DataFrame(reconstructed_data, columns=number_columns.columns)
            
            for i in range(len(target_columns)):
                reconstructed_df[f'{target_columns[i]}'] = data[f'{target_columns[i]}']
                
            st.dataframe(reconstructed_df, use_container_width=True)

        
        with col3:
            for i in range(len(target_columns)):
                ica_df[f'{target_columns[i]}'] = data[f'{target_columns[i]}']      
            st.dataframe(ica_df, use_container_width=True)
        
    def run(data: pd.DataFrame):
        content_image = Image.open("image/content8.png")
        col1, col2 = st.columns([4,1])
        with col1:
            st.markdown(" # Decomposition # ")
            st.markdown("""<p>Provide a variety of linear models for regresstion</p>""", unsafe_allow_html=True)
        with col2:
            st.image(content_image)
        st.markdown("---")
        st.write("#### Your data ####")
        with st.expander("See data", expanded=True):
            edit_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")
        st.markdown("---")
        decomposition_type = st.selectbox("", ["PCA", 'NMF', 'ICA'])
        if decomposition_type == "PCA":
            Decomposition.pca_decomposition(edit_data)
        if decomposition_type == "NMF":
            Decomposition.nmf_decomposition(edit_data)
        if decomposition_type == "ICA":
            Decomposition.ica_decomposition(edit_data)