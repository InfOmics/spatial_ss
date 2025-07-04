from Silhouette_Spatial_Score.silhouette_spatial import silhouette_spatial_score
from scipy.sparse import issparse
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
def pca (X, n_components=20,random_state=35):
    pca = PCA(n_components, random_state=random_state) 
    return pca.fit_transform(X)

def norm_data(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
def hvg (adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    return adata.var['highly_variable']

def read_adata(path, is_h5ad=False):
    #Todo: add check if Visium, if 'filtered_feature_bc_matrix.h5' or 'data_name_filtered_feature_bc_matrix.h5'
    if is_h5ad:
        adata = sc.read_h5ad(path)
        adata.var_names_make_unique()
        adata.obsm["spatial"]=adata.obsm["spatial"].astype(float)
    else: 
        adata = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        adata.obsm["spatial"]=adata.obsm["spatial"].astype(float)
    return adata



def preprocess(adata, n_components=20,random_seed=35):

    if 'highly_variable' not in adata.var:
        hvg_genes=hvg(adata)
        norm_data(adata)
        adata= adata[:,hvg_genes]

    if issparse(adata.X):
            data=pca ( adata.X.toarray(), n_components=20,random_state=random_seed) 
    else:
            data=pca ( adata.X, n_components=20,random_state=random_seed) #if data already a matrix

    adata.obsm["X_pca"]=data
    return adata


def run_SSS(
    path='Data/DLPFC/151673',
    is_h5ad=False,
    n_components=20,
    random_seed=35,
    is_visium=True,
    use_mock_clusters=True,
    n_clusters=6
):
    """
    Run Silhouette Spatial Score on spatial transcriptomics data.

    Parameters
    ----------
    path : str
        Path to dataset (folder or .h5ad file).
    is_h5ad : bool
        Whether the file is a preprocessed .h5ad.
    n_components : int
        Number of PCA components.
    random_seed : int
        Random seed for reproducibility.
    is_visium : bool
        Whether the input is Visium data (affects spatial penalty logic).
    use_mock_clusters : bool
        Whether to assign random clusters.
    n_clusters : int
        Number of clusters if using random assignment.

    Returns
    -------
    AnnData
        Processed object with SSS stored in `adata.uns['SSS']`.
    """
    adata = read_adata(path, is_h5ad=is_h5ad)
    adata = preprocess(adata, n_components=n_components, random_seed=random_seed)

    # Assign clusters (can be replaced with Leiden/Louvain/mclust etc.)
    if use_mock_clusters:
        adata.obs["cluster"] = np.random.randint(0, n_clusters, size=adata.shape[0])

    # Run SSS computation
    silhouette_spatial = silhouette_spatial_score(
        adata.obsm["X_pca"],
        adata.obs["cluster"],
        adata,
        metric="cosine",
        is_visium=is_visium
    )
    adata.uns["SSS"] = silhouette_spatial

    print("Silhouette Spatial Score (SSS):", np.round(silhouette_spatial, 4))

    if "average_penalty" in adata.uns:
        penalty = adata.uns["average_penalty"]
        print("SSS average penalty:", np.round(penalty, 4))
    else:
        print("Warning: average_penalty not found in `adata.uns`.")

    return adata

run_SSS( n_components=20, random_seed=35, is_visium=True)