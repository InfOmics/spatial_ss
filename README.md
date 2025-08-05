# spatial_ss
Spatial siloutte score for spatial transcriptomics

Extending the standard Silhouette score with penalty to account for spatial contiguity

Create an environment if necessary
`````
conda create -n sss python==3.10.0

conda activate sss

`````
install requirements
````
pip install -r requirements.txt

````
install SSS package
````
pip install git+https://github.com/InfOmics/spatial_ss.git
````
run_SSS.py contains the SSS test pipeline. 


### Data Availability ###
The spatial transcriptomics datasets are available at:  https://doi.org/10.5281/zenodo.15277298