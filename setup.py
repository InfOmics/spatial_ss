from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name='Silhouette_Spatial_Score',
    version='0.1.0',
    author='Gospel Ozioma Nnadi',
    description='Extending the standard Silhouette score with penalty to account for spatial contiguity',
    author_email ='gospelozioma.nnadi@univr.it',
    license = 'MIT',
    packages =find_packages(), 
    install_requires=requirements,
    python_requires='>=3.10',
    zip_safe = False,
    include_package_data = True,
    url='https://github.com/InfOmics/spatial_ss',
)