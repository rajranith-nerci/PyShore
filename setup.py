from setuptools import setup, find_packages

setup(
    name="pyshore",
    version="1.0.0",
    description="Automated shoreline change analysis using Google Earth Engine",
    author="PyShore",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "geopandas>=0.13",
        "shapely>=2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "statsmodels>=0.14",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "earthengine-api>=0.1.370",
        "geemap>=0.29",
    ],
    extras_require={
        "dev": ["pytest", "jupyter"],
    },
)
