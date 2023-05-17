### Abstract
This project consists of a series of notebooks that were used to test various unsuperivsed clustering models on NICER data. Sadly, none of the models tested proved effective. Part of the reason for this was a lack of sufficient computing power to thoroughly test the models.
### Dataset and Tools
The dataset was extracted from [SciServer](https://sciserver.org/) using the `heasoft` image. The code used for the collection of the data is availible in the `data.ipynb` notebook. You can also just use the `spectra.csv` dataset directly as is done in the other notebooks.

The `helpers.py` file is also needed for the `data` notebook to run.
### Notebooks
- `agglom-cluster-job` Created and plotted a agglomerative clustering model, the algorithm is too slow to scale to our data.
- `dbscan-cluster-job` Created and plotted a DBSCAN model. Used for comparison with HDBSCAN.
- `hdbscan-cluster-job` Contains both code for testing the hdbscan and also has some commented out chunks for testing various different distance computation methods.
- `DistanceCreator` Due to limitations in compute job time, this notebook was created to compute pairwisde distances in batches for use in the HDBSCAN model.
- `kmeans-cluster-job` Tested different parameters for a kmeans model and plotted results.
- `optics-cluster-job` Fitted different parameters for an OPTICS model and plotted results.