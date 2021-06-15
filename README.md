raman_spectrogram_analysis
==============================

Clustering and classification of Raman spectrograms.

My initial idea was to perform the analysis directly on the normalised spectrograms. This turned out to be not feasible due to the different sample length (k-means dooesn't support different input shapes); To deal with this problem, I used polynomial and linear interpolation to normalize the dimensionality of the data. In fact, I fitted a *n*-order polynomial to the data and clustered the samples based on the mapping of the range $`[0, 2400]`$ through the polynomial.

The results for this clustering are shown in the plot below:
*clustering results*

On a second thought I wanted to investigate the number and position of peaks, a meaningul feature when it comes to spectrogram analysis. I tried two different approaches for this:
- highest peak [position on the x axis](notebooks/fp-model-with-peak1-argmax.ipynb) and [position on the x axis with height](notebooks/fp-model-with-maxpeak2d.ipynb);
- same analysis for 5 peaks \[[argmax](notebooks/fp-model-5-maxpeak-argmax.ipynb)\] \[[argmax + height](notebooks/fp-model-5-maxpeak-2d.ipynb)\] and 10 peaks \[[argmax](notebooks/fp-model-10-maxpeak-argmax.ipynb)\];

Despite this seemed to be a good fit for the type of data to be processed, the elbow analysis on the clustering inertia was suggesting that the number of distinct clusters to classify was at max around 10, thus the feature set was not enough to represent the data for our purpose.

After some experimentations and a bit of research I referred to the field of audio signal analysis and in particular to the paper ["Classification of audio signals using statistical features on time and wavelet transform domains." by Lambrou et al.](references/papers/ic982120.pdf) to create a set of meaningul scalar features that could represent the Raman spectrograms for clustering purposes.
This achieved a good representative power and was performing well enough to be presented in this report. In fact, *inference results*

**NB** the number of clusters from the unlabeled samples clustering gave results which are different from what I was expecting, probably due to the fact that not all of the materials listed with names are actually present in the acquisitions S1, S2.

In order to associate labels to different clusters, I took the argmin with respect to all clusters in the list.
Despite this not be the results I expected (there is not a 1 to 1 correspondance between clusters and labeled samples), the results are reported for the sake of evaluation.

### Important notes
From the EDA notebooks [1](notebooks/fp-eda-S1-raman-data.ipynb) [2](notebooks/fp-eda-S2-raman-data.ipynb) [3](notebooks/README-consegne.pynb), I noticed a big gap in integral values between labeled and unlabeled samples (which is experimentally related to the duration of the acquisition), thus I decided to normalise the area under the spectrogram in order to achieve comparable feature sets.

## How to use the repo

The project is based on a cookiecutter template. In particular, this enable you to reproduce results in a fairly simply way.

Clone the repo with 

`git clone https://github.com/fpaissan/raman_spectrograms_analysis.git`

than enter the directory and follow the instructions below based on what you want to do.

### Preprocess data
Run `make data`.

### Train k-means/k-medoids
In the Makefile, change the argument at line *insert line* to chose one of the two methods, than run `make train`.

### Predict abbundances in the data
Run `make predict`.

### In order to visualise abbundance of materials and possibly other interesting plots
Have a look at the visualisations file and modify things if you want to have more insightful plots.
For the plots presented in this report run `make visualization`.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

