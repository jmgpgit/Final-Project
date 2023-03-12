# Standard imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import scipy.stats as stats
import networkx as nx
import os
import sys
import json
import random
import numpy.random as npr
import re
from math import ceil, floor, log, log2, log10, sqrt, exp, factorial, gcd, lcm, pi, e, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2, pow, atan, asin, acos, tan, sin, cos, sinh, cosh, tanh, asinh, acosh, atanh, degrees, radians, expm1, log1p, exp2, log2, fsum, gcd, lcm, tau, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2, pow, atan, asin, acos, tan, sin, cos, sinh, cosh, tanh, asinh, acosh, atanh, degrees, radians, expm1, log1p, exp2, log2, fsum, gcd, lcm, tau, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2, pow, atan, asin, acos, tan, sin, cos, sinh, cosh, tanh, asinh, acosh, atanh, degrees, radians, expm1, log1p, exp2, log2, fsum, gcd, lcm, tau, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2, pow, atan, asin, acos, tan, sin, cos, sinh, cosh, tanh, asinh, acosh, atanh, degrees, radians, expm1, log1p, exp2, log2, fsum, gcd, lcm, tau, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2
from collections import Counter, defaultdict, OrderedDict, namedtuple, deque
from functools import partial, partialmethod, reduce, wraps, cache, lru_cache, cached_property, singledispatch, singledispatchmethod
from itertools import count, cycle, product as cartesian_product, permutations, combinations, combinations_with_replacement, accumulate, starmap
from tqdm import tqdm # from tqdm.notebook import tqdm
from uuid import uuid4
from datetime import datetime, timedelta
from time import time, sleep
from toolz import memoize, curry, diff, unique, valmap, valfilter, itemmap, itemfilter, keymap, keyfilter, merge_sorted, interleave, isdistinct, diff, peek, peekn, countby, juxt, excepts, merge, merge_with, assoc, dissoc
from more_itertools import unzip, chunked, chunked_even, minmax, filter_except, numeric_range, make_decorator,replace, locate,countable,unique_everseen, always_iterable,unique_justseen,map_except,count_cycle, mark_ends, sample, distribute, bucket, peekable, seekable,spy,transpose, sieve,polynomial_from_roots,flatten, intersperse, partition, powerset, collapse, split_at, flatten,split_before, split_after, split_when, take

# sys.path.append(os.path.relpath("../../src/"))
# from * import * # my classes and functions
















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # %matplotlib inline
import seaborn as sns
import scipy.stats as stats
# from scipy.stats import bernoulli, binom, nbinom, geom, poisson, randint  # discrete distributions
# from scipy.stats import uniform, norm, beta, gamma, expon, lognorm, t, chi2, chi2_contingency, f, f_oneway , weibull_min, weibull_max, pareto  # continuous distributions
# from scipy.stats import skew, kurtosis, skewtest, kurtosistest, normaltest, shapiro, anderson, jarque_bera, boxcox, boxcox_normmax, probplot, moment, ttest_ind, ttest_rel, ttest_1samp, mannwhitneyu, wilcoxon, kruskal, friedmanchisquare, f_oneway, chi2_contingency, levene, bartlett, pearsonr, spearmanr, kendalltau, pointbiserialr, linregress, zscore, rankdata, ranksums, ks_2samp, kstest, chisquare, sem, tme
import statsmodels.api as sm
import statsmodels.stats as sms
import statsmodels.stats.api as smsa
#from statsmodels.proportion import proportion_confint, multinomial_proportions_confint, proportions_ztest
#from statsmodels.formula.api import ols
#from statsmodels.stats.anova import anova_lm
#from statsmodels.stats.manova import MANOVA
#from statsmodels.stats.weightstats import ztest, ttest_ind, ttest_1samp, ttest_rel, DescrStatsW, CompareMeans, CompareMeans, CompareMe
#from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
#from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence, summary_table
#from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import os
import sys
import json
import random
import re
#import warnings #warnings.filterwarnings('ignore')
#import zipfile
#import csv

from math import ceil, floor, log, log2, log10, sqrt, exp, factorial, gcd, lcm, pi, e, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2, pow, atan, asin, acos, tan, sin, cos, sinh, cosh, tanh, asinh, acosh, atanh, degrees, radians, expm1, log1p, exp2, log2, fsum, gcd, lcm, tau, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2, pow, atan, asin, acos, tan, sin, cos, sinh, cosh, tanh, asinh, acosh, atanh, degrees, radians, expm1, log1p, exp2, log2, fsum, gcd, lcm, tau, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2, pow, atan, asin, acos, tan, sin, cos, sinh, cosh, tanh, asinh, acosh, atanh, degrees, radians, expm1, log1p, exp2, log2, fsum, gcd, lcm, tau, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2, pow, atan, asin, acos, tan, sin, cos, sinh, cosh, tanh, asinh, acosh, atanh, degrees, radians, expm1, log1p, exp2, log2, fsum, gcd, lcm, tau, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2
from collections import Counter, defaultdict, OrderedDict, namedtuple, deque
from functools import partial, partialmethod, reduce, wraps, cache, lru_cache, cached_property, singledispatch, singledispatchmethod
from itertools import count, cycle, product as cartesian_product, permutations, combinations, combinations_with_replacement, accumulate, starmap
from tqdm import tqdm # from tqdm.notebook import tqdm
from uuid import uuid4
from datetime import datetime, timedelta
from time import time, sleep
from toolz import memoize, curry, diff, unique, valmap, valfilter, itemmap, itemfilter, keymap, keyfilter, merge_sorted, interleave, isdistinct, diff, peek, peekn, countby, juxt, excepts, merge, merge_with, assoc, dissoc
from more_itertools import unzip, chunked, chunked_even, minmax, filter_except, numeric_range, make_decorator,replace, locate,countable,unique_everseen, always_iterable,unique_justseen,map_except,count_cycle, mark_ends, sample, distribute, bucket, peekable, seekable,spy,transpose, sieve,polynomial_from_roots,flatten, intersperse, partition, powerset, collapse, split_at, flatten,split_before, split_after, split_when, take

# sys.path.append(os.path.relpath("../../src/"))
# from * import * # my classes and functions

from IPython.display import display, HTML, Markdown, Latex, Image, SVG, YouTubeVideo, Audio, Javascript, IFrame, clear_output
import IPython
import ipywidgets as widgets
import import_ipynb

import scipy



# Databases
import mysql.connector as conn
# conection = conn.connect(host='localhost',
#                         user='root',
#                         passwd='password')  

# cursor = conection.cursor() : cursor.execute('CREATE DATABASE mydb')
from sqlalchemy import create_engine
# str_conn = 'mysql+pymysql://root:password@localhost:3306' # mysql
# cursor = create_engine(str_conn)
# cursor.execute('drop database if exists app;')
# str_conn = 'postgresql+psycopg2://iudh:password@localhost:5432/apps' # postgres
from sqlalchemy import create_engine, Column, Float, Integer, JSON, DateTime, Text, DDL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


from pymongo import MongoClient
# str_conn='mongodb://localhost:27017'
# cursor=MongoClient(str_conn)
import folium


# Scraping
import requests as req
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from webdriver_manager.firefox import GeckoDriverManager

PATH=FirefoxService(GeckoDriverManager().install())
binary = FirefoxBinary(r'C:\Program Files\Mozilla Firefox\Firefox.exe')
opciones=Options()
#opciones.headless = True
#opciones.add_argument('--incognito')

# driver = webdriver.Firefox(firefox_binary=binary,options = opciones)
# driver.get(url)
# driver.quit()

from fake_useragent import UserAgent
usuario=UserAgent().random
#opciones.add_argument(f'user-agent={usuario}')




from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait   
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver import ActionChains as AC   
from selenium.webdriver.common.keys import Keys 
# opciones.add_experimental_option('excludeSwitches', ['enable-automation'])
# opciones.add_experimental_option('useAutomationExtension', False) 
# opciones.add_argument('--start-maximized')
# opciones.add_argument('--incognito')
#opciones.add_extension('driver/adblock.crx')       # adblocker
#opciones.add_argument('user-data-dir=selenium')  # cookies
PATH=ChromeDriverManager().install()
driver=webdriver.Chrome(PATH, options=opciones) 



import youtube_dl
# opciones={'format': 'bestaudio/best',
          
#           'postprocessors': [{  'key': 'FFmpegExtractAudio',
#                                 'preferredcodec': 'mp3',
#                                 'preferredquality': '192',
#                                                         }],
#          'outtmpl': '../data/archivo.m4a'
#          }

import feedparser
import xmltodict


from glob import glob

# Parallelization, multiprocessing, async

import asyncio
import multiprocessing as mp

# pool=mp.Pool(mp.cpu_count())
# pool=get_context('fork').Pool(6)  # grupo con 6 cores
# res=pool.map_async(fn, data).get()
# pool.close()
from joblib import Parallel, delayed
# paralelo = Parallel(n_jobs=-1, verbose=True)
# lst=paralelo(delayed(fn)(e) for e in data)


import dask
import dask.dataframe as dd
# df.DepDelay.max().visualize(filename='images/max_dask.png')

import vaex

import findspark
import pyspark
#findspark.init() 
from pyspark.sql import SparkSession
#spark=SparkSession.builder.appName('Nombre').getOrCreate()  # inicia la sesion de spark
#data=spark.read.csv(path, header=True, inferSchema=True, sep=';')
#data.show()
#data.printSchema()
#data.describe().show()

# Visualization


#pd.options.plotting.backend = 'plotly'
import chart_studio.plotly as py
import cufflinks as cf
import plotly.graph_objects as go
cf.go_offline()
# cf.set_config_file(offline=False, world_readable=True)
# df.iplot

import geopandas as gpd
from keplergl import KeplerGl

from bokeh.plotting import figure, output_file, show
from bokeh.io import curdoc, output_notebook, show, output_file
from bokeh.models import Slider, HoverTool, GeoJSONDataSource, LinearColorMapper, ColorBar
from IPython.display import YouTubeVideo
from bokeh.plotting import figure
from bokeh.palettes import brewer, all_palettes, small_palettes

import lux
import qgrid
from itables import init_notebook_mode
# init_notebook_mode(all_interactive=True)
# from itables import show
# import world_bank_data as wb
# df_c=wb.get_countries()
# show(df_c)
import atoti as tt
# sesion=tt.create_session()
# sesion.visualize()
import ipyvolume
# data=ipyvolume.datasets.hdz2000.fetch()
# ipyvolume.volshow(data.data, lighting=True, level=[.4,.6,.9])

from jupyter_nbutils import utils
# # install jupyter-require extension
# utils.install_nbextension('jupyter_require', overwrite=True)  # note there is an underscore, it's Python module name
# # load and enable the extension
# utils.load_nbextension('jupyter-require', enable=True)
from jupyter_datatables import init_datatables_mode
# %load_ext jupyter_require
# %requirejs d3 https://d3js.org/d3.v5.min
# init_datatables_mode()
# %reload_ext jupyter_require
from bqplot import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Machine Learning

import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, RepeatedKFold, LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedShuffleSplit, GroupKFold, GroupShuffleSplit, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_squared_log_error,mean_absolute_error, recall_score, precision_score, f1_score, roc_curve, roc_auc_score,confusion_matrix,r2_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, auc, silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score, mutual_info_score, normalized_mutual_info_score, fbeta_score, make_scorer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDRegressor, SGDClassifier, Perceptron, PassiveAggressiveRegressor, PassiveAggressiveClassifier, RidgeClassifier, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, TheilSenRegressor, HuberRegressor, RANSACRegressor, LogisticRegressionCV, RidgeClassifierCV, LassoLarsCV, LassoLarsIC, MultiTaskLasso, MultiTaskElasticNet, MultiTaskLassoCV, MultiTaskElasticNetCV, OrthogonalMatchingPursuitCV, Lars, LarsCV, LassoCV, LassoLarsCV, LassoLarsIC, ElasticNetCV, OrthogonalMatchingPursuitCV, BayesianRidge, ARDRegression, TheilSenRegressor, HuberRegressor, RANSACRegressor, LogisticRegressionCV, RidgeClassifierCV, SGDRegressor, SGDClassifier, Perceptron, PassiveAggressiveRegressor, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeRegressor, ExtraTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, AdaBoostRegressor, AdaBoostClassifier, GradientBoostingRegressor, GradientBoostingClassifier, BaggingRegressor, BaggingClassifier, VotingRegressor, VotingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, RadiusNeighborsRegressor, RadiusNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC, NuSVR, NuSVC, OneClassSVM
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation, MeanShift, SpectralClustering, OPTICS, Birch, estimate_bandwidth, FeatureAgglomeration, SpectralBiclustering, SpectralCoclustering, AgglomerativeClustering, Birch, DBSCAN, FeatureAgglomeration, KMeans, MiniBatchKMeans, MeanShift, OPTICS, SpectralBiclustering, SpectralCoclustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet, inconsistent, maxdists, maxinconsts, maxRstat, to_tree, leaders, fclusterdata, inconsistent, maxdists, maxinconsts, maxRstat, to_tree, leaders, fclusterdata
from hdbscan import HDBSCAN
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, TruncatedSVD, FastICA, FactorAnalysis, NMF, LatentDirichletAllocation, IncrementalPCA, MiniBatchSparsePCA, SparseCoder, DictionaryLearning, MiniBatchDictionaryLearning, KernelCenterer, MiniBatchNMF, TruncatedSVD, PCA, KernelPCA, SparsePCA, FastICA, FactorAnalysis, NMF, LatentDirichletAllocation, IncrementalPCA, MiniBatchSparsePCA, SparseCoder, DictionaryLearning, MiniBatchDictionaryLearning, KernelCenterer, MiniBatchNMF
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect, RFE, RFECV, SelectFromModel, VarianceThreshold, SelectFdr, SelectFpr, SelectFwe, SelectKBest, SelectPercentile, chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin, clone
from sklearn.utils import shuffle, resample, safe_indexing, check_random_state, check_array, check_X_y, column_or_1d, safe_mask
from sklearn.utils.validation import check_is_fitted, _check_sample_weight, _deprecate_positional_args
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.metaestimators import _BaseComposition, if_delegate_has_method
from sklearn.utils.fixes import delayed
from sklearn.utils.extmath import safe_sparse_dot, squared_norm, row_norms, stable_cumsum, _incremental_mean_and_var, _deterministic_vector_sign_flip, _deterministic_matrix_sign_flip
from sklearn.utils import check_array, check_consistent_length, check_random_state, check_X_y, column_or_1d, safe_mask
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, t_sne, spectral_embedding, isomap, locally_linear_embedding, mds
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, pairwise_distances, pairwise_distances_argmin, pairwise_distances_argmin_min, pairwise_kernels, pairwise_distances_chunked, pairwise_kernels_chunked, rbf_kernel, polynomial_kernel, laplacian_kernel, sigmoid_kernel, chi2_kernel, additive_chi2_kernel, linear_kernel, cosine_similarity, euclidean_distances, manhattan_distances, pairwise_distances, pairwise_distances_argmin, pairwise_distances_argmin_min, pairwise_kernels, pairwise_distances_chunked, pairwise_kernels_chunked, rbf_kernel, polynomial_kernel, laplacian_kernel, sigmoid_kernel, chi2_kernel, additive_chi2_kernel, linear_kernel

import scikitplot as skplt

from xgboost import XGBRegressor as XGBR
from catboost import CatBoostRegressor as CTR
from lightgbm import LGBMRegressor as LGBMR

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING, STATUS_FAIL, STATUS_NEW, STATUS_RUNNING, STATUS_SUSPENDED, STATUS_NOT_RUNNING, STATUS_NOT_STARTED, STATUS_PREEMPTED, STATUS_WAITING, STATUS_WAITING_FOR_RESOURCE, STATUS_WAITING_FOR_THREAD, STATUS_UNKNOWN, STATUS_ERROR, STATUS_LOST, STATUS_REMOVED, STATUS_D

from lazypredict.Supervised import LazyRegressor 
import h2o
from h2o.automl import H2OAutoML

from yellowbrick.cluster import KElbowVisualizer
from umap import UMAP

import pickle
# pickle.dump(rfc, open('random_forest_tsne_pulsar.pk', 'wb'))
# modelo_rf = pickle.load(open('random_forest_tsne_pulsar.pk', 'rb'))


import nltk
# descarga de paquetes
#nltk.download()
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer


import openai as ai

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from fbprophet import Prophet
import yfinance as yf

import mlflow
import mlflow.sklearn