import scipy
import numpy
import pandas
import sklearn
import pyspark
import pyclustering
import pydotplus
import tweepy
import lightgbm
import tqdm
import requests
import geopandas
import altair
import seaborn
import selenium
import IPython
import py4j
import unidecode
import simplejson
import nltk
import wordcloud
import elasticsearch
import lxml
import scrapy
import gensim
import skmob
import PIL
import fim
import tensorflow
import simpletransformers
import torch
import torchaudio
import torchvision
import cv2
import transformers
import rtree
import xailib

import sys
import pytest


TEST_INPUTS = [
    "scipy",
    "numpy",
    "pandas",
    "sklearn",
    "pyspark",
    "pyclustering",
    "pydotplus",
    "tweepy",
    "lightgbm",
    "tqdm",
    "requests",
    "geopandas",
    "altair",
    "seaborn",
    "selenium",
    "IPython",
    "py4j",
    "unidecode",
    "simplejson",
    "nltk",
    "wordcloud",
    "elasticsearch",
    "lxml",
    "scrapy",
    "gensim",
    "skmob",
    "PIL",
    "fim",
    "tensorflow",
    "simpletransformers",
    "torch",
    "torchvision",
    "torchaudio",
    "cv2",
    "transformers",
    "rtree",
    "xailib",
]


@pytest.mark.parametrize("test_input", TEST_INPUTS)
def test__imports(test_input):
    assert test_input in sys.modules


if __name__ == "__main__":
    pytest.main()
