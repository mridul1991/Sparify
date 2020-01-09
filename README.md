# Sparify

##### Project Overview and Statement:

In this project, PySpark will be used to predict the users that are most likely to  cancel the subscription for a online music streaming company called Sparkify. This classification problem has application all the areas such as pharma, retial, banking, etc.

#### Project Motivation

Every company wants to acquire and retain its loyal customer. If the company gets to know the characteristics and the userbase which has the highest probability to churn, it can take proactive steps to reduce the damage. 

The ability to efficiently manipulate large datasets with Spark is one of the highest-demand skills in the field of data. In this project we will be using a subset of the data ( 123 MB of the 12 GB) data and will be using Spark.

#### Files in the repository:

Raw Data: Data used for the project

Jupyter Notebook: Codes and analysis

Readme file: Project summary

Blog link: https://medium.com/@mkjabc/identifying-users-likely-to-churn-from-sparkify-fa089d081fb7

#### Important Steps


## 1. Installing the following libraries: 

from pyspark.sql import SparkSession

import pyspark.sql.functions as Ffrom pyspark.sql.functions import avg, col, desc, lit, min, max, split, udf, sum, when

from pyspark.sql.types import *

from datetime import datetime 

import time

import warnings

warnings.filterwarnings('ignore')

import datetime

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.metrics import roc_curve, auc

from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, GBTClassifier

from pyspark.mllib.tree import DecisionTree

from pyspark.ml.linalg import Vectors

from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric

from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

## 2. Loading the data

## 3. Exploratory Data Analysis

## 4. Feature Selection

## 5. Model Building

## 6. Evaluation and Fine tuneing

## 7. Conclusion:
The most important variable that predicts churn are total_songs_played, total_number_of_days, avg_session_length, avg_session_songs_user, total_number_of_likes, total_number_of_dislikes

#### Acknowledgment:
Udacity has been extremely helpful to help be complete this project.

