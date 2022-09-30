#code based on module priciples of data science from my MSc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from typing import List
from yellowbrick.classifier import ConfusionMatrix,PrecisionRecallCurve,ROCAUC,DiscriminationThreshold,ClassPredictionError,ClassificationReport

def df_merger(metrics: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
     Merge metrics and test dataframes based on names.

    Parameters
    ----------
    metrics : pd.DataFrame
        metrics dataframe
    test : pd.DataFrame
        test dataframe

    Returns
    -------
    pd.DataFrame
        merged dataframe
    """

    splitted = metrics["Name"].str.split()
    metrics["name"] = splitted.str[1] + ", " + splitted.str[0]

    # merge base on 'name' column
    df = pd.merge(metrics, test)

    # drop repeated col
    df.drop("name", axis=1, inplace=True)

    return df


def mean_imputer(df: pd.DataFrame, numeric_cols: List) -> pd.DataFrame:
    """
    Imputes mean values on datframe columns each of the columns passed in as a list

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to impute on
    numeric_cols : List
        cols to imput

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in numeric_cols:
        mean = df[col].mean()
        df[col].fillna(mean, inplace=True)

    return df

import xgboost

def plot_eval_metrics(model:xgboost.XGBClassifier,X_train:pd.DataFrame,y_train:pd.Series,X_test:pd.DataFrame,y_test:pd.Series) ->None:
    """
    Plots key performance metrics for an xgb classifier model

    Parameters
    ----------
    model : xgboost.XGBClassifier
        xgb estimator
    X_train : pd.DataFrame
        
    y_train : pd.Series

    X_test : pd.DataFrame
        
    y_test : pd.Series
        
    """
    visualDict = {"CM":ConfusionMatrix,
                "Classification Report": ClassificationReport,
                "ROC AUC":ROCAUC,
                "PR AUC": PrecisionRecallCurve,
                "Class Pred Error": ClassPredictionError}

    SetsDict = {"Train":0,
                "Test":1,
            }

    fig, axes = plt.subplots(5, 2)

    for i,key in  zip(range(0,5),visualDict):
        for t in SetsDict.keys():
            if t == "Train":
                viz = visualDict[key](model,ax=axes[i,SetsDict[t]],size=(1200, 1080),title="{} {}".format(t,key),is_fitted=True)
                viz.fit(X_train, y_train)
                viz.score(X_train, y_train)
                viz.finalize()    
                            
            elif t == "Test":
                viz = visualDict[key](model,ax=axes[i,SetsDict[t]],size=(1200, 1080),title="{} {}".format(t,key),is_fitted=True)
                viz.fit(X_train, y_train)
                viz.score(X_test, y_test)
                viz.finalize()
                
            if key == "ROC AUC":
                gini = round((viz.score_*2-1),2)
                axes[i,SetsDict[t]].set_title("{} {} - Gini: {}".format(t,key,gini))
