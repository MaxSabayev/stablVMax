#!/usr/bin/env python3

import copy
import json
import logging
import os
import random
import time
from datetime import datetime, timezone, timedelta
from math import floor
from pathlib import Path
import itertools
import pandas as pd
from pandas import DataFrame
import numpy as np
# from dask.distributed import Client, as_completed
from sklearn.model_selection import RepeatedStratifiedKFold, GroupShuffleSplit, GridSearchCV, RepeatedKFold
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from .stabl import Stabl, group_bootstrap
from .adaptive import ALogitLasso, ALasso
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .preprocessing import LowInfoFilter

BATCH_SIZE = 4096

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)



def timestamp() -> int:
    return floor(_now().timestamp())


def write_json(d: dict, fn: str):
    with open(fn, 'w') as json_file:
        for key in d.keys():
            if isinstance(d[key], np.ndarray):
                d[key] = d[key].tolist()
        json.dump(d, json_file, indent=4)


def read_json(fn: str) -> dict:
    with open(fn, 'r') as json_file:
        d = json.load(json_file)
    return d


def record_experiment(experiment: dict):
    table_name = experiment['table_name']
    now_ts = timestamp()
    write_json(experiment, table_name + f'-{now_ts}.json')


def spacerize(spaceDict: dict):
    if spaceDict["type"] == "log":
        return np.logspace(*spaceDict["val"])
    elif spaceDict["type"] == "lin":
        return np.linspace(*spaceDict["val"])

def unroll_parameters(params: dict) -> list:
    models = [k for k in params["models"].keys() if params["models"][k]]
    stablModels = [m for m in models if "stabl" in m]
    nonStablModels = [m for m in models if "stabl" not in m]
    
    stablParams = {"model":stablModels,**params["preprocessing"],**params["stabl_general"]}
    nonStablParams = {"model":nonStablModels,**params["preprocessing"]}

    experiments = [{key: value for key, value in zip(nonStablParams.keys(), combo)} for combo in itertools.product(*nonStablParams.values())]
    experiments.extend([{key: value for key, value in zip(stablParams.keys(), combo)} for combo in itertools.product(*stablParams.values())])
    for exp in experiments:
        for key in params["general"].keys():
            exp[key] = params["general"][key]
        for modelVariableName in params[exp["model"]].keys():
            if modelVariableName == "hyperparameters":
                for modelHyperParamName in params[exp["model"]]["hyperparameters"].keys():
                    exp[modelHyperParamName] = spacerize(params[exp["model"]]["hyperparameters"][modelHyperParamName])
            else:
                exp[modelVariableName] = params[exp["model"]][modelVariableName]
        exp["varNames"] = list(params[exp["model"]]["hyperparameters"].keys())
    
    num = len(params["datasets"])
    lfTag = 0
    experimentsFull = []
    for exp in experiments:
        cvSeed = np.random.randint(2**32-1)
        for i in range(num):
            newExp = copy.deepcopy(exp)
            newExp["dataset"] = params["datasets"][i]
            newExp["cvSeed"] = cvSeed
            newExp["lfTag"] = lfTag
            experimentsFull.append(newExp)
        lfTag += 1
    experiments = experimentsFull
    
    h = 0
    l = 0
    for exp in experiments:
        if 'en' in exp['model']:
            exp["shorthand"] = f"{h}_h"
            h += 1
        else:
            exp["shorthand"] = f"{l}_l"
            l += 1
    return experiments


    

def generateModel(paramSet: dict):
    preprocessingList = []
    if paramSet["varType"] == "thresh":
        preprocessingList.append(("varianceThreshold",VarianceThreshold(paramSet["varValues"])))
    else:
        raise Exception("Unimplemented variance thresholding type")
            
    preprocessingList.extend([("lif",LowInfoFilter(paramSet["lifThresh"])),
                               ("impute", SimpleImputer(strategy="median")),
                               ("std", StandardScaler())])
    preprocessing = Pipeline(steps=preprocessingList)
    lambdaGrid = None
    maxIter = int(paramSet["max_iter"])
    if paramSet["useRandomSeed"]:
        seed = None
    else:
        seed = int(paramSet["seed"])
    if paramSet["model"] == "stabl_lasso" or  paramSet["model"] == "lasso":
        submodel = LogisticRegression(penalty="l1", class_weight="balanced", 
                                            max_iter=maxIter, solver="liblinear", random_state=seed)
    elif paramSet["model"] == "stabl_alasso" or paramSet["model"] == "alasso":
        submodel = ALogitLasso(penalty="l1", solver="liblinear", 
                                    max_iter=maxIter, class_weight='balanced', random_state=seed)
    elif paramSet["model"] == "stabl_en" or paramSet["model"] == "en":
        submodel = LogisticRegression(penalty='elasticnet',solver='saga',
                                        class_weight='balanced',max_iter=maxIter,random_state=seed)
        if "stabl" in paramSet["model"]:
            lambdaGrid = [{b:paramSet[b] for b in paramSet["varNames"]}]
        # case "sgl":
        #     submodel = LogisticSGL(max_iter=int(1e3), l1_ratio=0.5)
    else:
        raise Exception("Invalid model type.")
    if lambdaGrid is None:
        lambdaGrid = {v:paramSet[v] for v in paramSet["varNames"]}
    if "stabl" in paramSet["model"]:
        model = Stabl(
                    submodel,
                    n_bootstraps=paramSet["n_bootstraps"],
                    artificial_type=paramSet["artificialTypes"],
                    artificial_proportion=paramSet["artificialProportions"],
                    replace=paramSet["replace"],
                    fdr_threshold_range=np.arange(*paramSet["fdrThreshParams"]),
                    sample_fraction=paramSet["sampleFractions"],
                    random_state=seed,
                    lambda_grid=lambdaGrid,
                    n_jobs=paramSet["n_jobs"],
                    verbose=1
                )
    else:
        chosen_inner_cv = RepeatedStratifiedKFold(n_splits=paramSet["innerCVvals"][0],n_repeats=paramSet["innerCVvals"][1], random_state=seed)
        model = GridSearchCV(submodel, param_grid=lambdaGrid, 
                             scoring="roc_auc", cv=chosen_inner_cv, n_jobs=paramSet["n_jobs_nonstabl"])
    
    return preprocessing,model



