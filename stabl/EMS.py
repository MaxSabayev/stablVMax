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
    models = [k for k in params["general"]["models"].keys() if params["general"]["models"][k]]
    stablModels = [m for m in models if "stabl" in m]
    nonStablModels = [m for m in models if "stabl" not in m]
    
    stablParams = {"model":stablModels,**params["preprocessing"],**params["stabl_general"]}
    nonStablParams = {"model":nonStablModels,**params["preprocessing"]}

    experiments = [{key: value for key, value in zip(nonStablParams.keys(), combo)} for combo in itertools.product(*nonStablParams.values())]
    experiments.extend([{key: value for key, value in zip(stablParams.keys(), combo)} for combo in itertools.product(*stablParams.values())])
    for exp in experiments:
        exp["varType"] = params["general"]["varType"]
        exp["innerCVvals"] = params["general"]["innerCVvals"]
        for modelVariableName in params[exp["model"]].keys():
            exp[modelVariableName] = spacerize(params[exp["model"]][modelVariableName])
        exp["varNames"] = list(params[exp["model"]].keys())

    return experiments

def parse_params(paramsFile: str)->tuple:
    params = read_json(paramsFile)
    paramList = unroll_parameters(params)
    os.makedirs("./tempProfiles/", exist_ok=True)
    highImpactIdx = np.argwhere(["en" in p["model"] for p in paramList]).flatten().astype(int)
    lowImpactIdx = np.array(list(set(range(len(paramList))).difference(set(highImpactIdx))))
    np.savetxt("./tempProfiles/highImpactIdx.txt",highImpactIdx,fmt="%i")
    np.savetxt("./tempProfiles/lowImpactIdx.txt",lowImpactIdx,fmt="%i")
    print(len(lowImpactIdx))
    print(len(highImpactIdx))
    
        



def generateModel(paramSet: dict):
    preprocessingList = []
    #match paramSet["varType"]:
    if paramSet["varType"] == "thresh":
        preprocessingList.append(("varianceThreshold",VarianceThreshold(paramSet["varValues"])))
    else:
        raise Exception("Unimplemented variance thresholding type")
            
    preprocessingList.extend([("lif",LowInfoFilter(paramSet["lifThresh"])),
                               ("impute", SimpleImputer(strategy="median")),
                               ("std", StandardScaler())])
    preprocessing = Pipeline(steps=preprocessingList)
    lambdaGrid = None
    if paramSet["model"] == "stabl_lasso" or  paramSet["model"] == "lasso":
        submodel = LogisticRegression(penalty="l1", class_weight="balanced", 
                                            max_iter=int(1e6), solver="liblinear", random_state=42)
    elif paramSet["model"] == "stabl_alasso" or paramSet["model"] == "alasso":
        submodel = ALogitLasso(penalty="l1", solver="liblinear", 
                                    max_iter=int(1e6), class_weight='balanced', random_state=42)
    elif paramSet["model"] == "stabl_en" or paramSet["model"] == "en":
        submodel = LogisticRegression(penalty='elasticnet',solver='saga',
                                        class_weight='balanced',max_iter=int(1e6),random_state=42)
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
                    random_state=42,
                    lambda_grid=lambdaGrid,
                    verbose=1
                )
    else:
        chosen_inner_cv = RepeatedStratifiedKFold(n_splits=paramSet["innerCVvals"][0],n_repeats=paramSet["innerCVvals"][1], random_state=42)
        model = GridSearchCV(submodel, param_grid=lambdaGrid, 
                             scoring="roc_auc", cv=chosen_inner_cv, n_jobs=-1)
    
    return preprocessing,model

            

    
# def do_experiment(instance: callable, parameters: list, client: Client): #db: Databases):
#     instance_count = len(parameters)
#     i = 0
#     logger.info(f'Number of Instances to calculate: {instance_count}')
#     # Start the computation.
#     tick = time.perf_counter()
#     futures = client.map(lambda p: instance(p), parameters, batch_size=BATCH_SIZE)
#     for batch in as_completed(futures, with_results=True).batches():
#         for future, result in batch:
#             i += 1
#             if not (i % 10):  # Log results every tenth output
#                 tock = time.perf_counter() - tick
#                 remaining_count = instance_count - i
#                 s_i = tock / i
#                 logger.info(f'Count: {i}; Time: {round(tock)}; Seconds/Instance: {s_i:0.4f}; ' +
#                             f'Remaining (s): {round(remaining_count * s_i)}; Remaining Count: {remaining_count}')
#                 logger.info(result)
#             future.release()  # As these are Embarrassingly Parallel tasks, clean up memory.

#     total_time = time.perf_counter() - tick
#     logger.info(f"Performed experiment in {total_time:0.4f} seconds")
#     if instance_count > 0:
#         logger.info(f"Count: {instance_count}, Seconds/Instance: {(total_time / instance_count):0.4f}")


# def do_on_cluster(parameterPath: str, function: callable, client: Client):
#     logger.info(f'{client}')
#     parameterList = read_json(parameterPath)
#     # Save the experiment domain.
#     record_experiment(parameterList)

#     # Prepare parameters.
#     parameters = unroll_parameters(parameterList)

#     if len(parameters) > 0:
#         random.shuffle(parameters)
#         do_experiment(function, parameters, client)
#     else:
#         logger.warning('Empty parameters.')
#     client.shutdown()

