from stabl.single_omic import single_omic_simple
from stabl.EMS import generateModel,do_on_cluster
from sklearn.model_selection import LeaveOneOut
from pathlib import Path
import numpy as np
import argparse
import json
import os
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import logging

logging.basicConfig(level=logging.INFO)

data = None
y = None
outerSplitter = LeaveOneOut()
taskType = 'binary'


paramFilePath = "./params.json"
savePath = "./Results"
os.makedirs(savePath, exist_ok=True)


def experiment(paramSet: dict):
    preprocessing,model = generateModel(paramSet)
    preds, selectedFeats = single_omic_simple(
        data,
        y,
        outerSplitter,
        model,
        paramSet["model"],
        preprocessing,
        taskType,
    )
    # preds.to_csv(Path(savePath,"preds-"+title+".csv"))
    # selectedFeats.to_csv(Path(savePath,"selectedFeats-"+title+".csv"))


def do_cluster_experiment(paramPath):
    with SLURMCluster(cores=8, memory='4GiB', processes=1, walltime='16:00:00') as cluster:
        cluster.scale(8)
        logging.info(cluster.job_script())
        with Client(cluster) as client:
            do_on_cluster(paramPath, experiment, client)
        cluster.scale(0)


if __name__ == "__main__":
    do_cluster_experiment(paramFilePath)