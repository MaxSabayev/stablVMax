from stabl.single_omic import single_omic_simple
from stabl.EMS import generateModel,do_on_cluster,read_json,unroll_parameters
from sklearn.model_selection import LeaveOneOut,LeaveOneGroupOut
from stabl.stabl import export_stabl_to_csv,plot_stabl_path,plot_fdr_graph
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import os
import logging



logging.basicConfig(level=logging.INFO)

data = pd.read_csv("../data/WayneProteome.csv",index_col=0)
y = pd.read_csv("../data/labelLong.csv",index_col=0).loc[data.index].PE
groups = pd.Series(data = [int(e.split("_")[0]) for e in data.index], index = data.index)
outerSplitter = LeaveOneGroupOut()
taskType = 'binary'


paramFilePath = "./params.json"
savePath = "./Results"
os.makedirs(savePath, exist_ok=True)



def experiment(paramSet: dict):
    preprocessing,model = generateModel(paramSet)
    preds, selectedFeats,scores = single_omic_simple(
        data,
        y,
        outerSplitter,
        model,
        paramSet["model"],
        preprocessing,
        taskType,
        outer_groups=groups
    )
    name = f"{paramSet["model"]}_{paramSet["varValues"]}_{paramSet['artificialTypes'] if 'artificialTypes' in paramSet.keys() else "m"}"
    os.makedirs(Path(savePath,name), exist_ok=True)
    preds.to_csv(Path(savePath,name,"cvPreds.csv"))
    selectedFeats.to_csv(Path(savePath,name,"selectedFeats.csv"))
    scores.to_csv(Path(savePath,name,"cvScores.csv"))
    if "stabl" in paramSet["model"]:
        data_std = pd.DataFrame(
            data=preprocessing.fit_transform(data),
            index=data.index,
            columns=preprocessing.get_feature_names_out()
        )
        model.fit(data_std,y,groups=groups)
        modelPath = Path(savePath,name,"fullModel")
        os.makedirs(modelPath,exist_ok = True)
        export_stabl_to_csv(model,modelPath)
        plot_fdr_graph(model,path=Path(modelPath,"fdr.pdf"))
        plot_stabl_path(model,path=Path(modelPath,"StablPath.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int, default=0)
    parser.add_argument("mode",type=int,default=0)
    args = parser.parse_args()
    
    if args.mode == 0:
        with open("./tempProfiles/lowImpactIdx.txt","r") as f:
            idx = int(f.readlines()[args.idx])
    elif args.mode == 1:
        with open("./tempProfiles/highImpactIdx.txt","r") as f:
            idx = int(f.readlines()[args.idx])
    paramList = unroll_parameters(read_json(paramFilePath))
    experiment(paramList[idx])
