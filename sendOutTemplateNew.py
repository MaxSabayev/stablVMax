from stabl.single_omic import single_omic_simple
from stabl.EMS import generateModel,do_on_cluster,read_json,write_json,unroll_parameters
from sklearn.model_selection import LeaveOneOut,LeaveOneGroupOut
from stabl.stabl import save_stabl_results
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

orderby = "ROC AUC" if taskType == "binary" else "R2"
keepTop = 5

paramFilePath = "./params.json"
savePath = "./Results"
os.makedirs(savePath, exist_ok=True)



def experiment(paramSet: dict,idx: int):
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
    name = str(idx)

    os.makedirs(Path(savePath,name), exist_ok=True)
    write_json(paramSet,Path(savePath,name,"experimentParams.json"))
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
        save_stabl_results(model,modelPath,data,y,override=True)

def postProcess(paramFilePath: str):
    paramList = unroll_parameters(read_json(paramFilePath))
    n = len(paramList)
    scores = pd.DataFrame(index=range(n))
    for i in range(n):
        scores[i] = pd.read_csv(Path(savePath,str(i),"cvScores.csv"),index_col=0)
        if i ==0:
            scores.columns = pd.read_csv(Path(savePath,str(i),"cvScores.csv"),index_col=0).columns
    scores["names"] = [f"{paramSet["model"]}_{b}" for paramSet,b in zip(paramList,scores.index)]
    for i in range(n):
        os.rename(Path(savePath,str(i)),Path(savePath,scores.loc[i,"names"]))
    scores = scores.reset_index(drop=True).set_index("names")
    scores = scores.sort_values(by=orderby,axis=1,ascending=False)
    
    scoresBottom = scores.iloc[keepTop:,:]
    os.makedirs(Path(savePath,"poor"),exist_ok=True)
    for name in scoresBottom.index:
        os.rename(Path(savePath,name),Path(savePath,"poor",name))
    scores.to_csv(Path(savePath,"scores.csv"))
    
        





    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=int, default=0)
    parser.add_argument("idx", type=int, default=0)
    parser.add_argument("runMode",type=int,default=0)
    args = parser.parse_args()
    if args.mode == 0:
        if args.runMode == 0:
            with open("./temp/lowImpactIdx.txt","r") as f:
                idx = int(f.readlines()[args.idx])
        elif args.runMode == 1:
            with open("./temp/highImpactIdx.txt","r") as f:
                idx = int(f.readlines()[args.idx])
        paramList = unroll_parameters(read_json(paramFilePath))
        experiment(paramList[idx],idx)
    elif args.mode == 1:
        postProcess(paramFilePath)
