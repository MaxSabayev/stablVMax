from stabl.single_omic import single_omic_simple, simpleScores
from stabl.EMS import generateModel,read_json,write_json,unroll_parameters
from sklearn.model_selection import LeaveOneOut,LeaveOneGroupOut
from stabl.stabl import save_stabl_results
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import os 
import logging



logging.basicConfig(level=logging.INFO)

datasets = {}

datasets["Proteome"] = pd.read_csv("../data/WayneProteome.csv",index_col=0)
datasets["CyTOF"] = pd.read_csv("../data/WayneCyTOF.csv",index_col=0)
y = pd.read_csv("../data/labelLong.csv",index_col=0)
groups = pd.Series(data = [int(e.split("_")[0]) for e in datasets["CyTOF"].index], index = datasets["CyTOF"].index)
outerSplitter = LeaveOneGroupOut()
taskType = 'binary'

datasets["EarlyFusion"] = pd.concat([d for _,d in datasets.items()],axis=1)

orderby = "ROC AUC" if taskType == "binary" else "R2"
keepTop = 5

paramFilePath = "./params.json"
savePath = "./Results"
os.makedirs(savePath, exist_ok=True)



def experiment(paramSet: dict,idx: int):
    data = datasets[paramSet["dataset"]]
    y = y.loc[data.index].PE
    preprocessing,model = generateModel(paramSet)
    preds, formattedFeats, selectedFeats,insamplePredictions = single_omic_simple(
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
    write_json({k :v.tolist() if type(v) == np.ndarray else v for k,v in paramSet.items()},Path(savePath,name,"experimentParams.json"))
    preds.to_csv(Path(savePath,name,"cvPreds.csv"))
    formattedFeats.to_csv(Path(savePath,name,"selectedFeats.csv"))
    scores = simpleScores(preds,y,formattedFeats,taskType)
    pd.DataFrame(selectedFeats).to_csv(Path(savePath,name,"noformatFeats.csv"))
    pd.DataFrame(insamplePredictions).to_csv(Path(savePath,name,"insamplePreds.csv"))
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
    scores = None
    for i in range(n):
        if scores is None:
            scores = pd.read_csv(Path(savePath,str(i),"cvScores.csv"),index_col=0)
        else:
            scores = pd.concat((scores, pd.read_csv(Path(savePath,str(i),"cvScores.csv"),index_col=0)),axis=1)
    scores = scores.T
    scores.index = range(n)

    scores["names"] = [f"{paramSet["model"]}_{b}" for paramSet,b in zip(paramList,scores.index)]
    for i in range(n):
        os.rename(Path(savePath,str(i)),Path(savePath,scores.loc[i,"names"]))
    scores = scores.reset_index(drop=True).set_index("names")
    scores = scores.sort_values(by=orderby,axis=0,ascending=False)
    
    scoresBottom = scores.iloc[keepTop:,:]
    os.makedirs(Path(savePath,"poor"),exist_ok=True)
    for name in scoresBottom.index:
        os.rename(Path(savePath,name),Path(savePath,"poor",name))
    scores.to_csv(Path(savePath,"scores.csv"))
    
        





    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=int, default=0)
    parser.add_argument("idx", type=int, default=0,nargs="?")
    parser.add_argument("runMode",type=int,default=0,nargs="?")
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
