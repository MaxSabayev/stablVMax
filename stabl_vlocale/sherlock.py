from .EMS import read_json,unroll_parameters,write_json
from .single_omic import simpleScores,late_fusion_combination_normal,late_fusion_combination_stabl
import os
import numpy as np
import pandas as pd
import re
from pathlib import Path
from .visualization import boxplot_binary_predictions, plot_roc
import time

defaultScript = """#!/usr/bin/bash
#SBATCH --job-name=NAME_V
#SBATCH --error=./logs/NAME_V_%a.err
#SBATCH --output=./logs/NAME_V_%a.out
#SBATCH --array=0-REP
#SBATCH --time=48:00:00
#SBATCH -p normal
#SBATCH -c COUNT
#SBATCH --mem=MEMOGB

ml python/3.12.1
time python3 ./sendOut.py 0 ${SLURM_ARRAY_TASK_ID} V"""

endScript = """#!/usr/bin/bash
#SBATCH --job-name=NAME_e
#SBATCH --error=./logs/NAME_e.err
#SBATCH --output=./logs/NAME_e.out
#SBATCH --time=48:00:00
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=8GB

ml python/3.12.1
time python3 ./sendOut.py 1"""

def parse_params(paramsFile: str,highMem: bool = False)->None:
    params = read_json(paramsFile)
    paramList = unroll_parameters(params)
    os.makedirs("./temp/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./results/", exist_ok=True)

    lowCount = 0
    highCount = 0
    for param in paramList:
        a,b = param["shorthand"].split("_")
        os.makedirs(f"./results/{b}/{a}",exist_ok=True)
        write_json(param,f"./results/{b}/{a}/params.json")
        if b == "h":
            highCount += 1
        else:
            lowCount += 1


    script = re.sub("NAME",params["Experiment_Name"],defaultScript)
    if highMem:
        script = re.sub("MEMO","32",script)
    else:
        script = re.sub("MEMO","16",script)
    
    if lowCount != 0:
        os.makedirs("./results/l/", exist_ok=True)
        with open('./temp/arrayLow.sh', 'w') as file:
            sc = re.sub("V","l",script)
            sc = re.sub("REP",str(lowCount-1),sc)
            sc = re.sub("COUNT","8",sc)
            file.write(sc)

    if highCount != 0:
        os.makedirs("./results/h/", exist_ok=True)
        with open('./temp/arrayHigh.sh', 'w') as file:
            sc = re.sub("V","h",script)
            sc = re.sub("REP",str(highCount-1),sc)
            sc = re.sub("COUNT","32",sc)
            file.write(sc)
    
    with open('./temp/end.sh', 'w') as file:
        eScript = re.sub("NAME",params["Experiment_Name"],endScript)
        file.write(eScript)
    


def run_end(paramsFile: str,
            data: pd.DataFrame,
            y: pd.Series,
            taskType: str
            )->None:
    params = read_json(paramsFile)
    intensities = os.listdir("./results/")
    ef = "EarlyFusion" in params["datasets"]
    n = len(params["datasets"])
    m = n -ef
    lf = m > 1
    scores = None
    for intensity in intensities:
        pathR = Path("./results/",intensity)
        exps = []
        existingParams = []
        for exp in os.listdir(pathR):
            if os.path.exists(Path(pathR,exp,"cvScores.csv")):
                existingParams.append(read_json(Path(pathR,exp,"params.json")))
                sc = pd.read_csv(Path(pathR,exp,"cvScores.csv"),index_col=0,names=[f"{existingParams[-1]['model']}_{exp}_{intensity}"],header=0)
                if scores is None:
                    scores = sc
                else:
                    scores = pd.concat((scores, sc),axis=1)
                exps.append(int(exp))

        if lf:
            lfGroupTags = np.array([e["lfTag"] for e in existingParams])
            lfGroups = [np.argwhere(lfGroupTags == i).flatten() for i in np.unique(lfGroupTags)]
            lfGroupsSTABL = [[existingParams[ee]["shorthand"].split("_")[0] for ee in e if existingParams[ee]["dataset"] != "EarlyFusion" and "stabl" in existingParams[ee]["model"]] for e in lfGroups]
            lfGroupsNonSTABL = [[existingParams[ee]["shorthand"].split("_")[0] for ee in e if existingParams[ee]["dataset"] != "EarlyFusion" and "stabl" not in existingParams[ee]["model"]] for e in lfGroups]
            lfGroupsSTABL = [np.sort(np.array(e).astype(int)) for e in lfGroupsSTABL if len(e) > 1]
            lfGroupsNonSTABL = [np.sort(np.array(e).astype(int)) for e in lfGroupsNonSTABL if len(e) > 1]

            for grp in lfGroupsSTABL:
                print(grp)
                selectedFeats = pd.concat([pd.read_csv(Path(pathR,str(e),"selectedFeats.csv"),index_col=0).astype(bool)for e in grp] ,axis=1)
                prd = pd.read_csv(Path(pathR,str(grp[0]),"cvPreds.csv"),index_col=0)
                splits = [[np.argwhere(prd[col].isna()).flatten(),np.argwhere(~prd[col].isna()).flatten()] for col in prd.columns]
                lfPreds = late_fusion_combination_stabl(data,y,selectedFeats,splits,taskType)
                tts = time.time()
                lfScores = simpleScores(lfPreds,y,selectedFeats,taskType)
                print(time.time()-tts)
                featCount = selectedFeats.sum(axis=0).T.sort_values(ascending=False)
                pathLF = Path(pathR,f"lf_{str(grp[0])}")
                p = read_json(Path(pathR,str(grp[0]),"params.json"))
                p["dataset"] = "LateFusion"
                os.makedirs(pathLF,exist_ok=True)
                featCount.to_csv(Path(pathLF,"featCount.csv"))
                lfScores.to_csv(Path(pathLF,"cvScores.csv"))
                lfPreds.to_csv(Path(pathLF,"cvPreds.csv"))
                write_json(p,Path(pathLF,"params.json"))
                lfScores.columns = [f"{p["model"]}_{str(grp[0])}_{intensity}_lf" ]
                scores = pd.concat((scores,lfScores),axis=1)
                if taskType == "binary":
                    plot_roc(y,lfPreds.median(axis=1),show_fig=False,path=Path(pathLF,"ROC.png"),export_file=True)  
                    boxplot_binary_predictions(y,lfPreds.median(axis=1),show_fig=False,path=Path(pathLF,"predBoxplot.png"),export_file=True)


            for grp in lfGroupsNonSTABL:
                print(grp)
                isPreds = [pd.read_csv(Path(pathR,str(e),"insamplePreds.csv"),index_col=0) for e in grp]
                oosPreds = [pd.read_csv(Path(pathR,str(e),"cvPreds.csv"),index_col=0) for e in grp]
                selectedFeats = pd.concat([pd.read_csv(Path(pathR,str(e),"selectedFeats.csv"),index_col=0).astype(bool) for e in grp] ,axis=1)
                lfPreds = late_fusion_combination_normal(y,oosPreds,isPreds)
                tts = time.time()
                lfScores = simpleScores(lfPreds,y,selectedFeats,taskType)
                print(time.time()-tts)
                featCount = selectedFeats.sum(axis=0).T.sort_values(ascending=False)
                pathLF = Path(pathR,f"lf_{str(grp[0])}")
                p = read_json(Path(pathR,str(grp[0]),"params.json"))
                p["dataset"] = "LateFusion"
                os.makedirs(pathLF,exist_ok=True)
                featCount.to_csv(Path(pathLF,"featCount.csv"))
                lfScores.to_csv(Path(pathLF,"cvScores.csv"))
                lfPreds.to_csv(Path(pathLF,"cvPreds.csv"))
                write_json(p,Path(pathLF,"params.json"))
                lfScores.columns = [f"{p["model"]}_{str(grp[0])}_{intensity}_lf"]
                scores = pd.concat((scores,lfScores),axis=1)
                if taskType == "binary":
                    plot_roc(y,lfPreds.median(axis=1),show_fig=False,path=Path(pathLF,"ROC.png"),export_file=True)  
                    boxplot_binary_predictions(y,lfPreds.median(axis=1),show_fig=False,path=Path(pathLF,"predBoxplot.png"),export_file=True)

    scores = scores.T
    scores = scores.astype(float)
    scores.sort_values(by=scores.columns[0],ascending=False).to_csv("./results/cvScores.csv")
    return 