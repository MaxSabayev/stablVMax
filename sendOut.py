import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"


n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 
             os.environ.get('SLURM_JOB_CPUS_PER_NODE', 8)))
os.environ["LOKY_MAX_CPU_COUNT"] = str(n_cpus)

from stabl.single_omic import single_omic_simple, save_single_omic_results, simpleScores
from stabl.EMS import generateModel,read_json,write_json,unroll_parameters
from sklearn.model_selection import GroupShuffleSplit,RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from stabl.stabl import save_stabl_results
from stabl.sherlock import run_end

from pathlib import Path
import numpy as np
import pandas as pd
import argparse



df = pd.read_csv("../../data/data.csv",index_col=0)
outcome = pd.read_csv("../../data/outcome.csv",index_col=0)

outcome = outcome.loc[df.index]

unstimFeatures = [e for e in df.columns if len(e.split("_"))>1 and e.split("_")[-1]=="Unstim"]
nFeatures = [e for e in df.columns if len(e.split("_"))>1 and e.split("_")[-1]=="N"]
pFeatures = [e for e in df.columns if len(e.split("_"))>1 and e.split("_")[-1]=="P"]
lFeatures = [e for e in df.columns if len(e.split("_"))>1 and e.split("_")[-1]=="L"]
proteomicFeatures = [e for e in df.columns if len(e.split("_")) == 1]
colCorrespondence = {"Unstim": unstimFeatures, "N": nFeatures, "P": pFeatures, "L": lFeatures, "Proteomic": proteomicFeatures}

y = outcome["death"]

groups = None


paramFilePath = "./params.json"
savePathRoot = "./results"
os.makedirs(savePathRoot, exist_ok=True)


def experiment(paramSet: dict,savePath: str):

    outerSplitter = RepeatedStratifiedKFold(n_splits=5,n_repeats=40,random_state=paramSet["cvSeed"])
    colType = paramSet["dataset"]
    ef = (colType == "EarlyFusion")
    if ef:
        data = df
    else:
        data = df[colCorrespondence[colType]]

    preprocessing,model = generateModel(paramSet)
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', paramSet.get("n_jobs", -1)))
    model.set_params(n_jobs=n_jobs)
    results = single_omic_simple(
        data,
        y,
        outerSplitter,
        model,
        paramSet["model"],
        preprocessing,
        paramSet["taskType"],
        ef = ef,
        outer_groups=groups
    )
    save_single_omic_results(y,results,savePath,paramSet["taskType"])

    data_std = pd.DataFrame(
            data=preprocessing.fit_transform(data),
            index=data.index,
            columns=preprocessing.get_feature_names_out()
        )
    model.fit(data_std,y)
    modelPath = Path(savePath,"fullModel")
    os.makedirs(modelPath,exist_ok = True)
    if "stabl" in paramSet["model"]:
        save_stabl_results(model,modelPath,data,y,override=True)
    else:
        selectedFeats = list(data_std.columns[np.where(model.best_estimator_.coef_.flatten())])
        pd.DataFrame({
            "Feature": selectedFeats
        }).to_csv(Path(modelPath,"Selected Features.csv"), index=False)


def postProcess(paramFilePath: str):
    run_end(paramFilePath,df,y,read_json(paramFilePath)["general"]["taskType"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=int, default=0)
    parser.add_argument("idx", type=int, default=0,nargs="?")
    parser.add_argument("intensity",type=str,default='l',nargs="?")
    args = parser.parse_args()
    if args.mode == 0:
        path = Path(savePathRoot,str(args.intensity),str(args.idx))
        experiment(read_json(Path(path,"params.json")),path)
    elif args.mode == 1:
        postProcess(paramFilePath)



