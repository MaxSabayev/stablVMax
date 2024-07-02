from .EMS import read_json,unroll_parameters
import os
import numpy as np
import re
import argparse

defaultScript = """#!/usr/bin/bash
#SBATCH --job-name=NAME_V
#SBATCH --error=./logs/NAME_V_%a.err
#SBATCH --output=./logs/NAME_V_%a.out
#SBATCH --array=0-REP
#SBATCH --time=24:00:00
#SBATCH -p normal
#SBATCH -c COUNT
#SBATCH --mem=8GB

ml python/3.12.1
time python3 ./sendOut.py 0 ${SLURM_ARRAY_TASK_ID} MODE"""

def parse_params(paramsFile: str)->None:
    params = read_json(paramsFile)
    paramList = unroll_parameters(params)
    os.makedirs("./temp/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    highImpactIdx = np.argwhere(["en" in p["model"] for p in paramList]).flatten().astype(int)
    lowImpactIdx = np.array(list(set(range(len(paramList))).difference(set(highImpactIdx))))
    np.savetxt("./temp/highImpactIdx.txt",highImpactIdx,fmt="%i")
    np.savetxt("./temp/lowImpactIdx.txt",lowImpactIdx,fmt="%i")
    script = re.sub("NAME",params["Experiment_Name"],defaultScript)

    with open('./temp/arrayLow.sh', 'w') as file:
        sc = re.sub("MODE","0",script)
        sc = re.sub("REP",str(len(lowImpactIdx)-1),sc)
        sc = re.sub("COUNT","8",sc)
        sc = re.sub("V","l",sc)
        file.write(sc)
    
    with open('./temp/arrayHigh.sh', 'w') as file:
        sc = re.sub("MODE","1",script)
        sc = re.sub("REP",str(len(highImpactIdx)-1),sc)
        sc = re.sub("COUNT","32",sc)
        sc = re.sub("V","h",sc)
        file.write(sc)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("paramsPath", type=str, default="./params.json")
#     args = parser.parse_args()
#     parse_params(args.paramsPath)