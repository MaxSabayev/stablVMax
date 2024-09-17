from .EMS import read_json,unroll_parameters,write_json
import os
import numpy as np
import re
import argparse

defaultScript = """#!/usr/bin/bash
#SBATCH --job-name=NAME_V
#SBATCH --error=./logs/NAME_V_%a.err
#SBATCH --output=./logs/NAME_V_%a.out
#SBATCH --array=0-REP
#SBATCH --time=48:00:00
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
    os.makedirs("./results/", exist_ok=True)
    os.makedirs("./results/h/", exist_ok=True)
    os.makedirs("./results/l/", exist_ok=True)

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

    if lowCount != 0:
        with open('./temp/arrayLow.sh', 'w') as file:
            sc = re.sub("MODE","0",script)
            sc = re.sub("REP",str(lowCount-1),sc)
            sc = re.sub("COUNT","8",sc)
            sc = re.sub("V","l",sc)
            file.write(sc)

    if highCount != 0:
        with open('./temp/arrayHigh.sh', 'w') as file:
            sc = re.sub("MODE","1",script)
            sc = re.sub("REP",str(highCount-1),sc)
            sc = re.sub("COUNT","32",sc)
            sc = re.sub("V","h",sc)
            file.write(sc)

    print(int(lowCount  != 0))
    print(int(highCount != 0))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("paramsPath", type=str, default="./params.json")
#     args = parser.parse_args()
#     parse_params(args.paramsPath)