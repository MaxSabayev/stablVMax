from stabl.single_omic import single_omic_simple
from stabl.split import generateModel
from sklearn.model_selection import LeaveOneOut
from pathlib import Path
import numpy as np
import argparse
import json
import os

data = None
y = None
outerSplitter = LeaveOneOut()
taskType = 'binary'


paramFilePath = "./params.json"
savePath = "./Results"
os.makedirs(savePath, exist_ok=True)



nameList = ["lasso","alasso","en","sgl"]
varNames = ["varThresh", "varPerc", "unqPerc"]
varNamesFull =["varThreshValues", "varPercentileValues", "uniquenessPercentileValues"]

def run(idx):
    params = json.loads(open(paramFilePath).read())
    varThreshCount =  len(params["preprocessing"]["varThreshValues"]) if params["preprocessing"]["useVarThresh"] else -1
    varPercentileCount =  len(params["preprocessing"]["varPercentileValues"]) if params["preprocessing"]["useVarPercentile"] else -1
    varUniquenessCount =  len(params["preprocessing"]["uniquenessPercentileValues"]) if params["preprocessing"]["useUniqueness"] else -1
    lifCount =  len(params["preprocessing"]["lifThresh"])

    varList = [varThreshCount,varPercentileCount,varUniquenessCount]
    varType = np.argmax(varList)
    varCount = varList[varType]

    nonStablList = [ params[nameList[i]]["use"] for i in range(4)]
    stablList = [ params["stabl_" + nameList[i]]["use"] for i in range(4)]
    preprocCount = varCount* lifCount
    nonStablCount = np.sum(nonStablList)
    stablCount =  np.sum(stablList)

    idxList = [0,0,0,0]

    if idx < preprocCount*nonStablCount:
        modelIdx = idx//preprocCount
        ii = 0
        while ii < modelIdx:
            ii += 1
            if not nonStablList[ii]: modelIdx += 1
        modelName = nameList[ii]
        v = idx%preprocCount

        varIdx = v//lifCount
        lifIdx = v%lifCount
        idxList[3] = lifIdx
        idxList[varType] = varIdx
        varValue = params["preprocessing"][varNamesFull[varType]][varIdx]
        lifValue = params["preprocessing"]["lifThresh"][lifIdx]

        title = f"{modelName}-{varNames[varIdx]}{varValue}-lif{lifValue}"

        preprocessing,model = generateModel(idxList,modelName,paramFilePath)
    else:
        idx -= preprocCount*nonStablCount 

        artificialTypeCount = len(params["stabl_general"]["artificialTypes"])
        artificialPropCount = len(params["stabl_general"]["artificialProportions"])
        sampleFracCount = len(params["stabl_general"]["sampleFractions"])
        stablHyperCount = artificialTypeCount * artificialPropCount * sampleFracCount

        modelIdx = idx//(preprocCount * stablHyperCount)
        ii = 0
        while ii < modelIdx:
            ii += 1
            if not stablList[ii]: modelIdx += 1
        modelName = "stabl_" + nameList[ii]

        v = idx%(preprocCount * stablHyperCount)

        lifIdx = v//(stablHyperCount* varCount)
        v = v%(stablHyperCount* varCount)
        varIdx = v//(stablHyperCount)
        v = v%stablHyperCount
        artificialTypeIdx = v//(artificialPropCount * sampleFracCount)
        v = v%(artificialPropCount * sampleFracCount)
        artificialPropIdx = v//(sampleFracCount)
        sampleFracIdx = v%(sampleFracCount)
        idxList[3] = lifIdx
        idxList[varType] = varIdx
        idxList.extend([artificialTypeIdx,artificialPropIdx,sampleFracIdx])

        varValue = params["preprocessing"][varNamesFull[varType]][varIdx]
        lifValue = params["preprocessing"]["lifThresh"][lifIdx]
        artificialType = params["stabl_general"]["artificalTypes"][artificialTypeIdx]
        artificialProp = params["stabl_general"]["artificalProportions"][artificialPropIdx]
        sampleFrac = params["stabl_general"]["sampleFractions"][sampleFracIdx]

        title = f"{modelName}-{varNames[varIdx]}{varValue}-lif{lifValue}-artType{artificialType}-artProp{artificialProp}-sampleFrac{sampleFrac}"

        preprocessing,model = generateModel(idxList,modelName,paramFilePath)
    
    preds, selectedFeats = single_omic_simple(
        data,
        y,
        outerSplitter,
        model,
        modelName,
        preprocessing,
        taskType,
    )
    preds.to_csv(Path(savePath,"preds-"+title+".csv"))
    selectedFeats.to_csv(Path(savePath,"selectedFeats-"+title+".csv"))







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int, default=0)
    args = parser.parse_args()
    run(args.idx)