import numpy as np
import pandas as pd
import json
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .preprocessing import LowInfoFilter
from .single_omic import single_omic_simple


from sklearn.model_selection import RepeatedStratifiedKFold, GroupShuffleSplit, GridSearchCV, RepeatedKFold
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from stabl.stabl import Stabl, group_bootstrap
from stabl.adaptive import ALogitLasso, ALasso
from groupyr import SGL, LogisticSGL
from sklearn.base import clone





task_type = "binary"

def generateModel(idxList,modelName,paramFilePath,chosen_inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)):
    params = json.loads(open(paramFilePath).read())
    useVarThresh = params["preprocessing"]["useVarThresh"]
    useVarPercentile = params["preprocessing"]["useVarPercentile"]
    useUniqueness = params["preprocessing"]["useUniqueness"]
    varThreshValue = params["preprocessing"]["varThreshValues"][idxList[0]]
    varPercentileValue= params["preprocessing"]["varPercentileValues"][idxList[1]]
    uniquenessPercentileValue = params["preprocessing"]["uniquenessPercentileValues"][idxList[2]]
    lifThresh = params["preprocessing"]["lifThresh"][idxList[3]]
    
    preprocessingList = []
    if useVarThresh:
        preprocessingList.append(("varianceThreshold",VarianceThreshold(varThreshValue)))
    # if useVarPercentile:
    #     preprocessingList.append(("variancePercentile",SelectPercentile(np.var,percentile=varPercentileValues[preprocIdx[1]])))
    # if useUniqueness:
    #     preprocessingList.append(())

    preprocessingList.append(("lif",LowInfoFilter(lifThresh)))
    
    preprocessingList.extend([("impute", SimpleImputer(strategy="median")),("std", StandardScaler())])
    preprocessing = Pipeline(steps=preprocessingList)

    
    if modelName == "lasso":
        a,b,c = params[modelName]["clogParams"]
        lasso = LogisticRegression(penalty="l1", class_weight="balanced", max_iter=int(1e6), solver="liblinear", random_state=42)
        model = GridSearchCV(lasso, param_grid={"C": np.logspace(a,b,c)}, scoring="roc_auc", cv=chosen_inner_cv, n_jobs=-1)
    
    if modelName == "alasso":
        a,b,c = params[modelName]["clogParams"]
        alasso = ALogitLasso(penalty="l1", solver="liblinear", max_iter=int(1e6), class_weight='balanced', random_state=42)
        model = GridSearchCV(alasso, scoring='roc_auc', param_grid={"C": np.logspace(a,b,c)}, cv=chosen_inner_cv, n_jobs=-1)

    if modelName == "en":
        a1,b1,c1 = params[modelName]["clogParams"]
        a2,b2,c2 = params[modelName]["l1ratiolinParams"]
        en = LogisticRegression(penalty='elasticnet',solver='saga',class_weight='balanced',max_iter=int(1e3),random_state=42)
        model = GridSearchCV(en, param_grid={"C": np.logspace(a1,b1,c1), "l1_ratio": np.linspace(a2,b2,c2)}, scoring="roc_auc", cv=chosen_inner_cv, n_jobs=-1)

    if modelName == "sgl":
        a1,b1,c1 = params[modelName]["alphalogParams"]
        a2,b2,c2 = params[modelName]["l1ratiolinParams"]
        sgl = LogisticSGL(max_iter=int(1e3), l1_ratio=0.5)
        model = GridSearchCV(sgl, scoring='roc_auc', param_grid={"alpha": np.logspace(a1,b1,c1), "l1_ratio": np.linspace(a2,b2,c2)}, cv=chosen_inner_cv, n_jobs=-1)

    if modelName[:5] == "stabl":
        nBootstraps = params["stabl_general"]["n_bootstraps"]
        replace = params["stabl_general"]["replace"]
        artificialType = params["stabl_general"]["artificialTypes"][idxList[4]]
        artificalProp = params["stabl_general"]["artificialProportions"][idxList[5]]
        sampleFraction = params["stabl_general"]["sampleFractions"][idxList[6]]
        a1,b1,c1 = params["stabl_general"]["fdrThreshParams"]

        if modelName == "stabl_lasso":
            a2,b2,c2 = params[modelName]["clogParams"]

            model = Stabl(
                    LogisticRegression(penalty="l1", class_weight="balanced",
                                        max_iter=int(1e6), solver="liblinear", random_state=42),
                    n_bootstraps=nBootstraps,
                    artificial_type=artificialType,
                    artificial_proportion=artificalProp,
                    replace=replace,
                    fdr_threshold_range=np.arange(a1,b1,c1),
                    sample_fraction=sampleFraction,
                    random_state=42,
                    lambda_grid={"C": np.logspace(a2, b2, c2)},
                    verbose=1
                )
        
        if modelName == "stabl_alasso":
            a2,b2,c2 = params[modelName]["clogParams"]

            model = Stabl(
                    ALogitLasso(penalty="l1", solver="liblinear", 
                            max_iter=int(1e6), class_weight='balanced', random_state=42),
                    n_bootstraps=nBootstraps,
                    artificial_type=artificialType,
                    artificial_proportion=artificalProp,
                    replace=replace,
                    fdr_threshold_range=np.arange(a1,b1,c1),
                    sample_fraction=sampleFraction,
                    random_state=42,
                    lambda_grid={"C": np.logspace(a2, b2, c2)},
                    verbose=1
                )
        
        if modelName == "stabl_en":
            a2,b2,c2 = params[modelName]["clogParams"]
            a3,b3,c3 = params[modelName]["l1ratiolinParams"]
            l1params = np.linspace(a3,b3,c3)
            lambdaGrid = [ {"C": np.logspace(a2,b2,c3),"l1_ratio": [v]} for v in l1params]

            model = Stabl(
                    LogisticRegression(penalty='elasticnet',solver='saga',
                                        class_weight='balanced',max_iter=int(1e3),random_state=42),
                    n_bootstraps=nBootstraps,
                    artificial_type=artificialType,
                    artificial_proportion=artificalProp,
                    replace=replace,
                    fdr_threshold_range=np.arange(a1,b1,c1),
                    sample_fraction=sampleFraction,
                    random_state=42,
                    lambda_grid=lambdaGrid,
                    verbose=1
                )
            
        if modelName == "stabl_sgl":
            a2,b2,c2 = params[modelName]["alphalogParams"]
            a3,b3,c3 = params[modelName]["l1ratiolinParams"]
            l1params = np.linspace(a3,b3,c3)
            lambdaGrid = [ {"alpha": np.logspace(a2,b2,c3),"l1_ratio": [v]} for v in l1params]

            model = Stabl(
                    LogisticSGL(max_iter=int(1e3), l1_ratio=0.5),
                    n_bootstraps=nBootstraps,
                    artificial_type=artificialType,
                    artificial_proportion=artificalProp,
                    replace=replace,
                    fdr_threshold_range=np.arange(a1,b1,c1),
                    sample_fraction=sampleFraction,
                    random_state=42,
                    lambda_grid=lambdaGrid,
                    perc_corr_group_threshold=params[modelName]["corrValues"][idxList[7]],
                    verbose=1
                )
    
    return preprocessing,model
            
        