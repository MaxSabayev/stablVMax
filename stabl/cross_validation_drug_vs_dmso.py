from .unionfind import UnionFind
import sys
from tqdm.autonotebook import tqdm
from .pipelines_utils import save_plots, compute_scores_table
from .preprocessing import remove_low_info_samples, LowInfoFilter
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, GroupShuffleSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn import clone
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import os
import numpy as np
import shap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
import pandas as pd
import warnings
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', category=ConvergenceWarning)
ConvergenceWarning('ignore')
import collections
outter_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
from collections import defaultdict
inner_reg_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
inner_group_cv = GroupShuffleSplit(n_splits=25, test_size=0.2, random_state=42)

nb_param = 50

logit = LogisticRegression(penalty=None, class_weight="balanced", max_iter=int(1e6), random_state=42)
linreg = LinearRegression()
randomforest=RandomForestRegressor(n_estimators=200, max_depth=5)
xgboost = XGBRegressor()

preprocessing = Pipeline(
    steps=[
        ("variance", VarianceThreshold(0.01)),
        ("lif", LowInfoFilter()),
        ("impute", SimpleImputer(strategy="median")),
        ("std", StandardScaler())
    ]
)


def _make_groups(X, percentile):
    n = X.shape[1]
    u = UnionFind(elements=range(n))
    corr_mat = pd.DataFrame(X).corr().values
    corr_val = corr_mat[np.triu_indices_from(corr_mat, k=1)]
    threshold = np.percentile(corr_val, percentile)
    for i in np.arange(n):
        for j in np.arange(n):
            if abs(corr_mat[i, j]) > threshold:
                u.union(i, j)
    res = list(map(list, u.components()))
    res = list(map(np.array, res))
    return res


@ignore_warnings(category=ConvergenceWarning)
def cv_on_existing_feats(
        data_dict,
        y,
        outer_splitter,
        estimators,
        task_type,
        model_chosen,
        models,
        fold_feats_path,
        save_path=None,
        outer_groups=None,
        early_fusion=False,
        late_fusion=False,
        n_iter_lf=10000,
):
    randomforest=RandomForestRegressor(n_estimators=200, max_depth=5)
    xgboost = XGBRegressor()
    if 'xgboost' in estimators.keys():
        xgboost=estimators["xgboost"]
    if 'random_forest' in estimators.keys():
        randomforest=estimators["random_forest"]
        
    stabl = estimators["stabl_lasso"]
    stabl_alasso = estimators["stabl_alasso"]
    stabl_en = estimators["stabl_en"]

    os.makedirs(Path(save_path, "Training CV"), exist_ok=True)
    os.makedirs(Path(save_path, "Summary"), exist_ok=True)

    # Initializing the df containing the data of all omics
    X_tot = pd.concat(data_dict.values(), axis="columns")

    predictions_dict = dict()
    selected_features_dict = dict()
    stabl_features_dict = dict()

    for model in models:
        predictions_dict[model] = pd.DataFrame(data=None, index=y.index)
        selected_features_dict[model] = []
        stabl_features_dict[model] = dict()
        for omic_name in data_dict.keys():
            if "STABL" in model:
                stabl_features_dict[model][omic_name] = pd.DataFrame(data=None, columns=["Threshold", "min FDP+"])
                
    selected_features_from_file = dict()
    for model in models:
        if "STABL" in model:
            df = pd.read_csv(Path(fold_feats_path, "Training CV", f"Selected Features {model}.csv"), index_col=0)
            #df = pd.read_csv(Path(fold_feats_path, f"Selected Features STABL ALasso (CellOT just stims).csv"), index_col=0)
            selected_features_from_file[model] = [
                eval(features) if isinstance(features, str) else features
                for features in df["Fold selected features"]
            ]
    k = 1
    #path2="/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/Preprocessed_OOL_Clinical.csv"
    #true_data=pd.read_csv(path2)
    #true_data.set_index("ID", inplace=True)
    for train, test in (tbar := tqdm(
            outer_splitter.split(X_tot, y, groups=outer_groups),
            total=outer_splitter.get_n_splits(X=X_tot, y=y, groups=outer_groups),
            file=sys.stdout
    )):
        train_idx, test_idx = y.iloc[train].index, y.iloc[test].index
        groups = outer_groups.loc[train_idx].values if outer_groups is not None else None

        predictions_dict_late_fusion = dict()
        fold_selected_features = dict()
        for model in models:
            fold_selected_features[model] = []
            if "STABL" not in model and "EF" not in model:
                predictions_dict_late_fusion[model] = dict()
                for omic_name in data_dict.keys():
                    predictions_dict_late_fusion[model][omic_name] = pd.DataFrame(data=None, index=test_idx)

        tbar.set_description(f"{len(train_idx)} train samples, {len(test_idx)} test samples")

        for model in models:
            if "STABL" in model:
                fold_selected_features[model] = selected_features_from_file[model][k - 1]
        
        
        #clinical_vars = ["EGA"] # ["EGA", "Age", "BMI"]
        for model in filter(lambda x: "STABL" in x, models):
            X_train = X_tot.loc[train_idx, fold_selected_features[model]]
            X_test = X_tot.loc[test_idx, fold_selected_features[model]]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]

            if len(fold_selected_features[model]) > 0:
                # Standardization
                std_pipe = Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy="median")),
                        ('std', StandardScaler())
                    ]
                )

                X_train = pd.DataFrame(
                    data=std_pipe.fit_transform(X_train),
                    index=X_train.index,
                    columns=X_train.columns
                )
                X_test = pd.DataFrame(
                    data=std_pipe.transform(X_test),
                    index=X_test.index,
                    columns=X_test.columns
                )
                '''
                clinical_train = true_data.loc[X_train.index, clinical_vars]
                clinical_train = pd.DataFrame(
                    SimpleImputer(strategy="median").fit_transform(clinical_train),
                    index=clinical_train.index,
                    columns=clinical_vars
                )
                clinical_train = pd.DataFrame(
                    StandardScaler().fit_transform(clinical_train),
                    index=clinical_train.index,
                    columns=clinical_vars
                )
                X_train = pd.concat([X_train, clinical_train], axis=1)
                
                clinical_test = true_data.loc[X_test.index, clinical_vars]
                clinical_test = pd.DataFrame(
                    SimpleImputer(strategy="median").fit_transform(clinical_test),
                    index=clinical_test.index,
                    columns=clinical_vars
                )
                clinical_test = pd.DataFrame(
                    StandardScaler().fit_transform(clinical_test),
                    index=clinical_test.index,
                    columns=clinical_vars
                )
                X_test = pd.concat([X_test, clinical_test], axis=1)
                '''

                # __Final Models__
                if task_type == "binary":
                    predictions = clone(logit).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()

                elif task_type == "regression":
                    if model_chosen=='linear_reg':
                        predictions = clone(linreg).fit(X_train, y_train).predict(X_test)
                    elif model_chosen=='random_forest':
                        predictions = clone(randomforest).fit(X_train, y_train).predict(X_test)
                    elif model_chosen=='xgboost':
                        predictions = clone(xgboost).fit(X_train, y_train).predict(X_test)
                else:
                    raise ValueError("task_type not recognized.")

                predictions_dict[model].loc[test_idx, f"Fold n°{k}"] = predictions

            else:
                print("No feature selected for this fold")
                if task_type == "binary":
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [0.5] * len(test_idx)
                elif task_type == "regression":
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [np.mean(y_train)] * len(test_idx)
                else:
                    raise ValueError("task_type not recognized.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for model in models:
            print(f"This fold: {len(fold_selected_features[model])} features selected for {model}")
            selected_features_dict[model].append(fold_selected_features[model])
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        k += 1

    # __SAVING_RESULTS__
    print("Saving results...")
    if y.name is None:
        y.name = "outcome"

    summary_res_path = Path(save_path, "Summary")
    cv_res_path = Path(save_path, "Training CV")

    formatted_features_dict = dict()

    for model in models:
        formatted_features_dict[model] = pd.DataFrame(
            data={
                "Fold selected features": selected_features_dict[model],
                "Fold nb of features": [len(el) for el in selected_features_dict[model]]
            },
            index=[f"Fold {i}" for i in range(outer_splitter.get_n_splits(X=X_tot, groups=outer_groups))]
        )
        formatted_features_dict[model].to_csv(Path(cv_res_path, f"Selected Features {model}.csv"))
        if "STABL" in model:
            for omic_name, val in stabl_features_dict[model].items():
                os.makedirs(Path(cv_res_path, f"Stabl features {model}"), exist_ok=True)
                val.to_csv(Path(cv_res_path, f"Stabl features {model}", f"Stabl features {model} {omic_name}.csv"))

    predictions_dict = {model: predictions_dict[model].median(axis=1) for model in predictions_dict.keys()}

    table_of_scores = compute_scores_table(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        selected_features_dict=formatted_features_dict
    )

    table_of_scores.to_csv(Path(summary_res_path, "Scores training CV.csv"))
    table_of_scores.to_csv(Path(cv_res_path, "Scores training CV.csv"))
    save_plots(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        save_path=cv_res_path
    )

    return predictions_dict
    
def cv_drug_vs_dmso_ega(
        data_dict,
        y,
        outer_splitter,
        estimators,
        task_type,
        model_chosen,
        models,
        fold_feats_path,
        save_path=None,
        outer_groups=None,
        early_fusion=False,
        late_fusion=False,
        n_iter_lf=10000,
):
    randomforest=RandomForestRegressor(n_estimators=200, max_depth=5)
    xgboost = XGBRegressor()
    if 'xgboost' in estimators.keys():
        xgboost=estimators["xgboost"]
    if 'random_forest' in estimators.keys():
        randomforest=estimators["random_forest"]
        
    stabl = estimators["stabl_lasso"]
    stabl_alasso = estimators["stabl_alasso"]
    stabl_en = estimators["stabl_en"]

    os.makedirs(Path(save_path, "Training CV"), exist_ok=True)
    os.makedirs(Path(save_path, "Summary"), exist_ok=True)

    # Initializing the df containing the data of all omics
    no_treat_dict=data_dict['No_treat']
    X_tot = pd.concat(no_treat_dict.values(), axis="columns")
    for drug in data_dict.keys():
        if drug!='No_treat':
            data_name = f"X_tot_{drug}"
            globals()[data_name] = pd.concat(data_dict[drug].values(), axis="columns")
    predictions_dict = dict()
    selected_features_dict = dict()
    stabl_features_dict = dict()

    for model in models:
        predictions_dict[model] = {}
        selected_features_dict[model] = []
        stabl_features_dict[model] = dict()
        for omic_name in no_treat_dict.keys():
            if "STABL" in model:
                stabl_features_dict[model][omic_name] = pd.DataFrame(data=None, columns=["Threshold", "min FDP+"])
        for drug in data_dict.keys():
            predictions_dict[model][drug]=pd.DataFrame(data=None, index=y.index)
            
    selected_features_from_file = dict()
    for model in models:
        if "STABL" in model:
            df = pd.read_csv(Path(fold_feats_path, "Training CV", f"Selected Features {model}.csv"), index_col=0)
            selected_features_from_file[model] = [
                eval(features) if isinstance(features, str) else features
                for features in df["Fold selected features"]
            ]
    k = 1
    path2="/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/Preprocessed_OOL_Clinical.csv"
    true_data=pd.read_csv(path2)
    true_data.set_index("ID", inplace=True)    
    for train, test in (tbar := tqdm(
            outer_splitter.split(X_tot, y, groups=outer_groups),
            total=outer_splitter.get_n_splits(X=X_tot, y=y, groups=outer_groups),
            file=sys.stdout
    )):
        train_idx, test_idx = y.iloc[train].index, y.iloc[test].index
        groups = outer_groups.loc[train_idx].values if outer_groups is not None else None

        predictions_dict_late_fusion = dict()
        fold_selected_features = dict()
        for model in models:
            fold_selected_features[model] = []
            if "STABL" not in model and "EF" not in model:
                predictions_dict_late_fusion[model] = dict()
                for omic_name in no_treat_dict.keys():
                    predictions_dict_late_fusion[model][omic_name] = pd.DataFrame(data=None, index=test_idx)

        tbar.set_description(f"{len(train_idx)} train samples, {len(test_idx)} test samples")

        for model in models:
            if "STABL" in model:
                fold_selected_features[model] = selected_features_from_file[model][k - 1]

        for model in filter(lambda x: "STABL" in x, models):
            X_train = X_tot.loc[train_idx, fold_selected_features[model]]
            X_test = X_tot.loc[test_idx, fold_selected_features[model]]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            for drug in data_dict.keys():
                if drug != 'No_treat':
                    data_name = f"X_tot_{drug}"
                    data_name_test = f"X_test_{drug}"
                    globals()[data_name_test] = globals()[data_name].loc[test_idx, fold_selected_features[model]]
            
            clinical_vars = ["EGA"]  # ou ["EGA", "Age", "BMI"]
        
            clinical_train = true_data.loc[train_idx, clinical_vars]
            clinical_test = true_data.loc[test_idx, clinical_vars]
        
            # Concatène les colonnes cliniques à X_train / X_test (non standardisées pour l’instant)
            X_train = pd.concat([X_train, clinical_train], axis=1)
            X_test = pd.concat([X_test, clinical_test], axis=1)
        
            for drug in data_dict.keys():
                if drug != 'No_treat':
                    data_name_test = f"X_test_{drug}"
                    globals()[data_name_test] = pd.concat(
                        [globals()[data_name_test], clinical_test], axis=1
                    )
        
            if len(fold_selected_features[model]) > 0:
                # --- Standardisation commune ---
                std_pipe = Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy="median")),
                        ('std', StandardScaler())
                    ]
                )
        
                X_train = pd.DataFrame(
                    data=std_pipe.fit_transform(X_train),
                    index=X_train.index,
                    columns=X_train.columns
                )
                X_test = pd.DataFrame(
                    data=std_pipe.transform(X_test),
                    index=X_test.index,
                    columns=X_test.columns
                )
                for drug in data_dict.keys():
                    if drug != 'No_treat':
                        data_name_test = f"X_test_{drug}"
                        globals()[data_name_test] = pd.DataFrame(
                            data=std_pipe.transform(globals()[data_name_test]),
                            index=globals()[data_name_test].index,
                            columns=globals()[data_name_test].columns
                        )
                if task_type == "binary":
                    predictions = clone(logit).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()

                elif task_type == "regression":
                    if model_chosen=='linear_reg':
                        fitted_model = clone(linreg).fit(X_train, y_train)
                    elif model_chosen=='random_forest':
                        fitted_model = clone(randomforest).fit(X_train, y_train)
                    elif model_chosen=='xgboost':
                        fitted_model = clone(xgboost).fit(X_train, y_train)
                    predictions={}
                    predictions['No_treat']=fitted_model.predict(X_test)
                    for drug in data_dict.keys():
                        if drug!='No_treat':
                            data_name_test = f"X_test_{drug}"
                            predictions[drug]=fitted_model.predict(globals()[data_name_test])
                else:
                    raise ValueError("task_type not recognized.")
                
                #predictions_dict[model].loc[test_idx, f"Fold n°{k}"] = predictions
                for key in predictions.keys():
                    predictions_dict[model][key].loc[test_idx, f"Fold n°{k}"] = predictions[key]

            else:
                print("No feature selected for this fold")
                if task_type == "binary":
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [0.5] * len(test_idx)
                elif task_type == "regression":
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [np.mean(y_train)] * len(test_idx)
                else:
                    raise ValueError("task_type not recognized.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for model in models:
            print(f"This fold: {len(fold_selected_features[model])} features selected for {model}")
            selected_features_dict[model].append(fold_selected_features[model])
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        k += 1

    # __SAVING_RESULTS__
    print("Saving results...")
    if y.name is None:
        y.name = "outcome"

    summary_res_path = Path(save_path, "Summary")
    cv_res_path = Path(save_path, "Training CV")

    formatted_features_dict = dict()

    for model in models:
        formatted_features_dict[model] = pd.DataFrame(
            data={
                "Fold selected features": selected_features_dict[model],
                "Fold nb of features": [len(el) for el in selected_features_dict[model]]
            },
            index=[f"Fold {i}" for i in range(outer_splitter.get_n_splits(X=X_tot, groups=outer_groups))]
        )
        formatted_features_dict[model].to_csv(Path(cv_res_path, f"Selected Features {model}.csv"))
        if "STABL" in model:
            for omic_name, val in stabl_features_dict[model].items():
                os.makedirs(Path(cv_res_path, f"Stabl features {model}"), exist_ok=True)
                val.to_csv(Path(cv_res_path, f"Stabl features {model}", f"Stabl features {model} {omic_name}.csv"))

    predictions_dict = {model: {drug: predictions_dict[model][drug].median(axis=1) for drug in predictions_dict[model].keys()} for model in predictions_dict.keys()}
    all_prediction_dicts = {}
    for drug in data_dict.keys():
        prediction_dict_drug={model: predictions_dict[model][drug] for model in predictions_dict.keys()}
        path_plots=os.path.join(cv_res_path, drug)
        save_plots(
            predictions_dict=prediction_dict_drug,
            y=y,
            task_type=task_type,
            save_path=path_plots
        )
        all_prediction_dicts[drug] = prediction_dict_drug

    return all_prediction_dicts
def cv_drug_vs_dmso(
        data_dict,
        y,
        outer_splitter,
        estimators,
        task_type,
        model_chosen,
        models,
        fold_feats_path,
        save_path=None,
        outer_groups=None,
        early_fusion=False,
        late_fusion=False,
        n_iter_lf=10000,
):
    randomforest=RandomForestRegressor(n_estimators=200, max_depth=5)
    xgboost = XGBRegressor()
    if 'xgboost' in estimators.keys():
        xgboost=estimators["xgboost"]
    if 'random_forest' in estimators.keys():
        randomforest=estimators["random_forest"]
        
    stabl = estimators["stabl_lasso"]
    stabl_alasso = estimators["stabl_alasso"]
    stabl_en = estimators["stabl_en"]

    os.makedirs(Path(save_path, "Training CV"), exist_ok=True)
    os.makedirs(Path(save_path, "Summary"), exist_ok=True)

    # Initializing the df containing the data of all omics
    no_treat_dict=data_dict['No_treat']
    X_tot = pd.concat(no_treat_dict.values(), axis="columns")
    for drug in data_dict.keys():
        if drug!='No_treat':
            data_name = f"X_tot_{drug}"
            globals()[data_name] = pd.concat(data_dict[drug].values(), axis="columns")
    predictions_dict = dict()
    selected_features_dict = dict()
    stabl_features_dict = dict()

    for model in models:
        predictions_dict[model] = {}
        selected_features_dict[model] = []
        stabl_features_dict[model] = dict()
        for omic_name in no_treat_dict.keys():
            if "STABL" in model:
                stabl_features_dict[model][omic_name] = pd.DataFrame(data=None, columns=["Threshold", "min FDP+"])
        for drug in data_dict.keys():
            predictions_dict[model][drug]=pd.DataFrame(data=None, index=y.index)
            
    selected_features_from_file = dict()
    for model in models:
        if "STABL" in model:
            df = pd.read_csv(Path(fold_feats_path, "Training CV", f"Selected Features {model}.csv"), index_col=0)
            selected_features_from_file[model] = [
                eval(features) if isinstance(features, str) else features
                for features in df["Fold selected features"]
            ]
    k = 1
    xgb_importances_per_fold = []
    shap_importances_per_fold = []
    shap_values_all_folds=[]
    for train, test in (tbar := tqdm(
            outer_splitter.split(X_tot, y, groups=outer_groups),
            total=outer_splitter.get_n_splits(X=X_tot, y=y, groups=outer_groups),
            file=sys.stdout
    )):
        train_idx, test_idx = y.iloc[train].index, y.iloc[test].index
        groups = outer_groups.loc[train_idx].values if outer_groups is not None else None

        predictions_dict_late_fusion = dict()
        fold_selected_features = dict()
        for model in models:
            fold_selected_features[model] = []
            if "STABL" not in model and "EF" not in model:
                predictions_dict_late_fusion[model] = dict()
                for omic_name in no_treat_dict.keys():
                    predictions_dict_late_fusion[model][omic_name] = pd.DataFrame(data=None, index=test_idx)

        tbar.set_description(f"{len(train_idx)} train samples, {len(test_idx)} test samples")

        for model in models:
            if "STABL" in model:
                fold_selected_features[model] = selected_features_from_file[model][k - 1]

        for model in filter(lambda x: "STABL" in x, models):
            X_train = X_tot.loc[train_idx, fold_selected_features[model]]
            X_test = X_tot.loc[test_idx, fold_selected_features[model]]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            for drug in data_dict.keys():
                if drug!='No_treat':
                    data_name = f"X_tot_{drug}"
                    data_name_test = f"X_test_{drug}"
                    globals()[data_name_test] = globals()[data_name].loc[test_idx, fold_selected_features[model]]
            if len(fold_selected_features[model]) > 0:
                # Standardization
                std_pipe = Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy="median")),
                        ('std', StandardScaler())
                    ]
                )

                X_train = pd.DataFrame(
                    data=std_pipe.fit_transform(X_train),
                    index=X_train.index,
                    columns=X_train.columns
                )
                X_test = pd.DataFrame(
                    data=std_pipe.transform(X_test),
                    index=X_test.index,
                    columns=X_test.columns
                )
                for drug in data_dict.keys():
                    if drug!='No_treat':
                        data_name_test = f"X_test_{drug}"
                        globals()[data_name_test] = pd.DataFrame(
                            data=std_pipe.transform(globals()[data_name_test]),
                            index=globals()[data_name_test].index,
                            columns=globals()[data_name_test].columns
                        )
                # __Final Models__
                if task_type == "binary":
                    predictions = clone(logit).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()

                elif task_type == "regression":
                    if model_chosen=='linear_reg':
                        fitted_model = clone(linreg).fit(X_train, y_train)
                    elif model_chosen=='random_forest':
                        fitted_model = clone(randomforest).fit(X_train, y_train)
                    elif model_chosen=='xgboost':
                        fitted_model = clone(xgboost).fit(X_train, y_train)
                        
                        # XGBoost importance
                        importances = fitted_model.get_booster().get_score(importance_type='weight')
                        xgb_importances_per_fold.append(importances)
                        
                        # SHAP importance
                        explainer = shap.Explainer(fitted_model, X_train)
                        shap_values = explainer(X_test)
                        shap_df = pd.DataFrame(np.abs(shap_values.values), columns=X_test.columns)
                        shap_importances_per_fold.append(shap_df.mean(axis=0).to_dict())
                        shap_values_all_folds.append(pd.DataFrame(
                            data=shap_values.values,
                            columns=X_test.columns,
                            index=X_test.index
                        ))
                    predictions={}
                    predictions['No_treat']=fitted_model.predict(X_test)
                    for drug in data_dict.keys():
                        if drug!='No_treat':
                            data_name_test = f"X_test_{drug}"
                            predictions[drug]=fitted_model.predict(globals()[data_name_test])
                else:
                    raise ValueError("task_type not recognized.")
                
                for key in predictions.keys():
                    predictions_dict[model][key].loc[test_idx, f"Fold n°{k}"] = predictions[key]

            else:
                print("No feature selected for this fold")
                if task_type == "binary":
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [0.5] * len(test_idx)
                elif task_type == "regression":
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [np.mean(y_train)] * len(test_idx)
                else:
                    raise ValueError("task_type not recognized.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for model in models:
            print(f"This fold: {len(fold_selected_features[model])} features selected for {model}")
            selected_features_dict[model].append(fold_selected_features[model])
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        k += 1
    print("Saving results...")
    if y.name is None:
        y.name = "outcome"
    
    summary_res_path = Path(save_path, "Summary")
    cv_res_path = Path(save_path, "Training CV")
    os.makedirs(summary_res_path, exist_ok=True)
    os.makedirs(cv_res_path, exist_ok=True)
    
    formatted_features_dict = dict()
    for model in models:
        formatted_features_dict[model] = pd.DataFrame(
            data={
                "Fold selected features": selected_features_dict[model],
                "Fold nb of features": [len(el) for el in selected_features_dict[model]]
            },
            index=[f"Fold {i}" for i in range(outer_splitter.get_n_splits(X=X_tot, groups=outer_groups))]
        )
        formatted_features_dict[model].to_csv(Path(cv_res_path, f"Selected Features {model}.csv"))
        if "STABL" in model:
            for omic_name, val in stabl_features_dict[model].items():
                stabl_dir = Path(cv_res_path, f"Stabl features {model}")
                stabl_dir.mkdir(parents=True, exist_ok=True)
                val.to_csv(stabl_dir / f"Stabl features {model} {omic_name}.csv")
    
    
    predictions_dict = {
        model: {
            drug: df.median(axis=1)
            for drug, df in predictions_dict[model].items()
        }
        for model in predictions_dict
    }
    
    all_prediction_dicts = {}
    for drug in data_dict:
        prediction_dict_drug = {
            model: predictions_dict[model][drug] for model in predictions_dict
        }
        path_plots = os.path.join(cv_res_path, drug)
        save_plots(predictions_dict=prediction_dict_drug, y=y, task_type=task_type, save_path=path_plots)
        all_prediction_dicts[drug] = prediction_dict_drug
    
    def aggregate_importances(importances_list):
        total = defaultdict(float)
        for imp in importances_list:
            for k, v in imp.items():
                total[k] += v
        return pd.Series({k: v / len(importances_list) for k, v in total.items()}).sort_values(ascending=False)
    
    xgb_avg_importance = aggregate_importances(xgb_importances_per_fold)
    shap_avg_importance = aggregate_importances(shap_importances_per_fold)
    xgb_avg_importance.to_csv(summary_res_path / "xgb_mean_importance.csv")
    shap_avg_importance.to_csv(summary_res_path / "shap_mean_importance.csv")
    
    # === Barplot XGBoost vs SHAP ===
    top_k = 20
    common_features = list(set(xgb_avg_importance.head(top_k).index) | set(shap_avg_importance.head(top_k).index))
    df_imp = pd.DataFrame({
        "XGBoost": xgb_avg_importance[common_features],
        "SHAP": shap_avg_importance[common_features]
    }).fillna(0).sort_values(by="SHAP", ascending=False)
    df_imp.plot(kind="barh", figsize=(10, 8))
    plt.title("Top Feature Importances - SHAP vs XGBoost (mean over folds)")
    plt.tight_layout()
    plt.savefig(summary_res_path / "combined_importance_barplot.pdf")
    plt.close()
    
    # === SHAP summary plot global ===
    top_k = 10
    top_features = shap_avg_importance.head(top_k).index.tolist()
    shap_values_all = pd.concat(shap_values_all_folds)[top_features]
    X_all = X_tot.loc[shap_values_all.index, top_features]

    shap.summary_plot(
        shap_values_all.values,
        features=X_all,
        feature_names=X_all.columns,
        show=False,
        plot_type="dot"
    )
    plt.tight_layout()
    plt.savefig(summary_res_path / "shap_summary_plot_global.pdf")
    plt.close()
    
    # === Top SHAP features ===
    

    # === Standardisation des top features ===
    #X_test_No_treat = X_tot.loc[:, top_features]
    X_test_No_treat = X_tot.loc[shap_values_all.index, top_features]
    X_test_per_drug = {}
    
    '''
    for drug in data_dict:
        if drug == 'No_treat':
            continue
        data_name = f"X_tot_{drug}"
        X_test_per_drug[drug] = globals()[data_name].loc[:, top_features]
    '''
    for drug in data_dict:
        if drug == 'No_treat':
            continue
        data_name = f"X_tot_{drug}"
        X_test_per_drug[drug] = globals()[data_name].loc[shap_values_all.index, top_features]

    std_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std', StandardScaler())
    ])
    X_test_No_treat = pd.DataFrame(
        std_pipe.fit_transform(X_test_No_treat),
        index=X_test_No_treat.index,
        columns=top_features
    )
    
    for drug in X_test_per_drug:
        X_test_per_drug[drug] = pd.DataFrame(
            std_pipe.transform(X_test_per_drug[drug]),
            index=X_test_per_drug[drug].index,
            columns=top_features
        )
    
    # === Heatmap du delta moyen par feature ===
    feature_deltas = pd.DataFrame(index=top_features)
    
    for drug in X_test_per_drug:
        X_drug = X_test_per_drug[drug]
        X_notreat = X_test_No_treat
        common_idx = X_drug.index.intersection(X_notreat.index)
        delta = (X_drug.loc[common_idx] - X_notreat.loc[common_idx]).mean(axis=0)
        feature_deltas[drug] = delta
    feature_deltas = feature_deltas.loc[top_features]
    plt.figure(figsize=(10, 6))
    sns.heatmap(feature_deltas, annot=True, cmap="RdBu_r", center=0, linewidths=0.5, fmt=".2f")
    plt.title("Average Δ feature values (Drug - No_treat)\nTop SHAP features only")
    plt.ylabel("Features")
    plt.xlabel("Drug")
    plt.tight_layout()
    plt.savefig(summary_res_path / "heatmap_top_features_deltas.pdf")
    plt.close()
    
    return all_prediction_dicts
    