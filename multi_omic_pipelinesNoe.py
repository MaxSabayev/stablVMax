from unionfind import UnionFind
import sys
from tqdm.autonotebook import tqdm
from itertools import combinations
from pipelines_utils import save_plots, compute_scores_table, compute_pvalues_table
from stacked_generalization import stacked_multi_omic
from metrics import jaccard_matrix
from stabl import save_stabl_results
from preprocessing import remove_low_info_samples, LowInfoFilter
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, GroupShuffleSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBRegressor,XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn import clone
from pathlib import Path
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', category=ConvergenceWarning)
ConvergenceWarning('ignore')

outter_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

inner_reg_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
inner_group_cv = GroupShuffleSplit(n_splits=25, test_size=0.2, random_state=42)

nb_param = 50

logit = LogisticRegression(penalty=None, class_weight="balanced", max_iter=int(1e6), random_state=42)
linreg = LinearRegression()
randomforest=RandomForestRegressor(n_estimators=200, max_depth=5, n_jobs=1)
xgboost = XGBRegressor(n_jobs=1)
xgboost_class=XGBClassifier(n_jobs=1)
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
def multi_omic_stabl_cv(
        data_dict,
        y,
        outer_splitter,
        estimators,
        task_type,
        model_chosen,
        models,
        save_path=None,
        outer_groups=None,
        early_fusion=False,
        late_fusion=False,
        n_iter_lf=10000,
):

    """
    Performs a cross validation on the data_dict using the models and saves the results in save_path.

    Parameters
    ----------
    data_dict: dict
        Dictionary containing the input omic-files. the input omic-files should be pandas DataFrames

    y: pd.Series
        pandas Series containing the outcomes for the use case. Note that y should contain the union of outcomes for
        the data_dict.

    outer_splitter: sklearn.model_selection._split.BaseCrossValidator
        Outer cross validation splitter

    estimators : dict[str, sklearn estimator]
        Dict of feature selectors to benchmark with their name as key. They must implement `fit`,
        `get_support(indices=True)`.
        It should contain :
        - "lasso" : Lasso in GridSearchCV
        - "alasso" : ALasso in GridSearchCV
        - "en" : ElasticNet in GridSearchCV
        - "sgl" : SGL in GridSearchCV
        - "stabl_lasso" : Stabl with Lasso as base estimator
        - "stabl_alasso" : Stabl with ALasso as base estimator
        - "stabl_en" : Stabl with ElasticNet as base estimator
        - "stabl_sgl" : Stabl with SGL as base estimator

    task_type: str
        Can either be "binary" for binary classification or "regression" for regression tasks.

    save_path: Path or str
        Where to save the results

    models: list of str
        List of models to use. Can contain :
        - "Lasso" : Lasso
        - "STABL Lasso" : Stabl with Lasso as base estimator
        - "ALasso" : ALasso
        - "STABL ALasso" : Stabl with ALasso as base estimator
        - "ElasticNet" : ElasticNet
        - "STABL ElasticNet" : Stabl with ElasticNet as base estimator

    outer_groups: pd.Series, default=None
        If used, should be the same size as y and should indicate the groups of the samples.

    early_fusion: bool, default=False
        If True, it will perform early fusion for each estimator.

    late_fusion: bool, default=true
        If True, it will perform late fusion for each estimator.

    n_iter_lf: int, default=10000
        Number of iterations for the late fusion.


    Returns
    -------
    predictions_dict: dict
        Dictionary containing the predictions of each model for each sample in Cross-Validation.
    """
    if early_fusion:
        models += ["EF " + model for model in models if "STABL" not in model]

    lasso = estimators["lasso"]
    alasso = estimators["alasso"]
    en = estimators["en"]

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

    k = 1
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

        for omic_name, X_omic in data_dict.items():
            test_idx_tmp = X_omic.index.intersection(test_idx)
            X_tmp = X_omic.drop(index=test_idx, errors="ignore")
            X_test_tmp = X_omic.loc[test_idx_tmp]
            # Preprocessing of X_tmp
            X_tmp = remove_low_info_samples(X_tmp)
            ###
            common_idx = X_tmp.index.intersection(y.index)
            X_tmp = X_tmp.loc[common_idx]
            y_tmp = y.loc[common_idx]
            
            groups = outer_groups.loc[common_idx] if outer_groups is not None else None
            
            #y_tmp = y.loc[X_tmp.index]
           

            groups = outer_groups[X_tmp.index] if outer_groups is not None else None

            X_tmp_std = pd.DataFrame(
                data=preprocessing.fit_transform(X_tmp),
                index=X_tmp.index,
                columns=preprocessing.get_feature_names_out()
            )

            X_test_tmp_std = pd.DataFrame(
                data=preprocessing.transform(X_test_tmp),
                index=X_test_tmp.index,
                columns=preprocessing.get_feature_names_out()
            )
            assert X_tmp_std.shape[0] == y_tmp.shape[0], f"Mismatch after cleaning and transform: X={X_tmp_std.shape}, y={y_tmp.shape}"
            print(X_tmp_std.shape)
            # __STABL__
            if "STABL Lasso" in models:
                # fit STABL Lasso
                print("Fitting of STABL Lasso")
                stabl.fit(X_tmp_std, y_tmp, groups=groups)
                tmp_sel_features = list(stabl.get_feature_names_out())
                fold_selected_features["STABL Lasso"].extend(tmp_sel_features)
                print(
                    f"STABL Lasso finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL Lasso"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl.min_fdr_
                stabl_features_dict["STABL Lasso"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl,
                        path=Path(save_path, "Training CV", f"STABL Lasso results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type
                    )

            if "STABL ALasso" in models:
                # fit STABL ALasso
                print("Fitting of STABL ALasso")
                stabl_alasso.fit(X_tmp_std, y_tmp, groups=groups)
                tmp_sel_features = list(stabl_alasso.get_feature_names_out())
                fold_selected_features["STABL ALasso"].extend(tmp_sel_features)
                print(
                    f"STABL ALasso finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL ALasso"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_alasso.min_fdr_
                stabl_features_dict["STABL ALasso"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_alasso.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl_alasso,
                        path=Path(save_path, "Training CV", f"STABL ALasso results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type
                    )

            if "STABL ElasticNet" in models:
                # fit STABL ElasticNet
                print("Fitting of STABL ElasticNet")
                stabl_en.fit(X_tmp_std, y_tmp, groups=groups)
                tmp_sel_features = list(stabl_en.get_feature_names_out())
                fold_selected_features["STABL ElasticNet"].extend(tmp_sel_features)
                print(
                    f"STABL ElasticNet finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL ElasticNet"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_en.min_fdr_
                stabl_features_dict["STABL ElasticNet"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_en.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl_en,
                        path=Path(save_path, "Training CV", f"STABL ElasticNet results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type
                    )

            if "Lasso" in models:
                # __Lasso__
                print("Fitting of Lasso")
                model = clone(lasso)
                if task_type == "binary":
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std)
                tmp_sel_features = list(X_tmp_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                fold_selected_features["Lasso"].extend(tmp_sel_features)
                print(
                    f"Lasso finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict_late_fusion["Lasso"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

            if "ALasso" in models:
                # __ALasso__
                print("Fitting of ALasso")
                model = clone(alasso)
                if task_type == "binary":
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std)
                tmp_sel_features = list(X_tmp_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                fold_selected_features["ALasso"].extend(tmp_sel_features)
                print(
                    f"ALasso finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict_late_fusion["ALasso"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

            if "ElasticNet" in models:
                # __EN__
                print("Fitting of ElasticNet")
                model = clone(en)
                if task_type == "binary":
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std)
                tmp_sel_features = list(X_tmp_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                fold_selected_features["ElasticNet"].extend(tmp_sel_features)
                print(
                    f"ElasticNet finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict_late_fusion["ElasticNet"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

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

                # __Final Models__
                if task_type == "binary":
                    if model_chosen=='xgboost':
                        predictions = clone(xgboost_class).fit(X_train, y_train).predict(X_test)[:, 1].flatten()
                    else:
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
                if task_type == "binary":
                    #predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [0.5] * len(test_idx)
                    # Create the column if it doesn't exist yet
                    if f'Fold n°{k}' not in predictions_dict[model].columns:
                        predictions_dict[model][f'Fold n°{k}'] = np.nan
                    
                    # Then assign
                    print(len(test_idx))
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = 0.5


                elif task_type == "regression":
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [np.mean(y_train)] * len(test_idx)

                else:
                    raise ValueError("task_type not recognized.")
        # __late fusion__
        if late_fusion:
            preds_lf = late_fusion_cv(
                predictions_dict_late_fusion, y[test_idx], task_type,
                Path(save_path, "Training CV"), n_iter=n_iter_lf
            )
            for model in preds_lf:
                predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = preds_lf[model]

        # __EF Lasso__
        if early_fusion:
            X_train = X_tot.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X_tot.loc[test_idx]
            X_train_std = pd.DataFrame(
                data=preprocessing.fit_transform(X_train),
                columns=preprocessing.get_feature_names_out(),
                index=X_train.index
            )

            X_test_std = pd.DataFrame(
                data=preprocessing.transform(X_test),
                columns=preprocessing.get_feature_names_out(),
                index=X_test.index
            )
            groups = outer_groups[train_idx] if outer_groups is not None else None

            if "EF Lasso" in models:
                # __Lasso__
                print("Fitting of EF Lasso")
                model = clone(lasso)
                if task_type == "binary":
                    predictions = model.fit(X_train_std, y_train, groups=groups).predict_proba(X_test_std)[:, 1]
                else:
                    predictions = model.fit(X_train_std, y_train, groups=groups).predict(X_test_std)
                tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                fold_selected_features["EF Lasso"] = tmp_sel_features
                print(
                    f"EF Lasso finished on {omic_name} ({X_train_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict["EF Lasso"].loc[test_idx, f"Fold n°{k}"] = predictions

            if "EF ALasso" in models:
                # __ALasso__
                print("Fitting of EF ALasso")
                model = clone(alasso)
                if task_type == "binary":
                    predictions = model.fit(X_train_std, y_train, groups=groups).predict_proba(X_test_std)[:, 1]
                else:
                    predictions = model.fit(X_train_std, y_train, groups=groups).predict(X_test_std)
                tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                fold_selected_features["EF ALasso"] = tmp_sel_features
                print(
                    f"EF ALasso finished on {omic_name} ({X_train_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict["EF ALasso"].loc[test_idx, f"Fold n°{k}"] = predictions

            if "EF ElasticNet" in models:
                # __EN__
                print("Fitting of EF ElasticNet")
                model = clone(en)
                if task_type == "binary":
                    predictions = model.fit(X_train_std, y_train, groups=groups).predict_proba(X_test_std)[:, 1]
                else:
                    predictions = model.fit(X_train_std, y_train, groups=groups).predict(X_test_std)
                tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                fold_selected_features["EF ElasticNet"] = tmp_sel_features
                print(
                    f"EF ElasticNet finished on {omic_name} ({X_train_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict["EF ElasticNet"].loc[test_idx, f"Fold n°{k}"] = predictions

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

    jaccard_matrix_dict = dict()
    formatted_features_dict = dict()

    for model in models:

        #jaccard_matrix_dict[model] = jaccard_matrix(selected_features_dict[model])

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

    p_values = compute_pvalues_table(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        selected_features_dict=formatted_features_dict
    )

    table_of_scores.to_csv(Path(summary_res_path, "Scores training CV.csv"))
    table_of_scores.to_csv(Path(cv_res_path, "Scores training CV.csv"))

    p_values_path = Path(cv_res_path, "p-values")
    os.makedirs(p_values_path, exist_ok=True)
    for m, p in p_values.items():
        p.to_csv(Path(p_values_path, f"{m}.csv"))

    save_plots(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        save_path=cv_res_path
    )

    return predictions_dict


def multi_omic_stabl(
        data_dict,
        y,
        estimators,
        task_type,
        save_path,
        models,
        model_chosen,
        stabl_params=None,
        groups=None,
        early_fusion=False,
        X_test=None,
        y_test=None,
        n_iter_lf=10000
):
    """
    Performs a cross validation on the data_dict using the models and saves the results in save_path.

    Parameters
    ----------
    data_dict: dict
        Dictionary containing the input omic-files.

    y: pd.Series
        pandas Series containing the outcomes for the use case. Note that y should contains the union of outcomes for
        the data_dict.

    estimators : dict[str, sklearn estimator]
        Dict of feature selectors to benchmark with their name as key. They must implement `fit`, `get_support(indices=True)`. It should contain :
        - "lasso" : Lasso in GridSearchCV
        - "alasso" : ALasso in GridSearchCV
        - "en" : ElasticNet in GridSearchCV
        - "sgl" : SGL in GridSearchCV
        - "stabl_lasso" : Stabl with Lasso as base estimator
        - "stabl_alasso" : Stabl with ALasso as base estimator
        - "stabl_en" : Stabl with ElasticNet as base estimator
        - "stabl_sgl" : Stabl with SGL as base estimator

    task_type: str
        Can either be "binary" for binary classification or "regression" for regression tasks.

    save_path: Path or str
        Where to save the results

    models: list of str
        List of models to use. Can contain :
        - "Lasso" : Lasso
        - "STABL Lasso" : Stabl with Lasso as base estimator
        - "ALasso" : ALasso
        - "STABL ALasso" : Stabl with ALasso as base estimator
        - "ElasticNet" : ElasticNet
        - "STABL ElasticNet" : Stabl with ElasticNet as base estimator
        - "SGL-90" : SGL with 0.90 as correlation threshold
        - "STABL SGL-90" : Stabl with SGL with 0.90 as correlation threshold as base estimator
        - "SGL-95" : SGL with 0.95 as correlation threshold
        - "STABL SGL-95" : Stabl with SGL with 0.95 as correlation threshold as base estimator

    stabl_params: dict, default=None
        Dictionary containing the parameters to use for STABL. It overrides default settings. It is used to change Stabl parameters between omics. 
        It is of the form :
        {
            "STABL model" : {
                "omic_name" : dict of lambda_grid
            }
        }

    groups: pd.Series, default=None
        If used, should be the same size as y and should indicate the groups of the samples.

    early_fusion: bool, default=False
        If True, it will perform early fusion for each estimator.

    X_test: dict or None, default=None
        Dictionary containing the test omic-files.

    y_test: pd.Series or None, default=None
        pandas Series containing the outcomes for the use case. Note that y should contains the union of outcomes for X_test.

    n_iter_lf: int, default=10000
        Number of iterations for the late fusion.

    Returns
    -------
    predictions_dict: dict
        Dictionary containing the predictions of each model for each sample of X_test.
    """

    if stabl_params is None:
        stabl_params = {}

    if early_fusion:
        models += ["EF " + model for model in models if "STABL" not in model]

    lasso = estimators["lasso"]
    alasso = estimators["alasso"]
    en = estimators["en"]

    stabl = estimators["stabl_lasso"]
    stabl_alasso = estimators["stabl_alasso"]
    stabl_en = estimators["stabl_en"]

    os.makedirs(Path(save_path, "Training-Validation"), exist_ok=True)
    os.makedirs(Path(save_path, "Summary"), exist_ok=True)

    # Initializing the df containing the data of all omics
    X_tot = pd.concat(data_dict.values(), axis="columns")
    if X_test is not None:
        X_test_tot = pd.concat(X_test.values(), axis="columns")
        X_test_tot = X_test_tot.loc[y_test.index]

    predictions_dict = dict()
    selected_features_dict = dict()

    for model in models:
        selected_features_dict[model] = []

    predictions_dict_train_late_fusion = dict()
    for model in models:
        if "STABL" not in model and "EF" not in model:
            predictions_dict_train_late_fusion[model] = pd.DataFrame(data=None, columns=data_dict.keys(), index=y.index)
    predictions_dict_test_late_fusion = dict()
    if X_test is not None and y_test is not None:
        for model in models:
            if "STABL" not in model and "EF" not in model:
                predictions_dict_test_late_fusion[model] = pd.DataFrame(data=None, columns=X_test.keys(), index=y_test.index)

    for omic_name, X_omic in data_dict.items():
        all_columns = X_omic.columns
        if X_test is not None:
            all_columns = all_columns.intersection(X_test[omic_name].columns)

        # if prepro is not None:
        #     preprocessing = prepro

        X_omic_std = pd.DataFrame(
            data=preprocessing.fit_transform(X_omic[all_columns]),
            index=X_omic.index,
            columns=preprocessing.get_feature_names_out()
        )
        y_omic = y.loc[X_omic_std.index]
        if X_test is not None:
            X_test_omic_std = pd.DataFrame(
                data=preprocessing.transform(X_test[omic_name][all_columns]),
                index=X_test[omic_name].index,
                columns=preprocessing.get_feature_names_out()
            )

        # __STABL__
        if "STABL Lasso" in models:
            # fit STABL Lasso
            print(f"Fitting of STABL Lasso on {omic_name}")
            if "STABL Lasso" in stabl_params and omic_name in stabl_params["STABL Lasso"]:
                stabl.set_params(lambda_grid=stabl_params["STABL Lasso"][omic_name])
            stabl.fit(X_omic_std, y_omic, groups=groups)
            tmp_sel_features = list(stabl.get_feature_names_out())
            selected_features_dict["STABL Lasso"].extend(tmp_sel_features)
            print(
                f"STABL Lasso finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            save_stabl_results(
                stabl=stabl,
                path=Path(save_path, "Training-Validation", f"STABL Lasso results on {omic_name}"),
                df_X=X_omic,
                y=y_omic,
                task_type=task_type
            )

        if "STABL ALasso" in models:
            # fit STABL ALasso
            print(f"Fitting of STABL ALasso on {omic_name}")
            if "STABL ALasso" in stabl_params and omic_name in stabl_params["STABL ALasso"]:
                stabl_alasso.set_params(lambda_grid=stabl_params["STABL ALasso"][omic_name])
            stabl_alasso.fit(X_omic_std, y_omic, groups=groups)
            tmp_sel_features = list(stabl_alasso.get_feature_names_out())
            selected_features_dict["STABL ALasso"].extend(tmp_sel_features)
            print(
                f"STABL ALasso finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            save_stabl_results(
                stabl=stabl_alasso,
                path=Path(save_path, "Training-Validation", f"STABL ALasso results on {omic_name}"),
                df_X=X_omic,
                y=y_omic,
                task_type=task_type
            )

        if "STABL ElasticNet" in models:
            # fit STABL ElasticNet
            print(f"Fitting of STABL ElasticNet on {omic_name}")
            if "STABL ElasticNet" in stabl_params and omic_name in stabl_params["STABL ElasticNet"]:
                stabl_en.set_params(lambda_grid=stabl_params["STABL ElasticNet"][omic_name])
            stabl_en.fit(X_omic_std, y_omic, groups=groups)
            tmp_sel_features = list(stabl_en.get_feature_names_out())
            selected_features_dict["STABL ElasticNet"].extend(tmp_sel_features)
            print(
                f"STABL ElasticNet finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            save_stabl_results(
                stabl=stabl_en,
                path=Path(save_path, "Training-Validation", f"STABL ElasticNet results on {omic_name}"),
                df_X=X_omic,
                y=y_omic,
                task_type=task_type
            )

        if "Lasso" in models:
            # __Lasso__
            print(f"Fitting of Lasso on {omic_name}")
            model = clone(lasso)
            try:
                model = model.fit(X_omic_std, y_omic, groups=groups)
            except:
                model = model.fit(X_omic_std, y_omic)

            model = model.best_estimator_

            if task_type == "binary":
                predictions = model.predict_proba(X_omic_std)[:, 1]
            else:
                predictions = model.predict(X_omic_std)

            tmp_sel_features = list(X_omic_std.columns[np.where(model.coef_.flatten())])
            selected_features_dict["Lasso"].extend(tmp_sel_features)
            print(
                f"Lasso finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )

            base_linear_model_coef = pd.DataFrame(
                {"Feature": tmp_sel_features,
                 "Associated weight": model.coef_.flatten()[np.where(model.coef_.flatten())]
                 }
            ).set_index("Feature")
            base_linear_model_coef.to_csv(Path(save_path, "Training-Validation", f"Lasso coefficients {omic_name}.csv"))

            predictions_dict_train_late_fusion["Lasso"].loc[X_omic.index, omic_name] = predictions
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_omic_std)[:, 1]
                else:
                    predictions = model.predict(X_test_omic_std)
                predictions_dict_test_late_fusion["Lasso"].loc[X_test_omic_std.index, omic_name] = predictions
                predictions_dict_test_late_fusion["Lasso"].fillna(np.median(predictions_dict_train_late_fusion["Lasso"]), inplace=True)

        if "ALasso" in models:
            # __ALasso__
            print(f"Fitting of ALasso on {omic_name}")
            model = clone(alasso)
            try:
                model = model.fit(X_omic_std, y_omic, groups=groups)
            except:
                model = model.fit(X_omic_std, y_omic)

            model = model.best_estimator_

            if task_type == "binary":
                predictions = model.predict_proba(X_omic_std)[:, 1]
            else:
                predictions = model.predict(X_omic_std)

            tmp_sel_features = list(X_omic_std.columns[np.where(model.coef_.flatten())])
            selected_features_dict["ALasso"].extend(tmp_sel_features)
            print(
                f"ALasso finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )

            base_linear_model_coef = pd.DataFrame(
                {"Feature": tmp_sel_features,
                 "Associated weight": model.coef_.flatten()[np.where(model.coef_.flatten())]
                 }
            ).set_index("Feature")
            base_linear_model_coef.to_csv(Path(save_path, "Training-Validation", f"ALasso coefficients {omic_name}.csv"))

            predictions_dict_train_late_fusion["ALasso"].loc[X_omic.index, omic_name] = predictions
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_omic_std)[:, 1]
                else:
                    predictions = model.predict(X_test_omic_std)
                predictions_dict_test_late_fusion["ALasso"].loc[X_test_omic_std.index, omic_name] = predictions
                predictions_dict_test_late_fusion["ALasso"].fillna(np.median(predictions_dict_train_late_fusion["ALasso"]), inplace=True)

        if "ElasticNet" in models:
            # __EN__
            print(f"Fitting of ElasticNet on {omic_name}")
            model = clone(en)
            try:
                model = model.fit(X_omic_std, y_omic, groups=groups)
            except:
                model = model.fit(X_omic_std, y_omic)

            model = model.best_estimator_

            if task_type == "binary":
                predictions = model.predict_proba(X_omic_std)[:, 1]
            else:
                predictions = model.predict(X_omic_std)

            tmp_sel_features = list(X_omic_std.columns[np.where(model.coef_.flatten())])
            selected_features_dict["ElasticNet"].extend(tmp_sel_features)
            print(
                f"ElasticNet finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )

            base_linear_model_coef = pd.DataFrame(
                {"Feature": tmp_sel_features,
                 "Associated weight": model.coef_.flatten()[np.where(model.coef_.flatten())]
                 }
            ).set_index("Feature")
            base_linear_model_coef.to_csv(
                Path(save_path, "Training-Validation", f"ElasticNet coefficients {omic_name}.csv")
            )

            predictions_dict_train_late_fusion["ElasticNet"].loc[X_omic.index, omic_name] = predictions
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_omic_std)[:, 1]
                else:
                    predictions = model.predict(X_test_omic_std)
                predictions_dict_test_late_fusion["ElasticNet"].loc[X_test_omic_std.index, omic_name] = predictions
                predictions_dict_test_late_fusion["ElasticNet"].fillna(
                    np.median(predictions_dict_train_late_fusion["ElasticNet"]), inplace=True
                )

    final_prepro = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("std", StandardScaler())]
    )

    for model in filter(lambda x: "STABL" in x, models):
        if len(selected_features_dict[model]) > 0:
            X_train = X_tot[selected_features_dict[model]]
            X_train_std = pd.DataFrame(
                data=preprocessing.fit_transform(X_train),
                index=X_tot.index,
                columns=preprocessing.get_feature_names_out()
            )

            if task_type == "binary":
                base_linear_model = logit

            else:
                if model_chosen=='linear_reg':
                    base_linear_model = linreg
                elif chosen_model=='random_forest':
                    base_linear_model=randomforest
                elif model_chosen=='xgboost':
                        predictions = clone(xgboost).fit(X_train, y_train).predict(X_test)

            base_linear_model.fit(X_train_std, y)
            if hasattr(base_linear_model, 'coef_'):
                # LinearRegression or binary classification
                importance = base_linear_model.coef_.flatten()
                metric_name = "Associated weight"
            elif hasattr(base_linear_model, 'feature_importances_'):
                # RandomForestRegressor
                importance = base_linear_model.feature_importances_
                metric_name = "Feature importance"
            else:
                raise ValueError("Model does not expose interpretable feature importances")
            
            base_model_coef = pd.DataFrame(
                {"Feature": selected_features_dict[model],
                 metric_name: importance}).set_index("Feature")

            base_model_coef.to_csv(Path(save_path, "Training-Validation", f"{model} coefficients.csv"))

            if X_test is not None:
                X_test_std = pd.DataFrame(
                    data=preprocessing.transform(X_test_tot[selected_features_dict[model]]),
                    index=X_test_tot.index,
                    columns=preprocessing.get_feature_names_out()
                )
                if task_type == "binary":
                    model_preds = base_linear_model.predict_proba(X_test_std)[:, 1]
                else:
                    model_preds = base_linear_model.predict(X_test_std)
        else:
            if X_test is not None:
                model_preds = np.zeros(X_test_tot.shape[0])
                if task_type == "binary":
                    model_preds[:] = 0.5
                elif task_type == "regression":
                    model_preds[:] = np.mean(y)
        if X_test is not None:
            predictions_dict[model] = pd.Series(
                model_preds,
                index=y_test.index,
                name=f"{model} predictions"
            )

    # __late fusion__
    preds_lf = late_fusion_validation(
        predictions_dict_train_late_fusion, predictions_dict_test_late_fusion, y,
        task_type, Path(save_path, "Training-Validation"), n_iter=n_iter_lf
    )
    if preds_lf is not None:
        for model in preds_lf:
            predictions_dict[model] = preds_lf[model]

    # __EF Lasso__
    if early_fusion:
        X_train_std = pd.DataFrame(
            data=preprocessing.fit_transform(X_tot),
            index=X_tot.index,
            columns=preprocessing.get_feature_names_out()
        )

        if X_test is not None:
            X_test_std = pd.DataFrame(
                data=preprocessing.transform(X_test_tot),
                index=X_test_tot.index,
                columns=preprocessing.get_feature_names_out()
            )

        if "EF Lasso" in models:
            # __Lasso__
            print("Fitting of EF Lasso")
            model = clone(lasso)
            try:
                model.fit(X_train_std, y, groups=groups)
            except:
                model.fit(X_train_std, y)
            tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
            selected_features_dict["EF Lasso"] = tmp_sel_features

            model_coef = pd.DataFrame(
                {"Feature": selected_features_dict["EF Lasso"],
                 "Associated weight": model.best_estimator_.coef_.flatten()[np.where(model.best_estimator_.coef_.flatten())]
                 }
            ).set_index("Feature")
            model_coef.to_csv(Path(save_path, "Training-Validation", "EF Lasso coefficients.csv"))

            print(
                f"EF Lasso finished on {omic_name} ({X_train_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_std)[:, 1]
                else:
                    predictions = model.predict(X_test_std)
                predictions_dict["EF Lasso"] = pd.Series(
                    predictions,
                    index=y_test.index,
                    name=f"EF Lasso predictions"
                )

        if "EF ALasso" in models:
            # __ALasso__
            print("Fitting of EF ALasso")
            model = clone(alasso)
            try:
                model.fit(X_train_std, y, groups=groups)
            except:
                model.fit(X_train_std, y)
            tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
            selected_features_dict["EF ALasso"] = tmp_sel_features

            model_coef = pd.DataFrame(
                {"Feature": selected_features_dict["EF ALasso"],
                 "Associated weight": model.best_estimator_.coef_.flatten()[np.where(model.best_estimator_.coef_.flatten())]
                 }
            ).set_index("Feature")
            model_coef.to_csv(Path(save_path, "Training-Validation", "EF ALasso coefficients.csv"))

            print(
                f"EF ALasso finished on {omic_name} ({X_train_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_std)[:, 1]
                else:
                    predictions = model.predict(X_test_std)
                predictions_dict["EF ALasso"] = pd.Series(
                    predictions,
                    index=y_test.index,
                    name=f"EF ALasso predictions"
                )

        if "EF ElasticNet" in models:
            # __EN__
            print("Fitting of EF ElasticNet")
            model = clone(en)
            try:
                model.fit(X_train_std, y, groups=groups)
            except:
                model.fit(X_train_std, y)
            tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
            selected_features_dict["EF ElasticNet"] = tmp_sel_features

            model_coef = pd.DataFrame(
                {"Feature": selected_features_dict["EF ElasticNet"],
                 "Associated weight": model.best_estimator_.coef_.flatten()[np.where(model.best_estimator_.coef_.flatten())]
                 }
            ).set_index("Feature")
            model_coef.to_csv(Path(save_path, "Training-Validation", "EF ElasticNet coefficients.csv"))

            print(
                f"EF ElasticNet finished on {omic_name} ({X_train_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_std)[:, 1]
                else:
                    predictions = model.predict(X_test_std)
                predictions_dict["EF ElasticNet"] = pd.Series(
                    predictions,
                    index=y_test.index,
                    name=f"EF ElasticNet predictions"
                )

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for model in models:
        print(f"This fold: {len(selected_features_dict[model])} features selected for {model}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    if X_test is not None:
        table_of_scores = compute_scores_table(
            predictions_dict=predictions_dict,
            y=y_test,
            task_type=task_type
        )
        p_values = compute_pvalues_table(
            predictions_dict=predictions_dict,
            y=y_test,
            task_type=task_type
        )

        table_of_scores.to_csv(Path(save_path, "Training-Validation", "Scores on Validation.csv"))
        table_of_scores.to_csv(Path(save_path, "Summary", "Scores on Validation.csv"))

        p_values_path = Path(Path(save_path, "Training-Validation"), "p-values")
        os.makedirs(p_values_path, exist_ok=True)
        for m, p in p_values.items():
            p.to_csv(Path(p_values_path, f"{m}.csv"))

        save_plots(
            predictions_dict=predictions_dict,
            y=y_test,
            task_type=task_type,
            save_path=Path(save_path, "Training-Validation"),
        )

    return predictions_dict


def late_fusion_cv(
    predictions_lf_dict,
    y,
    task_type,
    save_path,
    n_iter=10000
):
    """
    Perform a late fusion of omics using the prediction of each model on the train set.
    It uses stacked_multi_omic to perform the late fusion.

    Parameters
    ----------
    predictions_lf_dict: dict[str, pd.DataFrame]
        Dictionary containing the prediction on each omic.

    y: pd.Series
        pandas Series containing the outcomes for the use case. Note that y should contains the union of outcomes for
        the data_dict.

    task_type: str
        Can either be "binary" for binary classification or "regression" for regression tasks.

    save_path: Path or str
        Where to save the results

    n_iter: int
        Number of iterations for the late fusion.
    Returns
    -------
    final_predictions_dict: dict of pd.DataFrame
        Dictionary containing the final predictions of each model after late fusion.
    """
    final_predictions_dict = {}
    os.makedirs(save_path, exist_ok=True)
    for model_name, predictions in (tmodel := tqdm(predictions_lf_dict.items(), total=len(predictions_lf_dict))):
        tmodel.set_description(f"Late Fusion {model_name}")

        model_lf_path = Path(save_path, model_name)
        os.makedirs(model_lf_path, exist_ok=True)

        preds_omics = pd.DataFrame(data=None, columns=predictions.keys())
        for omic_name, preds in predictions.items():
            preds_omics[omic_name] = preds.median(axis=1)
        stacked_df, weights = stacked_multi_omic(preds_omics, y, task_type, n_iter=n_iter)
        weights.to_csv(Path(model_lf_path, f"Associated weights {model_name}.csv"))
        stacked_df.to_csv(
            Path(model_lf_path, f"Stacked Generalization predictions {model_name}.csv"))
        final_predictions_dict[model_name] = stacked_df["Stacked Gen. Predictions"]
    return final_predictions_dict


def late_fusion_validation(
    predictions_train_dict,
    predictions_valid_dict,
    y,
    task_type,
    save_path,
    n_iter
):
    """Perform a late fusion of omics using the prediction of each model on the train set.
    It uses stacked_multi_omic to perform the late fusion.

    Parameters
    ----------
    predictions_train_dict : dict of pd.DataFrame
        Predictions of each model on each omic of the train set.
    predictions_valid_dict : dict of pd.DataFrame or None
        Predictions of each model on each omic of the test set.
    y : pd.Series
        Outcome of the train set.
    task_type : str
        Task type of the use case. Can be "binary" or "regression".
    save_path : str or Path
        Path where to save the results.
    n_iter : int
        Number of iterations for the late fusion.

    Returns
    -------
    train_predictions_dict: dict of pd.DataFrame
        Predictions of each model on the train set after late fusion.

    final_predictions_dict: dict of pd.DataFrame or None
        Predictions of each model on the test set after late fusion.
        None if predictions_valid_dict is None.
    """
    final_predictions_dict = {}
    os.makedirs(save_path, exist_ok=True)
    for model_name, predictions in (tmodel := tqdm(predictions_train_dict.items(), total=len(predictions_train_dict))):
        tmodel.set_description(f"Late Fusion {model_name}")

        model_lf_path = Path(save_path, model_name)
        os.makedirs(model_lf_path, exist_ok=True)
        stacked_df, weights = stacked_multi_omic(predictions, y, task_type, n_iter=n_iter)
        weights.to_csv(Path(model_lf_path, f"Associated weights {model_name}.csv"))
        stacked_df.to_csv(
            Path(model_lf_path, f"Stacked Generalization predictions train {model_name}.csv"))
        if predictions_valid_dict:
            valid_preds = predictions_valid_dict[model_name]
            final_predictions_dict[model_name] = ((valid_preds * weights["Associated weight"]).sum(axis=1) /
                                                  weights["Associated weight"].sum())
    if predictions_valid_dict:
        return final_predictions_dict
    else:
        return None


def multi_omic_stabl_noe(
        data_dict,
        y,
        estimators,
        task_type,
        save_path,
        models,
        model_chosen,                      # <-- ajouté: choix du modèle final ("xgboost" ou "logit" en binaire; "lin_reg"/"rf"/"xgboost" en régression)
        stabl_params=None,
        groups=None,
        early_fusion=False,
        X_test=None,
        y_test=None,
        n_iter_lf=10000,
        importance_method: str = "weight", # <-- ajouté: pour STABL arbres/boosting
):
    """
    One-shot (pas de CV): sélection de variables par omique (STABL + baselines),
    puis fit final sur l'ensemble du train en concaténant les features sélectionnées,
    avec late-fusion optionnelle et early-fusion optionnelle.

    - Supporte Lasso/ALasso/ElasticNet et leurs versions STABL
    - Ajoute RandomForest / CatBoost / LightGBM / XGBoost et leurs versions STABL
    - `importance_method` passé aux STABL arbres/boosting
    - `model_chosen` pour le modèle final:
        * classification: "xgboost" => XGBClassifier sinon LogisticRegression
        * régression    : "lin_reg" | "rf" | "xgboost" (si fournis dans `estimators`)
    """

    # Imports locaux pour éviter les NameError si le module n'avait pas ces imports
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.base import clone
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import os

    try:
        from xgboost import XGBClassifier
    except Exception:
        XGBClassifier = None  # on gérera plus bas

    # --- config / garde-fous ---
    if stabl_params is None:
        stabl_params = {}

    if early_fusion:
        models = list(models) + ["EF " + m for m in models if "STABL" not in m]

    # --- récup des estimateurs (optionnels pour arbres/boosting) ---
    lasso  = estimators.get("lasso")
    alasso = estimators.get("alasso")
    en     = estimators.get("en")

    stabl_lasso  = estimators.get("stabl_lasso")
    stabl_alasso = estimators.get("stabl_alasso")
    stabl_en     = estimators.get("stabl_en")

    rf  = estimators.get("rf")   # GridSearchCV ou estimator pour RF
    cb  = estimators.get("cb")   # CatBoost (GridSearchCV), si dispo
    lgb = estimators.get("lgb")  # LightGBM (GridSearchCV), si dispo
    xgb = estimators.get("xgb")  # XGBoost (GridSearchCV / estimator), si dispo

    stabl_rf  = estimators.get("stabl_rf")
    stabl_cb  = estimators.get("stabl_cb")
    stabl_lgb = estimators.get("stabl_lgb")
    stabl_xgb = estimators.get("stabl_xgb")

    # Injection du paramètre d'importance dans les STABL arbres/boosting (si supporté)
    for _name in ["stabl_rf", "stabl_cb", "stabl_lgb", "stabl_xgb"]:
        if estimators.get(_name, None) is not None:
            try:
                estimators[_name].set_params(importance_method=importance_method)
            except ValueError:
                pass  # versions STABL sans ce paramètre

    # --- chemins ---
    os.makedirs(Path(save_path, "Training-Validation"), exist_ok=True)
    os.makedirs(Path(save_path, "Summary"), exist_ok=True)

    # --- concat multi-omic ---
    X_tot = pd.concat(data_dict.values(), axis="columns")
    if X_test is not None:
        X_test_tot = pd.concat(X_test.values(), axis="columns")
        X_test_tot = X_test_tot.loc[y_test.index]

    # --- conteneurs ---
    predictions_dict = dict()       # prédictions sur le set de validation si fourni (X_test/y_test)
    selected_features_dict = {m: [] for m in models}  # liste de features sélectionnées (toutes omiques concat.)

    # Pour la late-fusion: on stocke les prédictions par omique (train et test)
    predictions_dict_train_lf = {
        m: pd.DataFrame(data=None, columns=data_dict.keys(), index=y.index)
        for m in models if "STABL" not in m and "EF" not in m
    }
    predictions_dict_test_lf = {}
    if X_test is not None and y_test is not None:
        predictions_dict_test_lf = {
            m: pd.DataFrame(data=None, columns=X_test.keys(), index=y_test.index)
            for m in models if "STABL" not in m and "EF" not in m
        }

    # --- Boucle par omique: sélection (STABL + baselines) + prédictions per-omic pour late-fusion ---
    for omic_name, X_omic in data_dict.items():
        all_columns = X_omic.columns
        if X_test is not None and omic_name in X_test:
            all_columns = all_columns.intersection(X_test[omic_name].columns)

        # Prétraitement (pipeline global `preprocessing` supposé défini dans le module)
        X_omic_std = pd.DataFrame(
            data=preprocessing.fit_transform(X_omic[all_columns]),
            index=X_omic.index,
            columns=preprocessing.get_feature_names_out()
        )
        y_omic = y.loc[X_omic_std.index]

        if X_test is not None and y_test is not None:
            X_test_omic_std = pd.DataFrame(
                data=preprocessing.transform(X_test[omic_name][all_columns]),
                index=X_test[omic_name].index,
                columns=preprocessing.get_feature_names_out()
            )

        # ---------------- STABL (lasso / alasso / en) ----------------
        if "STABL Lasso" in models and stabl_lasso is not None:
            if "STABL Lasso" in stabl_params and omic_name in stabl_params["STABL Lasso"]:
                stabl_lasso.set_params(lambda_grid=stabl_params["STABL Lasso"][omic_name])
            stabl_lasso.fit(X_omic_std, y_omic, groups=groups)
            feats = list(stabl_lasso.get_feature_names_out())
            selected_features_dict["STABL Lasso"].extend(feats)
            save_stabl_results(
                stabl=stabl_lasso,
                path=Path(save_path, "Training-Validation", f"STABL Lasso results on {omic_name}"),
                df_X=X_omic, y=y_omic, task_type=task_type
            )

        if "STABL ALasso" in models and stabl_alasso is not None:
            if "STABL ALasso" in stabl_params and omic_name in stabl_params["STABL ALasso"]:
                stabl_alasso.set_params(lambda_grid=stabl_params["STABL ALasso"][omic_name])
            stabl_alasso.fit(X_omic_std, y_omic, groups=groups)
            feats = list(stabl_alasso.get_feature_names_out())
            selected_features_dict["STABL ALasso"].extend(feats)
            save_stabl_results(
                stabl=stabl_alasso,
                path=Path(save_path, "Training-Validation", f"STABL ALasso results on {omic_name}"),
                df_X=X_omic, y=y_omic, task_type=task_type
            )

        if "STABL ElasticNet" in models and stabl_en is not None:
            if "STABL ElasticNet" in stabl_params and omic_name in stabl_params["STABL ElasticNet"]:
                stabl_en.set_params(lambda_grid=stabl_params["STABL ElasticNet"][omic_name])
            stabl_en.fit(X_omic_std, y_omic, groups=groups)
            feats = list(stabl_en.get_feature_names_out())
            selected_features_dict["STABL ElasticNet"].extend(feats)
            save_stabl_results(
                stabl=stabl_en,
                path=Path(save_path, "Training-Validation", f"STABL ElasticNet results on {omic_name}"),
                df_X=X_omic, y=y_omic, task_type=task_type
            )

        # ---------------- STABL (arbres/boosting) ----------------
        if "STABL RandomForest" in models and stabl_rf is not None:
            stabl_rf.fit(X_omic_std, y_omic, groups=groups)
            selected_features_dict["STABL RandomForest"].extend(list(stabl_rf.get_feature_names_out()))
            save_stabl_results(
                stabl=stabl_rf,
                path=Path(save_path, "Training-Validation", f"STABL RandomForest results on {omic_name}"),
                df_X=X_omic, y=y_omic, task_type=task_type
            )

        if "STABL CatBoost" in models and stabl_cb is not None:
            stabl_cb.fit(X_omic_std, y_omic, groups=groups)
            selected_features_dict["STABL CatBoost"].extend(list(stabl_cb.get_feature_names_out()))
            save_stabl_results(
                stabl=stabl_cb,
                path=Path(save_path, "Training-Validation", f"STABL CatBoost results on {omic_name}"),
                df_X=X_omic, y=y_omic, task_type=task_type
            )

        if "STABL XGBoost" in models and stabl_xgb is not None:
            stabl_xgb.fit(X_omic_std, y_omic, groups=groups)
            selected_features_dict["STABL XGBoost"].extend(list(stabl_xgb.get_feature_names_out()))
            save_stabl_results(
                stabl=stabl_xgb,
                path=Path(save_path, "Training-Validation", f"STABL XGBoost results on {omic_name}"),
                df_X=X_omic, y=y_omic, task_type=task_type
            )

        if "STABL LightGBM" in models and stabl_lgb is not None:
            stabl_lgb.fit(X_omic_std, y_omic, groups=groups)
            selected_features_dict["STABL LightGBM"].extend(list(stabl_lgb.get_feature_names_out()))
            save_stabl_results(
                stabl=stabl_lgb,
                path=Path(save_path, "Training-Validation", f"STABL LightGBM results on {omic_name}"),
                df_X=X_omic, y=y_omic, task_type=task_type
            )

        # ---------------- Baselines non-STABL (per-omic, pour late-fusion) ----------------
        def _save_linear_model(model_obj, name: str):
            # enregistre coef/poids pour Lasso/ALasso/EN
            if hasattr(model_obj, "coef_"):
                nz = np.where(model_obj.coef_.flatten())
                tmp_sel = list(X_omic_std.columns[nz])
                selected_features_dict[name].extend(tmp_sel)
                coefs = pd.DataFrame(
                    {"Feature": tmp_sel, "Associated weight": model_obj.coef_.flatten()[nz]}
                ).set_index("Feature")
                coefs.to_csv(Path(save_path, "Training-Validation", f"{name} coefficients {omic_name}.csv"))

        if "Lasso" in models and lasso is not None:
            m = clone(lasso)
            try:
                m = m.fit(X_omic_std, y_omic, groups=groups).best_estimator_
            except Exception:
                m = m.fit(X_omic_std, y_omic).best_estimator_
            preds_tr = (m.predict_proba(X_omic_std)[:, 1] if task_type == "binary" else m.predict(X_omic_std))
            predictions_dict_train_lf["Lasso"].loc[X_omic_std.index, omic_name] = preds_tr
            _save_linear_model(m, "Lasso")
            if X_test is not None:
                preds_te = (m.predict_proba(X_test_omic_std)[:, 1] if task_type == "binary" else m.predict(X_test_omic_std))
                predictions_dict_test_lf["Lasso"].loc[X_test_omic_std.index, omic_name] = preds_te
                predictions_dict_test_lf["Lasso"].fillna(np.median(predictions_dict_train_lf["Lasso"]), inplace=True)

        if "ALasso" in models and alasso is not None:
            m = clone(alasso)
            try:
                m = m.fit(X_omic_std, y_omic, groups=groups).best_estimator_
            except Exception:
                m = m.fit(X_omic_std, y_omic).best_estimator_
            preds_tr = (m.predict_proba(X_omic_std)[:, 1] if task_type == "binary" else m.predict(X_omic_std))
            predictions_dict_train_lf["ALasso"].loc[X_omic_std.index, omic_name] = preds_tr
            _save_linear_model(m, "ALasso")
            if X_test is not None:
                preds_te = (m.predict_proba(X_test_omic_std)[:, 1] if task_type == "binary" else m.predict(X_test_omic_std))
                predictions_dict_test_lf["ALasso"].loc[X_test_omic_std.index, omic_name] = preds_te
                predictions_dict_test_lf["ALasso"].fillna(np.median(predictions_dict_train_lf["ALasso"]), inplace=True)

        if "ElasticNet" in models and en is not None:
            m = clone(en)
            try:
                m = m.fit(X_omic_std, y_omic, groups=groups).best_estimator_
            except Exception:
                m = m.fit(X_omic_std, y_omic).best_estimator_
            preds_tr = (m.predict_proba(X_omic_std)[:, 1] if task_type == "binary" else m.predict(X_omic_std))
            predictions_dict_train_lf["ElasticNet"].loc[X_omic_std.index, omic_name] = preds_tr
            _save_linear_model(m, "ElasticNet")
            if X_test is not None:
                preds_te = (m.predict_proba(X_test_omic_std)[:, 1] if task_type == "binary" else m.predict(X_test_omic_std))
                predictions_dict_test_lf["ElasticNet"].loc[X_test_omic_std.index, omic_name] = preds_te
                predictions_dict_test_lf["ElasticNet"].fillna(np.median(predictions_dict_train_lf["ElasticNet"]), inplace=True)

        # Arbres/boosting per-omic
        def _fit_predict_tree(grid_or_est, Xtr, ytr, Xte=None):
            mloc = clone(grid_or_est)
            try:
                mloc = mloc.fit(Xtr, ytr, groups=groups)
            except Exception:
                mloc = mloc.fit(Xtr, ytr)
            best = getattr(mloc, "best_estimator_", mloc)
            if task_type == "binary" and hasattr(best, "predict_proba"):
                ptr = best.predict_proba(Xtr)[:, 1]
                pte = (best.predict_proba(Xte)[:, 1] if Xte is not None else None)
            else:
                ptr = best.predict(Xtr)
                pte = (best.predict(Xte) if Xte is not None else None)
            return best, ptr, pte

        def _save_tree_importances(best, name: str):
            # importance > 0
            imps = None
            if hasattr(best, "feature_importances_"):
                imps = np.asarray(best.feature_importances_)
            elif hasattr(best, "get_feature_importance"):
                try:
                    imps = np.asarray(best.get_feature_importance())
                except Exception:
                    imps = None
            if imps is not None:
                sel_idx = np.where(imps > 0)
                feats = list(X_omic_std.columns[sel_idx])
                selected_features_dict[name].extend(feats)
                df_imp = pd.DataFrame({"Feature": feats, "Feature importance": imps[sel_idx]}).set_index("Feature")
                df_imp.to_csv(Path(save_path, "Training-Validation", f"{name} importances {omic_name}.csv"))

        if "RandomForest" in models and rf is not None:
            best, ptr, pte = _fit_predict_tree(rf, X_omic_std, y_omic, X_test_omic_std if X_test is not None else None)
            predictions_dict_train_lf["RandomForest"].loc[X_omic_std.index, omic_name] = ptr
            if X_test is not None:
                predictions_dict_test_lf["RandomForest"].loc[X_test_omic_std.index, omic_name] = pte
                predictions_dict_test_lf["RandomForest"].fillna(np.median(predictions_dict_train_lf["RandomForest"]), inplace=True)
            _save_tree_importances(best, "RandomForest")

        if "XGBoost" in models and xgb is not None:
            best, ptr, pte = _fit_predict_tree(xgb, X_omic_std, y_omic, X_test_omic_std if X_test is not None else None)
            predictions_dict_train_lf["XGBoost"].loc[X_omic_std.index, omic_name] = ptr
            if X_test is not None:
                predictions_dict_test_lf["XGBoost"].loc[X_test_omic_std.index, omic_name] = pte
                predictions_dict_test_lf["XGBoost"].fillna(np.median(predictions_dict_train_lf["XGBoost"]), inplace=True)
            _save_tree_importances(best, "XGBoost")

        if "CatBoost" in models and cb is not None:
            best, ptr, pte = _fit_predict_tree(cb, X_omic_std, y_omic, X_test_omic_std if X_test is not None else None)
            predictions_dict_train_lf["CatBoost"].loc[X_omic_std.index, omic_name] = ptr
            if X_test is not None:
                predictions_dict_test_lf["CatBoost"].loc[X_test_omic_std.index, omic_name] = pte
                predictions_dict_test_lf["CatBoost"].fillna(np.median(predictions_dict_train_lf["CatBoost"]), inplace=True)
            _save_tree_importances(best, "CatBoost")

        if "LightGBM" in models and lgb is not None:
            best, ptr, pte = _fit_predict_tree(lgb, X_omic_std, y_omic, X_test_omic_std if X_test is not None else None)
            predictions_dict_train_lf["LightGBM"].loc[X_omic_std.index, omic_name] = ptr
            if X_test is not None:
                predictions_dict_test_lf["LightGBM"].loc[X_test_omic_std.index, omic_name] = pte
                predictions_dict_test_lf["LightGBM"].fillna(np.median(predictions_dict_train_lf["LightGBM"]), inplace=True)
            _save_tree_importances(best, "LightGBM")

    # --- Fit final sur TOUTES les features sélectionnées de chaque modèle STABL ---
    final_prepro = Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("std", StandardScaler())])

    for model in filter(lambda x: x.startswith("STABL "), models):
        feats = selected_features_dict[model]
        if len(feats) > 0:
            X_train = X_tot[feats]
            X_train_std = pd.DataFrame(
                data=final_prepro.fit_transform(X_train),
                index=X_train.index,
                columns=final_prepro.get_feature_names_out()
            )

            # Choix du modèle final
            if task_type == "binary":
                if model_chosen == "xgboost":
                    if XGBClassifier is None:
                        raise ImportError("xgboost non disponible, impossible d'utiliser model_chosen='xgboost'.")
                    base_model = XGBClassifier(
                        eval_metric="logloss",
                        n_estimators=200, random_state=42, n_jobs=1
                    )
                else:
                    base_model = LogisticRegression(penalty=None, class_weight="balanced", max_iter=int(1e6), random_state=42)

            else:  # regression
                if model_chosen == "rf" and rf is not None:
                    base_model = clone(getattr(rf, "best_estimator_", rf))
                elif model_chosen == "xgboost" and xgb is not None:
                    base_model = clone(getattr(xgb, "best_estimator_", xgb))
                else:
                    base_model = LinearRegression()

            # fit final
            base_model.fit(X_train_std, y)

            # Sauvegarde importances/coeffs du modèle final
            if hasattr(base_model, "coef_"):
                importance = base_model.coef_.flatten()
                metric_name = "Associated weight"
            elif hasattr(base_model, "feature_importances_"):
                importance = base_model.feature_importances_
                metric_name = "Feature importance"
            else:
                importance, metric_name = None, None

            if importance is not None:
                df_imp = pd.DataFrame({"Feature": feats, metric_name: importance}, index=None).set_index("Feature")
                df_imp.to_csv(Path(save_path, "Training-Validation", f"{model} coefficients.csv"))

            # Prédictions sur validation (si fournie)
            if X_test is not None and y_test is not None:
                X_test_std = pd.DataFrame(
                    data=final_prepro.transform(X_test_tot[feats]),
                    index=X_test_tot.index,
                    columns=final_prepro.get_feature_names_out()
                )
                if task_type == "binary" and hasattr(base_model, "predict_proba"):
                    model_preds = base_model.predict_proba(X_test_std)[:, 1]
                else:
                    model_preds = base_model.predict(X_test_std)

                predictions_dict[model] = pd.Series(model_preds, index=y_test.index, name=f"{model} predictions")

        else:
            # Aucun feature sélectionné
            if X_test is not None and y_test is not None:
                base = 0.5 if task_type == "binary" else float(np.mean(y))
                predictions_dict[model] = pd.Series(np.full(y_test.shape[0], base), index=y_test.index, name=f"{model} predictions")

    # --- Late fusion (stacking) ---
    preds_lf = late_fusion_validation(
        predictions_train_dict=predictions_dict_train_lf,
        predictions_valid_dict=predictions_dict_test_lf if (X_test is not None and y_test is not None) else None,
        y=y, task_type=task_type,
        save_path=Path(save_path, "Training-Validation"),
        n_iter=n_iter_lf
    )
    if preds_lf is not None:
        for m, s in preds_lf.items():
            predictions_dict[m] = s

    # --- Early fusion (optionnel) ---
    if early_fusion:
        X_train_std = pd.DataFrame(
            data=preprocessing.fit_transform(X_tot),
            index=X_tot.index,
            columns=preprocessing.get_feature_names_out()
        )
        if X_test is not None and y_test is not None:
            X_test_std = pd.DataFrame(
                data=preprocessing.transform(X_test_tot),
                index=X_test_tot.index,
                columns=preprocessing.get_feature_names_out()
            )

        def _ef_fit_store(grid, name: str):
            m = clone(grid)
            try:
                m = m.fit(X_train_std, y, groups=groups).best_estimator_
            except Exception:
                m = m.fit(X_train_std, y).best_estimator_
            # sauvegarde des coeffs
            if hasattr(m, "coef_"):
                nz = np.where(m.coef_.flatten())
                feats = list(X_train_std.columns[nz])
                dfc = pd.DataFrame({"Feature": feats, "Associated weight": m.coef_.flatten()[nz]}).set_index("Feature")
                dfc.to_csv(Path(save_path, "Training-Validation", f"EF {name} coefficients.csv"))
            if X_test is not None and y_test is not None:
                preds = (m.predict_proba(X_test_std)[:, 1] if task_type == "binary" and hasattr(m, "predict_proba")
                         else m.predict(X_test_std))
                predictions_dict[f"EF {name}"] = pd.Series(preds, index=y_test.index, name=f"EF {name} predictions")

        if "EF Lasso" in models and lasso is not None:
            _ef_fit_store(lasso, "Lasso")
        if "EF ALasso" in models and alasso is not None:
            _ef_fit_store(alasso, "ALasso")
        if "EF ElasticNet" in models and en is not None:
            _ef_fit_store(en, "ElasticNet")

    # --- Scores/plots si validation fournie ---
    if X_test is not None and y_test is not None:
        table_of_scores = compute_scores_table(predictions_dict=predictions_dict, y=y_test, task_type=task_type)
        p_values = compute_pvalues_table(predictions_dict=predictions_dict, y=y_test, task_type=task_type)

        table_of_scores.to_csv(Path(save_path, "Training-Validation", "Scores on Validation.csv"))
        table_of_scores.to_csv(Path(save_path, "Summary", "Scores on Validation.csv"))

        p_values_path = Path(save_path, "Training-Validation", "p-values")
        os.makedirs(p_values_path, exist_ok=True)
        for m, p in p_values.items():
            p.to_csv(Path(p_values_path, f"{m}.csv"))

        save_plots(predictions_dict=predictions_dict, y=y_test, task_type=task_type, save_path=Path(save_path, "Training-Validation"))

    # log simple
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for m in models:
        print(f"Selected features for {m}: {len(selected_features_dict[m])}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    return predictions_dict

def multi_omic_stabl_cv_noe(
        data_dict,
        y,
        outer_splitter,
        estimators,
        task_type,
        model_chosen,
        models,
        save_path=None,
        importance_method: str = "weight",
        outer_groups=None,
        early_fusion=False,
        late_fusion=False,
        n_iter_lf=10000,
):
    if early_fusion:
        models += ["EF " + model for model in models if "STABL" not in model]

    lasso = estimators["lasso"]
    en = estimators["en"]
    alasso = estimators["alasso"]
    
    if 'rf' in estimators.keys():
        rf=estimators["rf"]
    if 'cb' in estimators.keys():
        cb=estimators["cb"]
    if 'lgb' in estimators.keys():
        lgb=estimators["lgb"]
    if 'xgb' in estimators.keys():
        xgb=estimators["xgb"]
    stabl_lasso = estimators["stabl_lasso"]
    stabl_en = estimators["stabl_en"]
    stabl_alasso = estimators["stabl_alasso"]
    if 'stabl_rf' in estimators.keys():
        stabl_rf=estimators["stabl_rf"]
    if 'stabl_cb' in estimators.keys():
        stabl_cb=estimators["stabl_cb"]
    if 'stabl_lgb' in estimators.keys():
        stabl_lgb=estimators["stabl_lgb"]
    if 'stabl_xgb' in estimators.keys():
        stabl_xgb=estimators["stabl_xgb"]
    
    # ------------------------------------------------------------------
    #  Injecte le paramètre "importance_method" dans chaque STABL arbre/boosting
    # ------------------------------------------------------------------
    for _name in ["stabl_rf", "stabl_cb", "stabl_lgb", "stabl_xgb"]:
        if _name in estimators:                         # s'il existe dans le dict
            try:
                estimators[_name].set_params(           # <‑‑ passe le paramètre
                    importance_method=importance_method
                )
            except ValueError:
                # pour les versions STABL antérieures : le paramètre n'existe pas,
                #   on ignore silencieusement (ou tu peux lever un avertissement)
                pass
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

    k = 1
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

        for omic_name, X_omic in data_dict.items():
            test_idx_tmp = X_omic.index.intersection(test_idx)
            X_tmp = X_omic.drop(index=test_idx, errors="ignore")
            X_test_tmp = X_omic.loc[test_idx_tmp]
            # Preprocessing of X_tmp
            X_tmp = remove_low_info_samples(X_tmp)
            y_tmp = y.loc[X_tmp.index]
            groups = outer_groups[X_tmp.index] if outer_groups is not None else None

            X_tmp_std = pd.DataFrame(
                data=preprocessing.fit_transform(X_tmp),
                index=X_tmp.index,
                columns=preprocessing.get_feature_names_out()
            )

            X_test_tmp_std = pd.DataFrame(
                data=preprocessing.transform(X_test_tmp),
                index=X_test_tmp.index,
                columns=preprocessing.get_feature_names_out()
            )

            # _STABL_
            if "STABL Lasso" in models:
                # fit STABL Lasso
                print("Fitting of STABL Lasso")
                stabl_lasso.fit(X_tmp_std, y_tmp, groups=groups)
                tmp_sel_features = list(stabl_lasso.get_feature_names_out())
                fold_selected_features["STABL Lasso"].extend(tmp_sel_features)
                print(
                    f"STABL Lasso finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL Lasso"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_lasso.min_fdr_
                stabl_features_dict["STABL Lasso"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_lasso.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl_lasso,
                        path=Path(save_path, "Training CV", f"STABL Lasso results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type
                    )

            if "STABL ALasso" in models:
                # fit STABL ALasso
                print("Fitting of STABL ALasso")
                stabl_alasso.fit(X_tmp_std, y_tmp, groups=groups)
                tmp_sel_features = list(stabl_alasso.get_feature_names_out())
                fold_selected_features["STABL ALasso"].extend(tmp_sel_features)
                print(
                    f"STABL ALasso finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL ALasso"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_alasso.min_fdr_
                stabl_features_dict["STABL ALasso"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_alasso.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl_alasso,
                        path=Path(save_path, "Training CV", f"STABL ALasso results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type
                    )

            if "STABL ElasticNet" in models:
                # fit STABL ElasticNet
                print("Fitting of STABL ElasticNet")
                stabl_en.fit(X_tmp_std, y_tmp, groups=groups)
                tmp_sel_features = list(stabl_en.get_feature_names_out())
                fold_selected_features["STABL ElasticNet"].extend(tmp_sel_features)
                print(
                    f"STABL ElasticNet finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL ElasticNet"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_en.min_fdr_
                stabl_features_dict["STABL ElasticNet"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_en.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl_en,
                        path=Path(save_path, "Training CV", f"STABL ElasticNet results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type
                    )
                    
            if "STABL RandomForest" in models:
                # fit STABL Lasso
                print("Fitting of STABL RandomForest")
                stabl_rf.fit(X_tmp_std, y_tmp, groups=groups)
                tmp_sel_features = list(stabl_rf.get_feature_names_out())
                fold_selected_features["STABL RandomForest"].extend(tmp_sel_features)
                print(
                    f"STABL RandomForest finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL RandomForest"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_rf.min_fdr_
                stabl_features_dict["STABL RandomForest"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_rf.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl_rf,
                        path=Path(save_path, "Training CV", f"STABL RandomForest results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type
                    )
            if "STABL CatBoost" in models:
                print("Fitting of STABL CatBoost")
                stabl_cb.fit(X_tmp_std, y_tmp, groups=groups)
                tmp_sel_features = list(stabl_cb.get_feature_names_out())
                fold_selected_features["STABL CatBoost"].extend(tmp_sel_features)
                print(
                    f"STABL CatBoost finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL CatBoost"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_cb.min_fdr_
                stabl_features_dict["STABL CatBoost"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_cb.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl_cb,
                        path=Path(save_path, "Training CV", f"STABL CatBoost results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type
                    )

            if "STABL XGBoost" in models:
                # fit STABL Lasso
                print("Fitting of STABL XGBoost")
                stabl_xgb.fit(X_tmp_std, y_tmp, groups=groups)
                tmp_sel_features = list(stabl_xgb.get_feature_names_out())
                fold_selected_features["STABL XGBoost"].extend(tmp_sel_features)
                print(
                    f"STABL XGBoost finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL XGBoost"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_xgb.min_fdr_
                stabl_features_dict["STABL XGBoost"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_xgb.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl_xgb,
                        path=Path(save_path, "Training CV", f"STABL XGBoost results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type
                    )

            if "STABL LightGBM" in models:
                print("Fitting of STABL LightGBM")
                stabl_lgb.fit(X_tmp_std, y_tmp, groups=groups)
                tmp_sel_features = list(stabl_lgb.get_feature_names_out())
                fold_selected_features["STABL LightGBM"].extend(tmp_sel_features)
                print(
                    f"STABL LightGBM finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL LightGBM"][omic_name].loc[f'Fold n°{k}', "min FDP+"]   = stabl_lgb.min_fdr_
                stabl_features_dict["STABL LightGBM"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_lgb.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl_lgb,
                        path=Path(save_path, "Training CV", f"STABL LightGBM results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type
                    )

            if "Lasso" in models:
                # _Lasso_
                print("Fitting of Lasso")
                model = clone(lasso)
                if task_type == "binary":
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std)
                tmp_sel_features = list(X_tmp_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                fold_selected_features["Lasso"].extend(tmp_sel_features)
                print(
                    f"Lasso finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict_late_fusion["Lasso"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

            if "ALasso" in models:
                # _ALasso_
                print("Fitting of ALasso")
                model = clone(alasso)
                if task_type == "binary":
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std)
                tmp_sel_features = list(X_tmp_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                fold_selected_features["ALasso"].extend(tmp_sel_features)
                print(
                    f"ALasso finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict_late_fusion["ALasso"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

            if "ElasticNet" in models:
                # _EN_
                print("Fitting of ElasticNet")
                model = clone(en)
                if task_type == "binary":
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std)
                tmp_sel_features = list(X_tmp_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                fold_selected_features["ElasticNet"].extend(tmp_sel_features)
                print(
                    f"ElasticNet finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict_late_fusion["ElasticNet"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions
        
            if "RandomForest" in models:
                # _Lasso_
                print("Fitting of RandomForest")
                model = clone(rf)
                if task_type == "binary":
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std)
                tmp_sel_features = list(X_tmp_std.columns[np.where(model.best_estimator_.feature_importances_.flatten())])
                fold_selected_features["RandomForest"].extend(tmp_sel_features)
                print(
                    f"RandomForest finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict_late_fusion["RandomForest"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions
            
            if "XGBoost" in models:
                # _Lasso_
                print("Fitting of XGBoost")
                model = clone(xgb)
                if task_type == "binary":
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std)
                tmp_sel_features = list(X_tmp_std.columns[np.where(model.best_estimator_.feature_importances_.flatten())])
                fold_selected_features["XGBoost"].extend(tmp_sel_features)
                print(
                    f"XGBoost finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict_late_fusion["XGBoost"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

            if "CatBoost" in models:
                # _CatBoost_
                print("Fitting of CatBoost")
                model = clone(cb)
                # entraînement + prédiction
                if task_type == "binary":
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std)
                # sélection des features dont l'importance > 0
                importances = model.best_estimator_.get_feature_importance()
                tmp_sel_features = list(X_tmp_std.columns[np.where(importances > 0)])
                fold_selected_features["CatBoost"].extend(tmp_sel_features)
                print(
                    f"CatBoost finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                # stockage des prédictions pour la fusion tardive
                predictions_dict_late_fusion["CatBoost"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

            if "LightGBM" in models:
                print("Fitting of LightGBM")
                model = clone(lgb)
                if task_type == "binary":
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std)
                importances = model.best_estimator_.feature_importances_
                tmp_sel_features = list(X_tmp_std.columns[np.where(importances > 0)])
                fold_selected_features["LightGBM"].extend(tmp_sel_features)
                print(
                    f"LightGBM finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict_late_fusion["LightGBM"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

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

                # _Final Models_
                if task_type == "binary":
                    if model_chosen == 'xgboost':
                        predictions = XGBClassifier().fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()                    
                    else : 
                        predictions = clone(logit).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()
                    
                elif task_type == "regression":
                    if model_chosen=='lin_reg':
                        predictions = clone(linreg).fit(X_train, y_train).predict(X_test)
                    elif model_chosen=='rf':
                        predictions = clone(randomforest).fit(X_train, y_train).predict(X_test)
                    elif model_chosen=='xgboost':
                        predictions = clone(xgboost).fit(X_train, y_train).predict(X_test)
                        
                else:
                    raise ValueError("task_type not recognized.")

                predictions_dict[model].loc[test_idx, f"Fold n°{k}"] = predictions

            else:
                if task_type == "binary":
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [0.5] * len(test_idx)

                elif task_type == "regression":
                
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [np.mean(y_train)] * len(test_idx)

                else:
                    raise ValueError("task_type not recognized.")
                
        # _late fusion_
        if late_fusion:
            preds_lf = late_fusion_cv(
                predictions_dict_late_fusion, y[test_idx], task_type,
                Path(save_path, "Training CV"), n_iter=n_iter_lf
            )
            for model in preds_lf:
                predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = preds_lf[model]

        # _EF Lasso_
        if early_fusion:
            X_train = X_tot.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X_tot.loc[test_idx]
            X_train_std = pd.DataFrame(
                data=preprocessing.fit_transform(X_train),
                columns=preprocessing.get_feature_names_out(),
                index=X_train.index
            )

            X_test_std = pd.DataFrame(
                data=preprocessing.transform(X_test),
                columns=preprocessing.get_feature_names_out(),
                index=X_test.index
            )
            groups = outer_groups[train_idx] if outer_groups is not None else None

            if "EF Lasso" in models:
                # _Lasso_
                print("Fitting of EF Lasso")
                model = clone(lasso)
                if task_type == "binary":
                    predictions = model.fit(X_train_std, y_train, groups=groups).predict_proba(X_test_std)[:, 1]
                else:
                    predictions = model.fit(X_train_std, y_train, groups=groups).predict(X_test_std)
                tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                fold_selected_features["EF Lasso"] = tmp_sel_features
                print(
                    f"EF Lasso finished on {omic_name} ({X_train_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict["EF Lasso"].loc[test_idx, f"Fold n°{k}"] = predictions

            if "EF ALasso" in models:
                # _ALasso_
                print("Fitting of EF ALasso")
                model = clone(alasso)
                if task_type == "binary":
                    predictions = model.fit(X_train_std, y_train, groups=groups).predict_proba(X_test_std)[:, 1]
                else:
                    predictions = model.fit(X_train_std, y_train, groups=groups).predict(X_test_std)
                tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                fold_selected_features["EF ALasso"] = tmp_sel_features
                print(
                    f"EF ALasso finished on {omic_name} ({X_train_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict["EF ALasso"].loc[test_idx, f"Fold n°{k}"] = predictions

            if "EF ElasticNet" in models:
                # _EN_
                print("Fitting of EF ElasticNet")
                model = clone(en)
                if task_type == "binary":
                    predictions = model.fit(X_train_std, y_train, groups=groups).predict_proba(X_test_std)[:, 1]
                else:
                    predictions = model.fit(X_train_std, y_train, groups=groups).predict(X_test_std)
                tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                fold_selected_features["EF ElasticNet"] = tmp_sel_features
                print(
                    f"EF ElasticNet finished on {omic_name} ({X_train_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict["EF ElasticNet"].loc[test_idx, f"Fold n°{k}"] = predictions

        print("~~~~~~~~~~~~~~~~~~~")
        for model in models:
            print(f"This fold: {len(fold_selected_features[model])} features selected for {model}")
            selected_features_dict[model].append(fold_selected_features[model])
        print("~~~~~~~~~~~~~~~~~~~\n")

        k += 1

    # _SAVING_RESULTS_
    print("Saving results...")
    if y.name is None:
        y.name = "outcome"

    summary_res_path = Path(save_path, "Summary")
    cv_res_path = Path(save_path, "Training CV")

    jaccard_matrix_dict = dict()
    formatted_features_dict = dict()

    for model in models:

        #jaccard_matrix_dict[model] = jaccard_matrix(selected_features_dict[model])

        formatted_features_dict[model] = pd.DataFrame(
            data={
                "Fold selected features": selected_features_dict[model],
                "Fold nb of features": [len(el) for el in selected_features_dict[model]]
            },
            index=[f"Fold {i}" for i in range(outer_splitter.get_n_splits(X=X_tot))]
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

    p_values = compute_pvalues_table(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        selected_features_dict=formatted_features_dict
    )

    table_of_scores.to_csv(Path(summary_res_path, "Scores training CV.csv"))
    table_of_scores.to_csv(Path(cv_res_path, "Scores training CV.csv"))

    p_values_path = Path(cv_res_path, "p-values")
    os.makedirs(p_values_path, exist_ok=True)
    for m, p in p_values.items():
        p.to_csv(Path(p_values_path, f"{m}.csv"))

    save_plots(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        save_path=cv_res_path
    )

    return predictions_dict
    

    
def multi_omic_stabl_cv_noe_test(
        data_dict,
        y,
        outer_splitter,
        estimators,
        task_type,
        model_chosen,
        models,
        save_path=None,
        outer_groups=None,
        early_fusion=False,
        late_fusion=False,
        n_iter_lf=10000,
        # NEW ↓↓↓
        fold_feats_path: str = None,
        use_existing_feats_only: bool = False,
):
    """
    Identique à multi_omic_stabl_cv_noe, mais si `use_existing_feats_only=True` ET `fold_feats_path` fourni :
      - on NE relance PAS la sélection (STABL ou baselines),
      - on charge les features déjà sélectionnées par fold depuis `fold_feats_path`,
      - on exécute uniquement la partie 'fit final' sur ces features (direct-fit),
      - on appelle en plus cv_on_existing_feats(...) pour un 'existing-fit' indépendant,
      - et on compare les deux (scores/plots).
    """

    if early_fusion:
        models += ["EF " + model for model in models if "STABL" not in model]

    # Récup des estimateurs comme dans ta version
    lasso  = estimators.get("lasso")
    en     = estimators.get("en")
    alasso = estimators.get("alasso")

    rf  = estimators.get("rf")
    cb  = estimators.get("cb")
    lgb = estimators.get("lgb")
    xgb = estimators.get("xgb")

    stabl_lasso  = estimators.get("stabl_lasso")
    stabl_en     = estimators.get("stabl_en")
    stabl_alasso = estimators.get("stabl_alasso")
    stabl_rf     = estimators.get("stabl_rf")
    stabl_cb     = estimators.get("stabl_cb")
    stabl_lgb    = estimators.get("stabl_lgb")
    stabl_xgb    = estimators.get("stabl_xgb")

    os.makedirs(Path(save_path, "Training CV"), exist_ok=True)
    os.makedirs(Path(save_path, "Summary"),     exist_ok=True)

    # Data totale concat multi-omic
    X_tot = pd.concat(data_dict.values(), axis="columns")

    predictions_dict       = {}
    selected_features_dict = {}
    stabl_features_dict    = {}

    for model in models:
        predictions_dict[model] = pd.DataFrame(data=None, index=y.index)
        selected_features_dict[model] = []
        stabl_features_dict[model] = {}
        for omic_name in data_dict.keys():
            if "STABL" in model:
                stabl_features_dict[model][omic_name] = pd.DataFrame(data=None, columns=["Threshold", "min FDP+"])

    # ============ NEW: mode "skip selection" = charge listes existantes ============
    if use_existing_feats_only:
        if fold_feats_path is None:
            raise ValueError("use_existing_feats_only=True, mais fold_feats_path est None. "
                             "Fournis le chemin d'un run qui contient 'Training CV/Selected Features {model}.csv'.")
        # Lecture des listes (par fold) pour chaque modèle STABL
        raw_sel = {}
        for m in models:
            if m.startswith("STABL "):
                csv_path = Path(fold_feats_path, "Training CV", f"Selected Features {m}.csv")
                if not csv_path.exists():
                    raise FileNotFoundError(f"Fichier introuvable pour {m}: {csv_path}")
                df = pd.read_csv(csv_path, index_col=0)
                # Colonne "Fold selected features" contient des listes sérialisées
                raw_sel[m] = [
                    eval(s) if isinstance(s, str) else s
                    for s in df["Fold selected features"]
                ]
        # On ne calculera PAS stabl_features_dict[...] (min_fdr, thresholds) en mode skip
    # ==============================================================================

    k = 1
    n_splits = outer_splitter.get_n_splits(X=X_tot, y=y, groups=outer_groups)

    for train, test in (tbar := tqdm(
            outer_splitter.split(X_tot, y, groups=outer_groups),
            total=n_splits,
            file=sys.stdout
    )):
        train_idx, test_idx = y.iloc[train].index, y.iloc[test].index
        groups_fold = outer_groups.loc[train_idx].values if outer_groups is not None else None
        tbar.set_description(f"{len(train_idx)} train samples, {len(test_idx)} test samples")

        # Conteneur pour late fusion si besoin
        predictions_dict_late_fusion = {}
        fold_selected_features = {m: [] for m in models}
        if not use_existing_feats_only:
            # Si on ne skippe pas: on prépare le dict pour stocker les prédictions par omic (non-STABL)
            for m in models:
                if "STABL" not in m and "EF" not in m:
                    predictions_dict_late_fusion[m] = {omic: pd.DataFrame(data=None, index=test_idx)
                                                       for omic in data_dict.keys()}

        # ---------------------- BRANCHE 1 : sélection + fit (COMME AVANT) ----------------------
        if not use_existing_feats_only:
            for omic_name, X_omic in data_dict.items():
                test_idx_tmp = X_omic.index.intersection(test_idx)
                X_tmp        = X_omic.drop(index=test_idx, errors="ignore")
                X_test_tmp   = X_omic.loc[test_idx_tmp]

                # Prétraitement omic
                X_tmp = remove_low_info_samples(X_tmp)
                y_tmp = y.loc[X_tmp.index]
                groups = outer_groups[X_tmp.index] if outer_groups is not None else None

                X_tmp_std = pd.DataFrame(
                    data=preprocessing.fit_transform(X_tmp),
                    index=X_tmp.index,
                    columns=preprocessing.get_feature_names_out()
                )
                X_test_tmp_std = pd.DataFrame(
                    data=preprocessing.transform(X_test_tmp),
                    index=X_test_tmp.index,
                    columns=preprocessing.get_feature_names_out()
                )

                # ----- STABL (Lasso / ALasso / EN / RF / CB / XGB / LGB) -----
                if "STABL Lasso" in models:
                    print("Fitting of STABL Lasso")
                    stabl_lasso.fit(X_tmp_std, y_tmp, groups=groups)
                    tmp = list(stabl_lasso.get_feature_names_out())
                    fold_selected_features["STABL Lasso"].extend(tmp)
                    stabl_features_dict["STABL Lasso"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_lasso.min_fdr_
                    stabl_features_dict["STABL Lasso"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_lasso.fdr_min_threshold_
                    if k == 1:
                        save_stabl_results(stabl=stabl_lasso,
                                           path=Path(save_path, "Training CV", f"STABL Lasso results on {omic_name}"),
                                           df_X=X_tmp_std, y=y_tmp, task_type=task_type)

                if "STABL ALasso" in models:
                    print("Fitting of STABL ALasso")
                    stabl_alasso.fit(X_tmp_std, y_tmp, groups=groups)
                    tmp = list(stabl_alasso.get_feature_names_out())
                    fold_selected_features["STABL ALasso"].extend(tmp)
                    stabl_features_dict["STABL ALasso"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_alasso.min_fdr_
                    stabl_features_dict["STABL ALasso"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_alasso.fdr_min_threshold_
                    if k == 1:
                        save_stabl_results(stabl=stabl_alasso,
                                           path=Path(save_path, "Training CV", f"STABL ALasso results on {omic_name}"),
                                           df_X=X_tmp_std, y=y_tmp, task_type=task_type)

                if "STABL ElasticNet" in models:
                    print("Fitting of STABL ElasticNet")
                    stabl_en.fit(X_tmp_std, y_tmp, groups=groups)
                    tmp = list(stabl_en.get_feature_names_out())
                    fold_selected_features["STABL ElasticNet"].extend(tmp)
                    stabl_features_dict["STABL ElasticNet"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_en.min_fdr_
                    stabl_features_dict["STABL ElasticNet"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_en.fdr_min_threshold_
                    if k == 1:
                        save_stabl_results(stabl=stabl_en,
                                           path=Path(save_path, "Training CV", f"STABL ElasticNet results on {omic_name}"),
                                           df_X=X_tmp_std, y=y_tmp, task_type=task_type)

                if "STABL RandomForest" in models and stabl_rf is not None:
                    print("Fitting of STABL RandomForest")
                    stabl_rf.fit(X_tmp_std, y_tmp, groups=groups)
                    tmp = list(stabl_rf.get_feature_names_out())
                    fold_selected_features["STABL RandomForest"].extend(tmp)
                    stabl_features_dict["STABL RandomForest"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_rf.min_fdr_
                    stabl_features_dict["STABL RandomForest"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_rf.fdr_min_threshold_
                    if k == 1:
                        save_stabl_results(stabl=stabl_rf,
                                           path=Path(save_path, "Training CV", f"STABL RandomForest results on {omic_name}"),
                                           df_X=X_tmp_std, y=y_tmp, task_type=task_type)

                if "STABL CatBoost" in models and stabl_cb is not None:
                    print("Fitting of STABL CatBoost")
                    stabl_cb.fit(X_tmp_std, y_tmp, groups=groups)
                    tmp = list(stabl_cb.get_feature_names_out())
                    fold_selected_features["STABL CatBoost"].extend(tmp)
                    stabl_features_dict["STABL CatBoost"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_cb.min_fdr_
                    stabl_features_dict["STABL CatBoost"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_cb.fdr_min_threshold_
                    if k == 1:
                        save_stabl_results(stabl=stabl_cb,
                                           path=Path(save_path, "Training CV", f"STABL CatBoost results on {omic_name}"),
                                           df_X=X_tmp_std, y=y_tmp, task_type=task_type)

                if "STABL XGBoost" in models and stabl_xgb is not None:
                    print("Fitting of STABL XGBoost")
                    stabl_xgb.fit(X_tmp_std, y_tmp, groups=groups)
                    tmp = list(stabl_xgb.get_feature_names_out())
                    fold_selected_features["STABL XGBoost"].extend(tmp)
                    stabl_features_dict["STABL XGBoost"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_xgb.min_fdr_
                    stabl_features_dict["STABL XGBoost"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_xgb.fdr_min_threshold_
                    if k == 1:
                        save_stabl_results(stabl=stabl_xgb,
                                           path=Path(save_path, "Training CV", f"STABL XGBoost results on {omic_name}"),
                                           df_X=X_tmp_std, y=y_tmp, task_type=task_type)

                if "STABL LightGBM" in models and stabl_lgb is not None:
                    print("Fitting of STABL LightGBM")
                    stabl_lgb.fit(X_tmp_std, y_tmp, groups=groups)
                    tmp = list(stabl_lgb.get_feature_names_out())
                    fold_selected_features["STABL LightGBM"].extend(tmp)
                    stabl_features_dict["STABL LightGBM"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_lgb.min_fdr_
                    stabl_features_dict["STABL LightGBM"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_lgb.fdr_min_threshold_
                    if k == 1:
                        save_stabl_results(stabl=stabl_lgb,
                                           path=Path(save_path, "Training CV", f"STABL LightGBM results on {omic_name}"),
                                           df_X=X_tmp_std, y=y_tmp, task_type=task_type)

                # ----- baselines non-STABL si tu en as besoin -----
                # (inchangé, je garde la structure de ta fonction)
                if "Lasso" in models and lasso is not None:
                    model = clone(lasso)
                    preds = (model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                             if task_type=="binary" else
                             model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std))
                    tmp = list(X_tmp_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                    fold_selected_features["Lasso"].extend(tmp)
                    predictions_dict_late_fusion["Lasso"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = preds

                if "ALasso" in models and alasso is not None:
                    model = clone(alasso)
                    preds = (model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                             if task_type=="binary" else
                             model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std))
                    tmp = list(X_tmp_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                    fold_selected_features["ALasso"].extend(tmp)
                    predictions_dict_late_fusion["ALasso"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = preds

                if "ElasticNet" in models and en is not None:
                    model = clone(en)
                    preds = (model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                             if task_type=="binary" else
                             model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std))
                    tmp = list(X_tmp_std.columns[np.where(model.best_estimator_.coef_.flatten())])
                    fold_selected_features["ElasticNet"].extend(tmp)
                    predictions_dict_late_fusion["ElasticNet"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = preds

                if "RandomForest" in models and rf is not None:
                    model = clone(rf)
                    preds = (model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                             if task_type=="binary" else
                             model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std))
                    tmp = list(X_tmp_std.columns[np.where(model.best_estimator_.feature_importances_.flatten())])
                    fold_selected_features["RandomForest"].extend(tmp)
                    predictions_dict_late_fusion["RandomForest"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = preds

                if "XGBoost" in models and xgb is not None:
                    model = clone(xgb)
                    preds = (model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                             if task_type=="binary" else
                             model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std))
                    tmp = list(X_tmp_std.columns[np.where(model.best_estimator_.feature_importances_.flatten())])
                    fold_selected_features["XGBoost"].extend(tmp)
                    predictions_dict_late_fusion["XGBoost"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = preds

                if "CatBoost" in models and cb is not None:
                    model = clone(cb)
                    preds = (model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                             if task_type=="binary" else
                             model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std))
                    importances = model.best_estimator_.get_feature_importance()
                    tmp = list(X_tmp_std.columns[np.where(importances > 0)])
                    fold_selected_features["CatBoost"].extend(tmp)
                    predictions_dict_late_fusion["CatBoost"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = preds

                if "LightGBM" in models and lgb is not None:
                    model = clone(lgb)
                    preds = (model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
                             if task_type=="binary" else
                             model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std))
                    importances = model.best_estimator_.feature_importances_
                    tmp = list(X_tmp_std.columns[np.where(importances > 0)])
                    fold_selected_features["LightGBM"].extend(tmp)
                    predictions_dict_late_fusion["LightGBM"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = preds

        # ---------------------- BRANCHE 2 : "skip selection" -> on injecte les listes ----------------------
        else:
            # On ne traite que les modèles STABL; les baselines sont ignorés en mode skip.
            for m in models:
                if m.startswith("STABL "):
                    feats_m = raw_sel[m][k-1]  # liste de features pour ce fold (index 0-based)
                    fold_selected_features[m] = feats_m

        # ---------------------- Fit final (identique à ta version) ----------------------
        for model_name in filter(lambda x: "STABL" in x, models):
            feats = fold_selected_features[model_name]
            X_train  = X_tot.loc[train_idx, feats] if len(feats)>0 else None
            X_test   = X_tot.loc[test_idx, feats]  if len(feats)>0 else None
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]

            if len(feats) > 0:
                std_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy="median")), ('std', StandardScaler())])
                X_train = pd.DataFrame(std_pipe.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
                X_test  = pd.DataFrame(std_pipe.transform(X_test),    index=X_test.index,  columns=X_test.columns)

                if task_type == "binary":
                    # même "fit actuel" par défaut que dans ta fonction (XGB simple)
                    predictions = clone(XGBClassifier(eval_metric="logloss",
                                                      n_estimators=200, random_state=42)).fit(X_train, y_train)\
                                  .predict_proba(X_test)[:, 1].flatten()
                elif task_type == "regression":
                    # garde ton dispatch; (linreg/randomforest/xgboost doivent être dans ton scope)
                    if model_chosen=='lin_reg':
                        predictions = clone(linreg).fit(X_train, y_train).predict(X_test)
                    elif model_chosen=='rf':
                        predictions = clone(randomforest).fit(X_train, y_train).predict(X_test)
                    elif model_chosen=='xgboost':
                        predictions = clone(xgboost).fit(X_train, y_train).predict(X_test)
                    else:
                        raise ValueError("Unknown model_chosen for regression.")
                else:
                    raise ValueError("task_type not recognized.")

                predictions_dict[model_name].loc[test_idx, f"Fold n°{k}"] = predictions
            else:
                # aucun feat -> baseline constante
                if task_type == "binary":
                    predictions_dict[model_name].loc[test_idx, f'Fold n°{k}'] = [0.5] * len(test_idx)
                elif task_type == "regression":
                    predictions_dict[model_name].loc[test_idx, f'Fold n°{k}'] = [np.mean(y_train)] * len(test_idx)
                else:
                    raise ValueError("task_type not recognized.")

        # Late fusion (si non skip et demandé)
        if late_fusion and not use_existing_feats_only:
            preds_lf = late_fusion_cv(predictions_dict_late_fusion, y[test_idx], task_type,
                                      Path(save_path, "Training CV"), n_iter=n_iter_lf)
            for m in preds_lf:
                predictions_dict[m].loc[test_idx, f'Fold n°{k}'] = preds_lf[m]

        # Early fusion (si non skip et demandé)
        if early_fusion and not use_existing_feats_only:
            X_train = X_tot.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test  = X_tot.loc[test_idx]
            X_train_std = pd.DataFrame(preprocessing.fit_transform(X_train),
                                       columns=preprocessing.get_feature_names_out(), index=X_train.index)
            X_test_std  = pd.DataFrame(preprocessing.transform(X_test),
                                       columns=preprocessing.get_feature_names_out(), index=X_test.index)
            groups_fold = outer_groups[train_idx] if outer_groups is not None else None

            if "EF Lasso" in models and lasso is not None:
                model = clone(lasso)
                preds = (model.fit(X_train_std, y_train, groups=groups_fold).predict_proba(X_test_std)[:, 1]
                         if task_type=="binary" else
                         model.fit(X_train_std, y_train, groups=groups_fold).predict(X_test_std))
                predictions_dict["EF Lasso"].loc[test_idx, f"Fold n°{k}"] = preds

            if "EF ALasso" in models and alasso is not None:
                model = clone(alasso)
                preds = (model.fit(X_train_std, y_train, groups=groups_fold).predict_proba(X_test_std)[:, 1]
                         if task_type=="binary" else
                         model.fit(X_train_std, y_train, groups=groups_fold).predict(X_test_std))
                predictions_dict["EF ALasso"].loc[test_idx, f"Fold n°{k}"] = preds

            if "EF ElasticNet" in models and en is not None:
                model = clone(en)
                preds = (model.fit(X_train_std, y_train, groups=groups_fold).predict_proba(X_test_std)[:, 1]
                         if task_type=="binary" else
                         model.fit(X_train_std, y_train, groups=groups_fold).predict(X_test_std))
                predictions_dict["EF ElasticNet"].loc[test_idx, f"Fold n°{k}"] = preds

        # Log per-fold + stockage des listes
        print("~~~~~~~~~~~~~~~~~~~")
        for m in models:
            # En mode skip : fold_selected_features[m] est alimenté uniquement pour les modèles STABL
            nb = len(fold_selected_features[m]) if isinstance(fold_selected_features.get(m, []), list) else 0
            print(f"This fold: {nb} features selected for {m}")
            selected_features_dict[m].append(fold_selected_features.get(m, []))
        print("~~~~~~~~~~~~~~~~~~~\n")

        k += 1

    # ===== Sauvegarde des listes =====
    print("Saving results...")
    if y.name is None:
        y.name = "outcome"

    summary_res_path = Path(save_path, "Summary")
    cv_res_path      = Path(save_path, "Training CV")

    formatted_features_dict = {}
    for m in models:
        # On écrit quand même un CSV "Selected Features {m}.csv" dans CE run,
        # même en mode skip (il contiendra les listes chargées depuis fold_feats_path).
        dfm = pd.DataFrame(
            data={
                "Fold selected features": selected_features_dict[m],
                "Fold nb of features":   [len(el) if isinstance(el, list) else 0 for el in selected_features_dict[m]]
            },
            index=[f"Fold {i}" for i in range(n_splits)]
        )
        formatted_features_dict[m] = dfm
        dfm.to_csv(Path(cv_res_path, f"Selected Features {m}.csv"))

    # ===== NEW: comparaison avec cv_on_existing_feats =====
    existing_preds = {}
    existing_feats_dfs = {}

    if fold_feats_path is not None:
        print(f"[INFO] Comparing with cv_on_existing_feats using existing selections from: {fold_feats_path}")

        # mapping clés attendues par cv_on_existing_feats
        estimators_existing = dict(estimators)
        if "random_forest" not in estimators_existing and "rf" in estimators_existing:
            estimators_existing["random_forest"] = estimators_existing["rf"]
        if "xgboost" not in estimators_existing and "xgb" in estimators_existing:
            estimators_existing["xgboost"] = estimators_existing["xgb"]

        existing_save = Path(save_path, "ExistingFit")
        os.makedirs(existing_save, exist_ok=True)

        # Appel refit-only (lit les CSV du repo existant)
        existing_preds = cv_on_existing_feats(
            data_dict=data_dict, y=y, outer_splitter=outer_splitter,
            estimators=estimators_existing, task_type=task_type,
            model_chosen=model_chosen, models=models,
            fold_feats_path=fold_feats_path, save_path=existing_save,
            outer_groups=outer_groups, early_fusion=early_fusion,
            late_fusion=late_fusion, n_iter_lf=n_iter_lf, use_ega=False
        )

        # Charger ses CSV "Selected Features ..." pour les inclure aux tableaux
        existing_cv_dir = existing_save / "Training CV"
        for m in models:
            csv_path = existing_cv_dir / f"Selected Features {m}.csv"
            if csv_path.exists():
                df_sel = pd.read_csv(csv_path, index_col=0)
                existing_feats_dfs[f"{m} (existing-fit)"] = df_sel

    # ===== médianes des prédictions de CE run =====
    predictions_dict = {m: predictions_dict[m].median(axis=1) for m in predictions_dict.keys()}

    # ===== fusionner avec les prédictions existing-fit =====
    if existing_preds:
        for m, s in existing_preds.items():
            predictions_dict[f"{m} (existing-fit)"] = s

    # ===== scores & p-values (inclure aussi les DF existing-fit) =====
    combined_features_dict = dict(formatted_features_dict)
    combined_features_dict.update(existing_feats_dfs)

    table_of_scores = compute_scores_table(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        selected_features_dict=combined_features_dict
    )

    p_values = compute_pvalues_table(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        selected_features_dict=combined_features_dict
    )

    table_of_scores.to_csv(Path(summary_res_path, "Scores training CV.csv"))
    table_of_scores.to_csv(Path(cv_res_path,      "Scores training CV.csv"))

    p_values_path = Path(cv_res_path, "p-values")
    os.makedirs(p_values_path, exist_ok=True)
    for m, p in p_values.items():
        p.to_csv(Path(p_values_path, f"{m}.csv"))

    save_plots(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        save_path=cv_res_path
    )

    return predictions_dict
