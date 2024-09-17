from .unionfind import UnionFind
import sys
from tqdm.autonotebook import tqdm
from .preprocessing import remove_low_info_samples
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn import clone
from scipy.optimize import nnls
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error, mean_absolute_error
from .utils import compute_CI
from .metrics import jaccard_matrix
from pathlib import Path

logit = LogisticRegression(penalty=None, class_weight="balanced", max_iter=int(1e6))
linreg = LinearRegression()
std_pipe = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy="median")),
        ('std', StandardScaler())
    ]
)

fromPreprocessing = lambda data,prepro: pd.DataFrame(data=prepro.fit_transform(data),index=data.index,columns=prepro.get_feature_names_out())
fromPreprocessingRep = lambda data,prepro: pd.DataFrame(data=prepro.transform(data),index=data.index,columns=prepro.get_feature_names_out())

@ignore_warnings(category=ConvergenceWarning)
def single_omic_simple(
        data,
        y,
        outer_splitter,
        estimator,
        estimator_name,
        preprocessing,
        task_type,
        ef = False,
        outer_groups=None
):
    """
    Performs a cross validation on the data_dict using the models and saves the results in save_path.

    Parameters
    ----------
    data: pd.DataFrame
        pandas DataFrames containing the data

    y: pd.Series
        pandas Series containing the outcomes for the use case. Note that y should contain the union of outcomes for
        the data_dict.

    outer_splitter: sklearn.model_selection._split.BaseCrossValidator
        Outer cross validation splitter

    estimator : sklearn estimator
        One of the following, with the corresponding estimator_name:
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

    outer_groups: pd.Series, default=None
        If used, should be the same size as y and should indicate the groups of the samples.

    ef: bool, default=False
        If True, doesnt return the insample predictions for non-stabl models.

    Returns
    -------
    predictions: pandas DataFrame
        DataFrame containing the predictions of the model for each sample in Cross-Validation.
    """

    stablFlag = "stabl" in estimator_name
    foldIdx = [f"Fold_{i}" for i in range(outer_splitter.get_n_splits(X=data, y=y, groups=outer_groups))]

    predictions = pd.DataFrame(data=None, index=y.index, columns=foldIdx)
    selected_features= pd.DataFrame(data=False, columns=data.columns, index=foldIdx)
    if stablFlag:
        stabl_features= pd.DataFrame(data=None, columns=["Threshold", "min FDP+"], index=foldIdx)
    else:
        best_params = []
        if not ef:
            insamplePredictions = pd.DataFrame(data=None, index=y.index, columns=foldIdx)

    k = 1
    for train, test in (tbar := tqdm(
            outer_splitter.split(data, y, groups=outer_groups),
            total=outer_splitter.get_n_splits(X=data, y=y, groups=outer_groups),
            file=sys.stdout
    )):
        train_idx, test_idx = y.iloc[train].index, y.iloc[test].index
        groups = outer_groups.iloc[train].values if outer_groups is not None else None

        fold_selected_features = []

        X_train = data.iloc[train,:]
        X_test = data.iloc[test,:]
        y_train = y.iloc[train]
        y_test = y.iloc[test]

        # X_train = remove_low_info_samples(X_train)
        # y_train = y_train.loc[X_train.index]
        # groups = outer_groups[X_tmp.index] if outer_groups is not None else None

        X_train_std = fromPreprocessing(X_train,preprocessing)
        X_test_std = fromPreprocessingRep(X_test,preprocessing)

        if estimator_name in ["lasso", "alasso","en"]:
            model = clone(estimator)
            model.fit(X_train_std, y_train, groups=groups)
            if task_type == "binary":
                pred = model.predict_proba(X_test_std)[:, 1]
                if not ef:
                    insamplePreds = model.predict_proba(X_train_std)[:, 1]
            else:
                pred = model.predict(X_test_std)
                if not ef:
                    insamplePreds = model.predict(X_train_std)
            if not ef:
                insamplePredictions.loc[train_idx,f'Fold_{k}'] = insamplePreds
            tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
            fold_selected_features.extend(tmp_sel_features)
            predictions.loc[test_idx, f"Fold_{k}"] = pred
            best_params.append(model.best_params_)

        # __STABL__
        if stablFlag:
            estimator.fit(X_train_std, y_train, groups=groups)
            tmp_sel_features = list(estimator.get_feature_names_out())
            fold_selected_features.extend(tmp_sel_features)
            stabl_features.loc[f'Fold_{k}', "min FDP+"] = estimator.min_fdr_
            stabl_features.loc[f'Fold_{k}', "Threshold"] = estimator.fdr_min_threshold_

            X_train = X_train[fold_selected_features]
            X_test = X_test[fold_selected_features]

            if len(fold_selected_features) > 0:
                # Standardization
                X_train = fromPreprocessing(X_train,std_pipe)
                X_test = fromPreprocessingRep(X_test,std_pipe)
                # __Final Models__
                if task_type == "binary":
                    pred = clone(logit).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()
                elif task_type == "regression":
                    pred = clone(linreg).fit(X_train, y_train).predict(X_test)
                else:
                    raise ValueError("task_type not recognized.")

                predictions.loc[test_idx, f"Fold_{k}"] = pred

            else:
                if task_type == "binary":
                    predictions.loc[test_idx, f'Fold_{k}'] = [0.5] * len(test_idx)
                elif task_type == "regression":
                    predictions.loc[test_idx, f'Fold_{k}'] = [np.mean(y_train)] * len(test_idx)

                else:
                    raise ValueError("task_type not recognized.")

        selected_features.loc[f'Fold_{k}', fold_selected_features] = True
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        k += 1

    if stablFlag:
        return 1,predictions,selected_features,stabl_features
    else:
        if ef:
            return 2,predictions,selected_features,best_params
        return 0,predictions,selected_features,best_params, insamplePredictions

def save_single_omic_results(results,savePath):
    match results[0]:
        case 0:
            preds,selectedFeats,bestParams,insamplePredictions = results[1:]
            stablFeats = None
        case 1:
            preds,selectedFeats,stablFeats = results[1:]
            bestParams = None
            insamplePredictions = None
        case 2:
            preds,selectedFeats,bestParams = results[1:]
            stablFeats = None
            insamplePredictions = None
    preds.to_csv(Path(savePath,"cvPreds.csv"))
    selectedFeats.to_csv(Path(savePath,"selectedFeats.csv"))
    if bestParams is not None:
        pd.DataFrame(bestParams).to_csv(Path(savePath,"bestParams.csv"))
    if insamplePredictions is not None:
        insamplePredictions.to_csv(Path(savePath,"insamplePreds.csv"))
    if stablFeats is not None:
        stablFeats.to_csv(Path(savePath,"stablFeats.csv"))
    featCount = {}
    for feat in selectedFeats.values.flatten():
        if feat != np.nan:
            if feat in featCount:
                featCount[feat] += 1
            else:
                featCount[feat] = 1
    featCount = pd.DataFrame.from_dict(featCount,orient='index').sort_values(by=0,ascending=False)
    featCount.to_csv(Path(savePath,"featCount.csv"))
    
    
    

        
def late_fusion_combination_normal(
        data,
        y,
        oosPredictions,
        isPredictions,
        outer_splitter,
        outer_groups=None
    ):
    """
    data : pd.DataFrame
        pandas DataFrame containing the original data
    oosPredictions : list of pd.Series
        for each omic, the pandas DataFrame containing the oos predictions
    isPredictions : list of pd.DataFrames
        for each omic, the in-sample predictions over each of the folds of the crosvalidation

    """
    predictions = pd.DataFrame(data=None, index=y.index)
    k = 1
    for train, test in (tbar := tqdm(
            outer_splitter.split(data, y, groups=outer_groups),
            total=outer_splitter.get_n_splits(X=data, y=y, groups=outer_groups),
            file=sys.stdout
    )):
        train_idx, test_idx = y.iloc[train].index, y.iloc[test].index

        foldData = pd.concat([pd.Series(pred[k-1]) for pred in isPredictions], axis=1).fillna(0.5).to_numpy()
        # foldData = np.nan_to_num(np.stack([ np.array(pred[k-1]) for pred in isPredictions]))
        foldY = pd.concat([pred.loc[test_idx, f'Fold #{k}'] for pred in oosPredictions], axis=1).fillna(0.5).to_numpy()

        yFit = y.loc[train_idx].to_numpy().flatten()
        # print(foldData.shape, foldY.shape, yFit.shape, np.sum(np.isnan(foldData)), np.sum(np.isnan(foldY)),np.sum(np.isnan(yFit)))
        beta,_ = nnls(foldData, yFit)
        print(beta)
        prediction = foldY @ beta

        predictions.loc[test_idx, f"Fold #{k}"] = prediction

        k += 1
    
    return predictions

def late_fusion_combination_stabl(
        data,
        y,
        selected_features,
        outer_splitter,
        task_type,
        outer_groups=None
    ):
    """
    data : pd.DataFrame
        pandas DataFrame containing the original data
    selected_features : list of lists
        for each omic, the list of selected features over each of the folds of the crosvalidation
    """
    predictions = pd.DataFrame(data=None, index=y.index)
    total_features = []
    k = 1
    for train, test in (tbar := tqdm(
            outer_splitter.split(data, y, groups=outer_groups),
            total=outer_splitter.get_n_splits(X=data, y=y, groups=outer_groups),
            file=sys.stdout
    )):
        train_idx, test_idx = y.iloc[train].index, y.iloc[test].index

        fold_selected_features = []
        for omic in selected_features:
            fold_selected_features.extend(omic[k-1])
        total_features.append(fold_selected_features)

        X_train = data.loc[train_idx, fold_selected_features]
        X_test = data.loc[test_idx, fold_selected_features]
        y_train = y.loc[train_idx]

        if len(fold_selected_features) > 0:
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
                pred = clone(logit).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()
                print(pred, y.loc[test_idx].to_numpy().flatten())


            elif task_type == "regression":
                pred = clone(linreg).fit(X_train, y_train).predict(X_test)

            else:
                raise ValueError("task_type not recognized.")

            predictions.loc[test_idx, f"Fold #{k}"] = pred

        else:
            if task_type == "binary":
                predictions.loc[test_idx, f'Fold #{k}'] = [0.5] * len(test_idx)

            elif task_type == "regression":
                predictions.loc[test_idx, f'Fold #{k}'] = [np.mean(y_train)] * len(test_idx)

            else:
                raise ValueError("task_type not recognized.")
        

        k += 1


    formatted_features = pd.DataFrame(
            data={
                "Fold selected features": total_features,
                "Fold nb of features": [len(el) for el in total_features]
            },
            index=[f"Fold {i}" for i in range(outer_splitter.get_n_splits(X=data,groups=outer_groups))]
        )
    
    return predictions,formatted_features




def simpleScores(
        predictions,
        y,
        features,
        task_type
):
    predictions = predictions.median(axis=1)

    if task_type == "binary":
        scores_columns = ["ROC AUC", "Average Precision", "N features", "CVS"]
    elif task_type == "regression":
        scores_columns = ["R2", "RMSE", "MAE", "N features", "CVS"]
    
    table_of_scores = []

    for metric in scores_columns:
        if metric == "ROC AUC":
            model_roc = roc_auc_score(y, predictions)
            model_roc_CI = compute_CI(y, predictions, scoring="roc_auc")
            cell_value = [f"{model_roc:.3f}",f"{model_roc_CI[0]:.3f}", f"{model_roc_CI[1]:.3f}"]

        elif metric == "Average Precision":
            model_ap = average_precision_score(y, predictions)
            model_ap_CI = compute_CI(y, predictions, scoring="average_precision")
            cell_value = [f"{model_ap:.3f}", f"{model_ap_CI[0]:.3f}", f"{model_ap_CI[1]:.3f}"]

        elif metric == "N features":
            sel_features = np.sum(features, axis=1)
            median_features = np.median(sel_features)
            iqr_features = np.quantile(sel_features, [.25, .75])
            cell_value = [f"{median_features:.3f}", f"{iqr_features[0]:.3f}", f"{iqr_features[1]:.3f}"]

        elif metric == "CVS":
            jaccard_mat = jaccard_matrix(features, remove_diag=False)
            jaccard_val = jaccard_mat[np.triu_indices_from(
                jaccard_mat, k=1)]
            jaccard_median = np.median(jaccard_val)
            jaccard_iqr = np.quantile(jaccard_val, [.25, .75])
            cell_value = [f"{jaccard_median:.3f}",f"{jaccard_iqr[0]:.3f}", f"{jaccard_iqr[1]:.3f}"]

        elif metric == "R2":
            model_r2 = r2_score(y, predictions)
            model_r2_CI = compute_CI(y, predictions, scoring="r2")
            cell_value = [f"{model_r2:.3f}",f"{model_r2_CI[0]:.3f}", f"{model_r2_CI[1]:.3f}"]

        elif metric == "RMSE":
            model_rmse = np.sqrt(mean_squared_error(y, predictions))
            model_rmse_CI = compute_CI(y, predictions, scoring="rmse")
            cell_value = [f"{model_rmse:.3f}", f"{model_rmse_CI[0]:.3f}", f"{model_rmse_CI[1]:.3f}"]

        elif metric == "MAE":
            model_mae = mean_absolute_error(y, predictions)
            model_mae_CI = compute_CI(y, predictions, scoring="mae")
            cell_value = [f"{model_mae:.3f}", f"{model_mae_CI[0]:.3f}", f"{model_mae_CI[1]:.3f}"]

        table_of_scores.extend(cell_value)

    table_of_scores = pd.DataFrame(data=table_of_scores, index=[ a + b for a in scores_columns for b in ["", " LB", " UB"]])
    return table_of_scores