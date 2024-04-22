from .unionfind import UnionFind
import sys
from tqdm.autonotebook import tqdm
from .preprocessing import remove_low_info_samples
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn import clone
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

logit = LogisticRegression(penalty='none', class_weight="balanced", max_iter=int(1e6), random_state=42)
linreg = LinearRegression()

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
def single_omic_simple(
        data,
        y,
        outer_splitter,
        estimator,
        estimator_name,
        preprocessing,
        task_type,
        outer_groups=None,
        sgl_corr =90
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

    sgl_corr_percentile: float, default=90
        Correlation threshold to use for SGL or STABL SGL.

    Returns
    -------
    predictions: pandas DataFrame
        DataFrame containing the predictions of the model for each sample in Cross-Validation.
    """


    predictions = pd.DataFrame(data=None, index=y.index)
    selected_features= []
    stabl_features= pd.DataFrame(data=None, columns=["Threshold", "min FDP+"])
    best_params = []

    k = 1
    for train, test in (tbar := tqdm(
            outer_splitter.split(data, y, groups=outer_groups),
            total=outer_splitter.get_n_splits(X=data, y=y, groups=outer_groups),
            file=sys.stdout
    )):
        train_idx, test_idx = y.iloc[train].index, y.iloc[test].index
        groups = outer_groups.loc[train_idx].values if outer_groups is not None else None

        fold_selected_features = []

        test_idx_tmp = data.index.intersection(test_idx)
        X_tmp = data.drop(index=test_idx, errors="ignore")
        X_test_tmp = data.loc[test_idx_tmp]

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

        # __STABL__
        if "stabl" in estimator_name:
            estimator.fit(X_tmp_std, y_tmp, groups=groups)
            tmp_sel_features = list(estimator.get_feature_names_out())
            fold_selected_features.extend(tmp_sel_features)
            stabl_features.loc[f'Fold #{k}', "min FDP+"] = estimator.min_fdr_
            stabl_features.loc[f'Fold #{k}', "Threshold"] = estimator.fdr_min_threshold_



        if estimator_name in ["lasso", "alasso","en"]:
            model = clone(estimator)
            if task_type == "binary":
                pred = model.fit(X_tmp_std, y_tmp, groups=groups).predict_proba(X_test_tmp_std)[:, 1]
            else:
                pred = model.fit(X_tmp_std, y_tmp, groups=groups).predict(X_test_tmp_std)
            tmp_sel_features = list(X_tmp_std.columns[np.where(model.best_estimator_.coef_.flatten())])
            fold_selected_features.extend(tmp_sel_features)
            predictions.loc[test_idx_tmp, f"Fold #{k}"] = pred
            best_params.append(model.best_params_)


        if "stabl" in estimator_name:
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

        selected_features.append(fold_selected_features)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        
  
        k += 1

    # __SAVING_RESULTS__

    if y.name is None:
        y.name = "outcome"


    formatted_features = pd.DataFrame(
            data={
                "Fold selected features": selected_features,
                "Fold nb of features": [len(el) for el in selected_features]
            },
            index=[f"Fold {i}" for i in range(outer_splitter.get_n_splits(X=data,groups=outer_groups))]
        )

    predictions = predictions.median(axis=1)

    # if task_type == "binary":
    #     scores_columns = ["ROC AUC", "Average Precision", "N features", "CVS"]
    # elif task_type == "regression":
    #     scores_columns = ["R2", "RMSE", "MAE", "N features", "CVS"]
    
    # table_of_scores = []

    # for metric in scores_columns:
    #     if metric == "ROC AUC":
    #         model_roc = roc_auc_score(y, predictions)
    #         model_roc_CI = compute_CI(y, predictions, scoring="roc_auc")
    #         cell_value = f"{model_roc:.3f} [{model_roc_CI[0]:.3f}, {model_roc_CI[1]:.3f}]"

    #     elif metric == "Average Precision":
    #         model_ap = average_precision_score(y, predictions)
    #         model_ap_CI = compute_CI(y, predictions, scoring="average_precision")
    #         cell_value = f"{model_ap:.3f} [{model_ap_CI[0]:.3f}, {model_ap_CI[1]:.3f}]"

    #     elif metric == "N features":
    #         sel_features = formatted_features["Fold nb of features"]
    #         median_features = np.median(sel_features)
    #         iqr_features = np.quantile(sel_features, [.25, .75])
    #         cell_value = f"{median_features:.3f} [{iqr_features[0]:.3f}, {iqr_features[1]:.3f}]"

    #     elif metric == "CVS":
    #         jaccard_mat = jaccard_matrix(formatted_features["Fold selected features"], remove_diag=False)
    #         jaccard_val = jaccard_mat[np.triu_indices_from(
    #             jaccard_mat, k=1)]
    #         jaccard_median = np.median(jaccard_val)
    #         jaccard_iqr = np.quantile(jaccard_val, [.25, .75])
    #         cell_value = f"{jaccard_median:.3f} [{jaccard_iqr[0]:.3f}, {jaccard_iqr[1]:.3f}]"

    #     elif metric == "R2":
    #         model_r2 = r2_score(y, predictions)
    #         model_r2_CI = compute_CI(y, predictions, scoring="r2")
    #         cell_value = f"{model_r2:.3f} [{model_r2_CI[0]:.3f}, {model_r2_CI[1]:.3f}]"

    #     elif metric == "RMSE":
    #         model_rmse = np.sqrt(mean_squared_error(y, predictions))
    #         model_rmse_CI = compute_CI(y, predictions, scoring="rmse")
    #         cell_value = f"{model_rmse:.3f} [{model_rmse_CI[0]:.3f}, {model_rmse_CI[1]:.3f}]"

    #     elif metric == "MAE":
    #         model_mae = mean_absolute_error(y, predictions)
    #         model_mae_CI = compute_CI(y, preds, scoring="mae")
    #         cell_value = f"{model_mae:.3f} [{model_mae_CI[0]:.3f}, {model_mae_CI[1]:.3f}]"

    #     table_of_scores.append(cell_value)


    return predictions,formatted_features

        
