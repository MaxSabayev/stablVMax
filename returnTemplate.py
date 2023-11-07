import os
import pandas as pd
from pathlib import Path
from stabl.pipelines_utils import save_plots, compute_scores_table, compute_pvalues_table

y = None
taskType = 'binary'

files = os.listdir(Path("./Results/Preds/"))

predDict = dict()
selFeatDict = dict()

for file in files:
    df = pd.read_csv(Path("./Results/Preds/",file),index_col=0)
    if file[:5] == "preds":
        print(file[6:])
        predDict[file[6:]] = df
    if file[:5] == "selec":
        print(file[14:])
        selFeatDict[file[14:]] = df



predDict = {model : predDict[model].median(axis=1) for model in predDict.keys() }

table_of_scores = compute_scores_table(
        predictions_dict=predDict,
        y=y,
        task_type=taskType,
        selected_features_dict=selFeatDict
    )

p_values = compute_pvalues_table(
        predictions_dict=predDict,
        y=y,
        task_type=taskType,
        selected_features_dict=selFeatDict
    )

table_of_scores.to_csv("./Results/CVscores.csv")

p_values_path = Path("./Results/", "p-values")
os.makedirs(p_values_path, exist_ok=True)
for m, p in p_values.items():
    p.to_csv(Path(p_values_path, f"{m}.csv"))

save_plots(
    predictions_dict=predDict,
    y=y,
    task_type=taskType,
    save_path="./Results/CVfigs/"
)




    
