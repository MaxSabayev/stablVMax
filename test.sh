mkdir logs
var=($(python3 -c 'from stabl.EMS import parse_params; parse_params("./paramsNew.json")'))
cat ~/arrayDefaultLow.sh | sed "s/rep/${var[1]}/" | sbatch 
cat ~/arrayDefaultHigh.sh | sed "s/rep/${var[2]}/" | sbatch
