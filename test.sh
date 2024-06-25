python3 -c 'from stabl.sherlock import parse_params; parse_params("./params.json")'
sbatch -W ./temp/arrayLow.sh &
sbatch -W ./temp/arrayHigh.sh &
wait
python3 ./sendOut.py 1

