x = ($(python3 -c 'from stabl.sherlock import parse_params; parse_params("./params.json")'))
if ${x[0]}; then
    sbatch -W ./temp/arrayLow.sh &
fi
if ${x[1]}; then
    sbatch -W ./temp/arrayHigh.sh &
fi
wait
python3 ./sendOut.py 1

