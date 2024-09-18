x=$(python3 -c 'from stabl.sherlock import parse_params; parse_params("./params.json")')
IFS=$'\n' read -r -d '' result1 result2 <<<"$x"
if [ "$result1" == 1 ]; then
    sbatch -W ./temp/arrayLow.sh &
fi
if [ "$result2" == 1 ]; then
    sbatch -W ./temp/arrayHigh.sh &
fi
wait
python3 ./sendOut.py 1

