python3 -c 'from stabl.sherlock import parse_params; parse_params("./params.json")'
if [ -d ./results/l/ ]; then
    sbatch -W ./temp/arrayLow.sh &
fi
if [ -d ./results/h/ ]; then
    sbatch -W ./temp/arrayHigh.sh &
fi
wait
sbatch -W ./temp/end.sh
echo "All done!"

