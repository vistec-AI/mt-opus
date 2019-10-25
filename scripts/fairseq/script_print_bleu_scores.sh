start=$1
end=$2
path=$3
# e.g. script_print_b;eu_scores.sh 1 25 ./data/opentsubtitles_model/exp004-1.1/transformer_base
echo "Read result_checkpoint_<epoch_numver>.txt file from ${path}"

for (( i=$start; i<=$end; i+=1 ))
do
  echo "# Epoch ${i}";
  tail -n 1 $path/result_checkpoint_$i.txt
done