#!/bin/bash
start=$1
end=$2
step=$3
data_path=$4
checkpoint_prefix=$5
beam=$6
max_tokens=$7
opts=$8

for (( i=$start; i<=$end; i+=$step ))
do
  echo "Evaluate BLEU at epoch number $i of the model checkpoint: $checkpoint_prefix/checkpoint$i.pt";
  echo "beam_size=$beam, max_tokens=$max_tokens";

  fairseq-generate $data_path \
	--path ${checkpoint_prefix}/checkpoint${i}.pt \
	--beam $beam \
	--max-tokens $max_tokens $opts > ${checkpoint_prefix}/result_checkpoint_${i}.txt

  echo "Done evaluation for epoch $i";

  
done
