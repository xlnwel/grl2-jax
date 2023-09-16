#!/bin/bash

export PYTHONPATH=.
source activate chenxw
conda activate chenxw

args1=(1e-1)
args2=(
  # 50 
  # 50 
  100
  100
  200
  200
  400
  400
)
args3=(10)
state=100
ad=10

commands=()
for i in {0..5}; do
  la=${#args1[@]}
  idx=$(($i % $la))
  lr=${args1[$idx]}
  la=${#args2[@]}
  idx=$(($i % $la))
  h=${args2[$idx]}
  la=${#args3[@]}
  idx=$(($i % $la))
  e=${args3[$idx]}
  s=3
  py_script="python standalone/lka.py -lr $lr -i 100 -e $e -H $h -s $s --state_size $state --action_dims $ad $ad $ad"
  echo $py_script
  commands+=("$py_script")
done

printf '%s\n' "${commands[@]}" | xargs -I COMMAND -P 2 -L 1 bash -c COMMAND
