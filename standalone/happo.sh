#!/bin/bash

export PYTHONPATH=.
source activate chenxw
conda activate chenxw

args1=(1e-1 1e-2 1e-3)
args2=(3 3 3)
args3=(3 3 3)

for i in {0..2}; do
  lr=${args1[$i]}
  h=${args2[$i]}
  e=${args3[$i]}
  s=5
  py_script="python standalone/happo.py -lr $lr -i 1000 -e $e -H $h -s $s --state_size 5 --action_dims 4 4 4 &"
  echo $py_script
  eval $py_script
done

wait
