#!/bin/bash

export PYTHONPATH=.
source activate chenxw
conda activate chenxw

args1=(1e-1 1e-1)
args2=(5 10)
args3=(4 4)

for i in {0..1}; do
    lr=${args1[$i]}
    h=${args2[$i]}
    e=${args3[$i]}
    s=5
    py_script="python standalone/lka.py -lr $lr -i 5000 -e $e -H $h -s $s --state_size 5 --action_dims 4 4 4 &"
    echo $py_script
    eval $py_script
done

wait
