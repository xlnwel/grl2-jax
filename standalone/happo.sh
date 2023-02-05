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
    s=5
    py_script="python standalone/happo.py -lr $lr -i 100000 -H $h -s $s &"
    echo $py_script
    eval $py_script
done

wait
