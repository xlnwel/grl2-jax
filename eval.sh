#!/bin/zsh

if [ $# -eq 1 ]
then
    opp=rb
else
    opp=$2
fi

outfile=eval_$1_vs_$opp

python run/eval.py logs/card_gd/zero/$1 -n 10000 -nw 80 >& $outfile
