#!/bin/zsh

source activate grl

date=$(date +"%m%d")
date=0825

while true
do
    rm -rf html-logs
    python run/html_plt.py $1 -p $2
    sleep 30m
done
