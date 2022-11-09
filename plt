#!/bin/zsh

source activate grl


while true
do
    date=$(date -d '+8 hour' +"%m%d")
    # rm -rf html-logs
    python run/html_plt.py $1 -p $2 #-d $date
    sleep 10m
done
