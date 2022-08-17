#!/bin/zsh

# remove tmp files every priod of time

date=$(date +"%m%d")

while true
do
    rm -rf html-logs
    python run/html_plt.py $1 -p $2 -n $date
    sleep 10m
done
