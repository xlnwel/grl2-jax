#!/bin/zsh

# remove tmp files every priod of time

while true
do
    rm -rf html-logs
    python run/html_plt.py $1 -p $2
    sleep 3m
done
