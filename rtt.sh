#!/bin/zsh

# remove tmp files every priod of time

while true
do
    rm -rf /tmp/ray
    rm -rf /tmp/__pycache__
    sleep 1d
done
