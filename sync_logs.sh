#!/bin/zsh

SRC="$HOME/work/Polixir/nas/"
DST=""

while true;
do
    rsync -avz $1 $DST
    sleep 10m
done