#!/bin/zsh

SRC=.
DST1=ubuntu@36.111.131.37:grl
DST2=ubuntu@36.111.131.39:grl
DST3=ubuntu@36.111.131.41:grl

while true;
do
    rsync -avz --exclude-from=exclude.list $SRC $DST1
    rsync -avz --exclude-from=exclude.list $SRC $DST2
    rsync -avz --exclude-from=exclude.list $SRC $DST3
    sleep 3
done
