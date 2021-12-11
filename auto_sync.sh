#!/bin/zsh

SRC=.
DST1=ubuntu@36.111.131.39:~/grl
DST2=ubuntu@36.111.131.41:~/grl
DST3=ubuntu@36.111.128.2:~/grl

while true;
do
    rsync -avz --exclude logs $SRC $DST1
    rsync -avz --exclude logs $SRC $DST2
    rsync -avz --exclude logs -e 'ssh -p 44139' $SRC $DST3
    sleep 3s
done
