#!/bin/sh

SRC=.
DST1=polixir@172.16.0.114:grl
DST2=polixir@172.16.0.99:grl
# DST2=ubuntu@36.111.131.41:~/grl
# DST3=ubuntu@172.16.0.114:~/grl

while true;
do
    rsync -avz --exclude logs* $SRC $DST1
    rsync -avz --exclude logs* $SRC $DST2
    # rsync -avz --exclude logs* outs $SRC $DST3
    # rsync -avz --exclude logs outs -e 'ssh -p 44139' $SRC $DST3
    sleep 3
done
