SRC=.
DST=ubuntu@36.111.131.39:~/grl
PASS=lfr2nA4pIJLgnj3
 
while true;
do
    ./run_rsync.sh $SRC $DST $PASS > auto_sync.log
    sleep 1s
