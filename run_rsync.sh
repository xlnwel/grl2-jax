#!/usr/bin/expect
 
set SRC [lindex $argv 0]
set DST [lindex $argv 1]
set PASS [lindex $argv 2]
 
#--delete参数删除多余文件
spawn rsync -vaz --inplace --progress $SRC $DST --delete logs
 
expect {
"yes/no" { send "yes\r";exp_continue }
"password:" { send "$PASS\r" }
}
 
expect eof
 
if [catch wait] {
    puts "rsync failed"
    exit 1
}
