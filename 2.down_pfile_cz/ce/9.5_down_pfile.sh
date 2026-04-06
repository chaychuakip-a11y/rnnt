part=($(seq 0 9))
for i in ${part[*]}
do
    echo $i
    perl 102_get_pfile_from_hdfs_ce.pl $i >102_get_pfile_from_hdfs_ce.$i.log 2>&1 &
done
