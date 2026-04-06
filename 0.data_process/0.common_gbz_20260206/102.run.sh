for i in $(seq 1 9)
do
  echo "数字：$i"
  perl 102_get_pfile_from_hdfs.pl  $i &
done
