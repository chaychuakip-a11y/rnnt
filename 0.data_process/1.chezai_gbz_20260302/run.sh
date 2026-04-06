#原始数据
perl 0.0.ori.rand.pl
perl 1.0.ubfa.ori.pl

perl pak.getsize.ori.pl

###加噪
hdfs dfs -cat /workdir/asrdictt/dasrdictt/hjwang11/chezai/french/french_chezai_gbz_20260302_rand/*-part-* | /work1/asrdictt/hjwang11/bin/fea_lab_lat_unpack_1 - .
perl 2.0.GenSeedMlf.pl ./out.record.scp ./noise/seed.mlf

cd noise
chmod 755 *
sh run_adapt.sh
sh run_chime.sh
sh run_dictt.sh
cd ..

perl 3.0_rm_head.pl

##变速
perl 4.0.speedup1.2.pl
perl 4.2.ubfa.speedup.pl

###小音量
perl 5.0.ampReduce.pl

###lsa
perl 6.0.LsaDenoise.pl
perl 6.2.ubfa.LsaDenoise.pl

###mae_open
perl 7.0.MaeDenoiseOpen.pl
perl 7.2.ubfa.MaeDenoiseOpen.pl

###mae_close
perl 8.0.MaeDenoiseClose.pl
perl 8.2.ubfa.MaeDenoiseClose.pl

perl 9.0.random_data_fea_fa.pl

perl pak.getsize.final.pl
