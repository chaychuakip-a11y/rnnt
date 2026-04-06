./raw_fea raw_fea.config fea | ./cmvn_simple 2 24 1 fea | ./fep_decoder_lattice_test -c atom_hadoop.cfg -mtn 1 -dtn 1 | ./selecttail wav mlf_sy fbnocmn40 mlf_fa_ph | ./randname
