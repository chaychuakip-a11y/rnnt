# cat dict/french_ly1213h_ly2k_xw696h_openSRC412h_fz1655h_ttsMusicVideoPoi_v16.word_dict_phone_nosp.checked ../check_oov/all.oov.dict > french_ly1213h_ly2k_xw696h_openSRC412h_fz1655h_ttsMusicVideoPoi_v16.word_dict_phone_nosp.add_pred_20260302

# language=french  ###set
# perl /work1/asrdictt/hjwang11/sbin/dict/CheckDictWithHmmlist.pl french_ly1213h_ly2k_xw696h_openSRC412h_fz1655h_ttsMusicVideoPoi_v16.word_dict_phone_nosp.add_pred_20260302 /work1/asrdictt/hjwang11/sbin/dict/phoneset.$language >dict.fail

# sed "s/\t.*//g" french_ly1213h_ly2k_xw696h_openSRC412h_fz1655h_ttsMusicVideoPoi_v16.word_dict_phone_nosp.add_pred_20260302 > french_ly1213h_ly2k_xw696h_openSRC412h_fz1655h_ttsMusicVideoPoi_v16.word_dict_phone_nosp.add_pred_20260302.wlist

sh 0.ngramtrain.sh
perl 1.package_atom_addsp.pl
