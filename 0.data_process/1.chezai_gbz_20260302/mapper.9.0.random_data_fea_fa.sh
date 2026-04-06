#!/bin/bash
./raw_fea config.fea.16K_offCMN_PowerFB40 fbnocmn40 | ./selecttail wav fbnocmn40 mlf_sy mlf_fa_ph | ./randname
