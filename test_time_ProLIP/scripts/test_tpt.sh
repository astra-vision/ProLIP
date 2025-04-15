#!/bin/bash

data_root='/lustre/fsn1/projects/rech/noo/uyd44xs/DATA'
testsets=$1
arch=RN50
bs=64
ctx_init=a_photo_of_a

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --ctx_init ${ctx_init}