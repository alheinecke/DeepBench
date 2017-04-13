#!/bin/sh

rm -rf overfeat_conv_bench alexnet_conv_bench vgga_conv_bench deepbench_conv_bench googlenetv1_conv_bench
mv input.h input.h.bak

cp input_overfeat.h input.h
make clean && make
mv std_conv_bench overfeat_conv_bench

cp input_alexnet.h input.h
make clean && make
mv std_conv_bench alexnet_conv_bench

cp input_vgga.h input.h
make clean && make
mv std_conv_bench vgga_conv_bench

cp input_deepbench.h input.h
make clean && make
mv std_conv_bench deepbench_conv_bench

cp input_googlenetv1.h input.h
make clean && make
mv std_conv_bench googlenetv1_conv_bench

mv input.h.bak input.h
