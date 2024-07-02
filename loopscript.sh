#!/bin/bash

for n in {1..8};
do
	python train.py --config configs/gnt_lensless.txt
	sleep 4h
done

