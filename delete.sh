#!/bin/bash

dir="data/real_iconic_noface";

for entry in "$dir"/*
do
	rm -rf "$entry/images_4";
	if [ -d "$entry/images" ]; then
		rm -r "$entry/images";
	fi
done
