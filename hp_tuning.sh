#!/bin/bash

declare -a resBlocks=("0" "1" "2" "3" "4" "5")

for resBlock in "${resBlocks[@]}"
do
    python3 train.py --resBlocks $resBlock --outDir output_resBlock --epochs 7
done

