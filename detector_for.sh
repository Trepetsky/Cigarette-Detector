#!/bin/bash

input_dir="image/"
output_dir="image_outs/"

mkdir -p $output_dir
total_files=$(ls $input_dir | wc -l)

count=0
for filename in $(ls $input_dir); do
  python3 detector.py --source "$input_dir$filename"
  mv "${input_dir}${filename%.*}_out.png" $output_dir
  count=$((count+1))
  echo "Processed $count/$total_files: $filename"
done | tqdm --total $total_files
