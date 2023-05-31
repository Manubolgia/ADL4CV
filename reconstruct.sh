#!/bin/bash

while getopts i:v:n:m: flag
do
    case "${flag}" in
        i) input_path=${OPTARG};;
        v) views=${OPTARG};;
        n) n=${OPTARG};;
        m) mode=${OPTARG};;
    esac
done
input_dir="${input_path}/${views}"
mkdir "${input_dir}_$n"
mkdir "${input_dir}_$n/views"
cd "${input_dir}/views"
imgs_num=$(ls | wc -l)
let "imgs_num = $imgs_num / 5 - 1"
indexes=$(shuf -i 0-$imgs_num -n $n)


for index in $indexes
do
        file=$(printf "cam%06d.png" $index)
        ln -s ${input_dir}/views/$file ${input_dir}_$n/views/$file
        file_k=$(printf "cam%06d_k.txt" $index)
        ln -s ${input_dir}/views/$file_k ${input_dir}_$n/views/$file_k
        file_r=$(printf "cam%06d_r.txt" $index)
        ln -s ${input_dir}/views/$file_r ${input_dir}_$n/views/$file_r
        file_t=$(printf "cam%06d_t.txt" $index)
        ln -s ${input_dir}/views/$file_t ${input_dir}_$n/views/$file_t
        file_t=$(printf "cam%06d_normal.png" $index)
        ln -s ${input_dir}/views/$file_t ${input_dir}_$n/views/$file_t
done
ln -s ${input_dir}/bbox.txt ${input_dir}_$n/bbox.txt
cd ~/ADL4CV

python3 reconstruct.py --input_dir ${input_dir}_$n/views --input_bbox ${input_dir}_$n/bbox.txt

rm -r ${input_dir}_$n