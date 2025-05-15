#! /usr/bin/env bash

data_dir="$1"
word_ali_file="$2"
ali_dir="$3"
model_dir="$4"
reps_dir="$5"
layer="$6"
utts_file="$7"

bash cut_out_words.sh "$data_dir" "$word_ali_file" "$ali_dir"

python extract_hubert.py -l "$layer" -o "$reps_dir" "$model_dir" "$ali_dir"

python word2vec_w_hubert_clustering.py "$reps_dir" "$utts_file"