#! /usr/bin/env bash

data_dir="$1"
word_ali_file="$2"
out_dir="$3"

if [ $# -ne 3 ]; then
    echo "Usage: $0 data-dir word-ali-file out-dir"
    exit 1
fi

mkdir -p "$out_dir"

utt_num=0
last_utt_name=""
text_num=1

while read -r line; do
    IFS=$'\t' read -ra utt_data <<< "$line"

    utt_name="${utt_data[0]}"
    utt_spk="${utt_data[1]}"
    utt_source_wav="$(find "$data_dir" -name "${utt_name}.wav")"
    utt_start="${utt_data[2]}"
    utt_end="${utt_data[3]}"
    utt_text="${utt_data[4]}"

    if [ "$utt_name" != "$last_utt_name" ]; then
        last_utt_name="$utt_name"
        ((utt_num+=1))
        text_num=1
    fi

    subout_dir="${out_dir}/utt_${utt_num}"
    mkdir -p "$subout_dir"

    word_utt_id="$(printf "utt%05d_word%02d_%s_%s_%08.f_%08.f_%s" "$utt_num" "$text_num" "$utt_text" "$utt_spk" "$(bc -l <<< "$utt_start * 100")" "$(bc -l <<< "$utt_end * 100")" "$utt_file")"
    if ! sox "$utt_source_wav" -b 16 -r 16k -c 1 "$subout_dir/${word_utt_id}.wav" trim "$utt_start" ="$utt_end"; then
        echo "$utt_start"
        echo "$utt_end"
        exit 3
    fi
    ((text_num+=1))
done < "$word_ali_file"