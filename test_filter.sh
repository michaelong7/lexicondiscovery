#! /usr/bin/env bash

words_file="$1"
lexicon_file="$2"

awk \
'BEGIN {
    cat_name = "";
    }
NF != 1 {
    cat_name = $1;
    print $0
    }
NF == 1 {
    print cat_name "\t" $1
    }' "$lexicon_file" |
grep -wFf "$words_file"