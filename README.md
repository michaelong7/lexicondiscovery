# Installing
## Pip
``` bash
pip install -r requirements.txt
```

# running
``` bash
bash get_lexicon.sh data_dir word_ali_file ali_dir model_dir reps_dir layer utts_file
```

currently this writes the output lexicon file to test_1500.txt and writes stats to the files test_real_cat_nums / test_single_cat_nums

# format
utts_file should be transcriptions separated by newlines for each utt

word_ali_file should be in the (tab-separated) format:
\[wav_name (no extension)\]	\[speaker\]	\[word_start_time\]	\[word_end_time\]	\[transcription\]
make sure that the file has a trailing newline

test_real_cat_nums has the format:
\[iteration number\]	\[# of categories with more than one variant\]

test_single_cat_nums has the format:
\[iteration number\]	\[# of categories with exactly one variant\]

the output lexicon file has the format:
{iteration number}c{category number}s{subcategory number}	\[variant\]
\(the category number is determined by the k-means clustering process\)
\(the subcategory number is determined by the normalized edit distance + jaro-winkler distance between words in the same category\)

# lexicon file processing
``` bash
python get_singles.py lexicon_file out_path
```

get_singles.py outputs all variants that do not share a category with another variant

``` bash
bash test_filter.sh words_file lexicon_file
```
test_filter.sh filters the given lexicon file to only include variants from the given words file

the words file should have each word on a separate line