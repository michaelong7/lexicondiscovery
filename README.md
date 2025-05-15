# Installing
## Pip
``` bash
pip install -r requirements.txt
```

# format
utts_file should be transcriptions separated by newlines for each utt

word_ali_file should be in the (tab-separated) format:
\[wav_name (no extension)\]	\[speaker\]	\[word_start_time\]	\[word_end_time\]	\[transcription\]
make sure that the file has a trailing newline

bash get_lexicon.sh data_dir word_ali_file ali_dir model_dir reps_dir layer utts_file
