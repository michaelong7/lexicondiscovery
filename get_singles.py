#! /usr/bin/env python

from collections import Counter
import sys

lexicon_file = sys.argv[1]
out_path = sys.argv[2]

lexicon = dict()

with open(lexicon_file, "r") as f:
    for line in f:
        cat, token = line.strip().rsplit("\t", maxsplit=1)
        lexicon[token] = cat

x = Counter(lexicon.values())

with open(out_path, "w") as out:
    for cat in x:
        if x[cat] == 1:
            out.write(f"{list(lexicon.keys())[list(lexicon.values()).index(cat)]}\n")
