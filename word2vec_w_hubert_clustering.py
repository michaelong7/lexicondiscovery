#! /usr/bin/env python

import numpy as np
import faiss
import editdistance
import jarowinkler

import networkx as nx

import argparse
from pathlib import Path

from tqdm import tqdm
from collections import Counter, defaultdict
from itertools import combinations

import gensim
import re

def phone_normalize(word):
    return word.replace("ɪ", "i").replace("ɛ", "e").replace("ɔ", "o").replace("ʊ", "u").replace("w", "v").replace("ː", "")

def cluster(audio_reps, text_reps, types, cluster_iter, new_audio_dim=5, text_dim=5, ncentroids=5, niter=80, verbose=True, ned_k=0.25, jaro_k=0.1, jaro_pref = 0.1):
    xb = np.vstack(audio_reps)
    faiss.normalize_L2(xb)
    orig_dim = xb.shape[1]

    mat = faiss.PCAMatrix(orig_dim, new_audio_dim)
    mat.train(xb)
    xb = mat.apply(xb)
    faiss.normalize_L2(xb)

    xb = np.concatenate((xb, np.vstack(text_reps)), axis=1)

    new_dim = new_audio_dim + text_dim

    index = faiss.IndexFlatL2(new_dim)
    index.add(xb)

    kmeans = faiss.Kmeans(new_dim, ncentroids, niter=niter, verbose=verbose, gpu=1)
    kmeans.train(xb)
    _, I = kmeans.index.search(xb, 1)
    cats_list = [i for j in I.tolist() for i in j]
    cats = dict()
    for i in range(len(types)):
        cats.setdefault(cats_list[i], []).append(types[i])

    subcat_lexicon = dict()
    G = nx.Graph()
    for cat in cats:
        if len(cats[cat]) > 1:
            cat_pairs = list(combinations(cats[cat], 2))
            for worda, wordb in cat_pairs:
                penalty = 0
                if worda[:1] != wordb[:1]:
                    penalty = 0.5
                jarowinkler_dist = 1 - jarowinkler.jarowinkler_similarity(phone_normalize(worda), phone_normalize(wordb), prefix_weight=jaro_pref)
                normed_edit_dist = (editdistance.eval(phone_normalize(worda), phone_normalize(wordb)) + penalty) / max(len(worda), len(wordb))
                if jarowinkler_dist < jaro_k and normed_edit_dist < ned_k:
                    G.add_edge(worda, wordb)
                G.add_edge(worda, worda)
                G.add_edge(wordb, wordb)
            subcats = list(nx.connected_components(G))
            for i in range(0, len(subcats)):
                subcat_lexicon[(cluster_iter, cat, i)] = list(subcats[i])
            G.clear()
        else:
            subcat_lexicon[(cluster_iter, cat, 0)] = [cats[cat][0]]

    return subcat_lexicon

def get_cluster_reps(words: list | dict):
    rec_audio_reps = list()
    rec_text_reps = list()
    rec_types = list()
    for word_id in words:
        rec_types.append(word_id)
        rec_audio_reps.append(np.mean(audio_reps[word_id], axis=0))
        rec_text_reps.append(word_model.wv[model_types.index(word_id)])
    return (rec_audio_reps, rec_text_reps, rec_types)

def write_lexicon_to_file(file_path, lexicon):
    with open(file_path, "a") as out:
        for cat in sorted(lexicon):
            cat_shown = False
            for word in lexicon[cat]:
                cat_str = "{:02d}".format(cat[0]) + "c" + "{:04d}".format(cat[1]) + "s" + "{:04d}".format(cat[2])
                # out.write(f"{cat_str}\t{' '.join(word)} _\n") # for spaces between characters + underscores added to line ends
                if not cat_shown:
                    out.write(f"{cat_str}\t{''.join(word)}\n")
                    cat_shown = True
                else:
                    out.write(f"\t\t\t{''.join(word)}\n")

def load_lexicon_from_file(lexicon_path):
    lexicon = dict()
    cat = list()
    max_iter = 0
    with open(lexicon_path, "r") as f:
        for line in f:
            x = line.strip().split("\t")
            if len(x) != 1:
                cat = list(map(int, re.split(r'[cs]', x[0])))
                lexicon.setdefault(tuple(cat), []).append(x[1])
                max_iter = cat[0]
            else:
                lexicon[tuple(cat)].append(x[0])

    return lexicon, max_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("means_training_feats", type=Path)
    parser.add_argument("means_utt_list", type=Path)
    parser.add_argument("-r", "--resume", action="store_true")
    parser.add_argument(
        "-l",
        "--lexicon_path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to existing lexicon file",
    )
    args = parser.parse_args()

    if args.resume and not args.lexicon_path:
        raise Exception("No lexicon file provided") 

    out_file = 'test_1500.txt'

    audio_reps = dict()
    
    with open(args.means_utt_list, "r") as g:
        utts = [utt.split() for utt in g]

    text_dim = 100
    word_model = gensim.models.Word2Vec(sentences=utts, vector_size=text_dim, window=5, min_count=1)
    model_types = list(word_model.wv.index_to_key)

    for path in tqdm(sorted(args.means_training_feats.glob("**/*.npy"))):
        x = np.load(path)
        word_id = path.stem.split("_")[2]
        audio_reps.setdefault(word_id, []).append(np.mean(x, axis=0))

    orig_audio_reps, text_reps, orig_types = get_cluster_reps(audio_reps)

    if not args.resume:
        lexicon = cluster(orig_audio_reps, text_reps, orig_types, 1, ncentroids=5, ned_k=0.2, jaro_k=0.1, jaro_pref=0.1, text_dim=text_dim)

        with open("test_real_cat_nums", "w") as out:
            out.write("0\t0\n")
        with open("test_single_cat_nums", "w") as out:
            out.write("0" + "\t" + str(len(model_types)) + "\n")

        current_iter = 1
    else:
        lexicon, current_iter = load_lexicon_from_file(args.lexicon_path)

    single_words = list()

    for i in range(current_iter + 1, 2):
        single_words.clear()
        real_cat_num = 0
        ned_k = 0.2 + ((i-1000)/10000)
        jaro_k = 0.1 + ((i-1000)/10000)
        jaro_pref = 0.1 + ((i-1000)/10000)

        for cat in list(lexicon):
            if (len(lexicon[cat]) == 1):
                single_words.append(lexicon.pop(cat)[0])
            else:
                real_cat_num += 1
        if not single_words:
            break
        with open("test_real_cat_nums", "a") as out:
            out.write(str(i - 1) + "\t" + str(real_cat_num) + "\n")
        with open("test_single_cat_nums", "a") as out:
            out.write(str(i - 1) + "\t" + str(len(single_words)) + "\n")
        rec_audio_reps, rec_text_reps, rec_types = get_cluster_reps(single_words)
        lexicon.update(cluster(rec_audio_reps, rec_text_reps, rec_types, i, ncentroids=5, ned_k=ned_k, jaro_k=jaro_k, jaro_pref=jaro_pref, text_dim=text_dim))

        if (i % 100 == 0):
            temp_lexicon = out_file + "_" + "iter_" + str(i)
            write_lexicon_to_file(temp_lexicon, lexicon)

    write_lexicon_to_file(out_file, lexicon)
    print(len(lexicon))

    # with open(lexicon_file, "w") as out:
    #     for key in clusters:
    #         out.write(f"{clusters[key]}\n")
