# %%
import sys
sys.path.append('../')
import datasets
import log_reg
from dataproc import extract_wvs
from dataproc import get_discharge_summaries
from dataproc import concat_and_split
from dataproc import build_vocab
from dataproc import vocab_index_descriptions
from dataproc import word_embeddings
# MIMIC_4_DIR='./mimicdata/physionet.org/files/mimiciv/2.2'
MIMIC_4_DIR = "/home/justjosh/Projects/PLM-ICD/data/mimic4"
MIMIC_4_SAVE_DIR='/home/justjosh/Projects/PLM-ICD/data/mimic4/mimic4_icd10'


import numpy as np
import pandas as pd

from collections import Counter, defaultdict
import csv
import math
import operator

# %% [markdown]
# Let's do some data processing in a much better way, with a notebook.
# 
# First, let's define some stuff.

# %%
# MIMIC_4_DIR = "/home/justjosh/Turing-Test/mimic-eval/data/"

# %%
Y = 'full' #use all available labels in the dataset for prediction
notes_file = f'{MIMIC_4_DIR}/note/discharge.csv' # raw note events downloaded from MIMIC-III
vocab_size = 'full' #don't limit the vocab size to a specific number
vocab_min = 3 #discard tokens appearing in fewer than this many documents

# %% [markdown]
# # Data processing

# %% [markdown]
# ## Combine diagnosis and procedure codes and reformat them

# %% [markdown]
# The codes in MIMIC-III are given in separate files for procedures and diagnoses, and the codes are given without periods, which might lead to collisions if we naively combine them. So we have to add the periods back in the right place.

# %%
dfproc = pd.read_csv(f'{MIMIC_4_DIR}/hosp/procedures_icd.csv',
                     dtype={"icd_code": str})
dfdiag = pd.read_csv(f'{MIMIC_4_DIR}/hosp/diagnoses_icd.csv',
                     dtype={"icd_code": str})

# %%
print(len(dfproc[dfproc['icd_version']==9]))
print(len(dfproc[dfproc['icd_version']==10]))
print(len(dfproc['icd_version']))

# %% [markdown]
# There are both icd9 and icd10 code version in mimic IV, so limit to only icd 10 code
# 

# %%
dfproc10=dfproc[dfproc['icd_version']==10]
dfdiag10=dfdiag[dfdiag['icd_version']==10]

# %%
n=5
dfproc10.head(n)

# %%
n=5
dfdiag10.head(n)

# %%
dfdiag10['absolute_code'] = dfdiag10['icd_code']
dfproc10['absolute_code'] = dfproc10['icd_code']

# %%
n=5
dfdiag10.head(n)

# %%
n=5
dfproc10.head(n)

# %%
dfcodes10 = pd.concat([dfdiag10, dfproc10])

# %%
n=5
dfcodes10.head(n)

# %%
dfcodes10.to_csv(f'{MIMIC_4_SAVE_DIR}/ALL_CODES.csv', index=False,
               columns=['subject_id', 'hadm_id', 'seq_num', 'absolute_code'],
               header=['subject_id', 'hadm_id', 'seq_num', 'ICD10_CODE'])

# %% [markdown]
# ## How many codes are there?

# %%
#In the full dataset (not just discharge summaries)
df = pd.read_csv('%s/ALL_CODES.csv' % MIMIC_4_SAVE_DIR, dtype={"ICD10_CODE": str})
len(df['ICD10_CODE'].unique())

# %%
n=5
df.head(n)

# %% [markdown]
# ## Tokenize and preprocess raw text

# %% [markdown]
# Preprocessing time!
# 
# This will:
# - Select only discharge summaries and their addenda
# - remove punctuation and numeric-only tokens, removing 500 but keeping 250mg
# - lowercase all tokens

# %%
"""
    Reads NOTEEVENTS file, finds the discharge summaries, preprocesses them and writes out the filtered dataset.
"""
import csv

from nltk.tokenize import RegexpTokenizer

from tqdm import tqdm

#retain only alphanumeric
tokenizer = RegexpTokenizer(r'\w+')

def write_discharge_summaries(out_file, data_dir):
    # notes_file = '%s/NOTEEVENTS.csv' % (data_dir)
    notes_file = f'{data_dir}/note/discharge.csv'
    print("processing notes file")
    with open(notes_file, 'r', encoding='utf-8') as csvfile:
        with open(out_file, 'w', encoding='utf-8') as outfile:
            print("writing to %s" % (out_file))
            outfile.write(','.join(['subject_id', 'hadm_id', 'charttime', 'text']) + '\n')
            notereader = csv.reader(csvfile)
            #header
            next(notereader)
            i = 0
            for line in tqdm(notereader):
                subj = int(line[1])
                category = line[3]
                if category == "DS":
                    note = line[7]
                    #tokenize, lowercase and remove numerics
                    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
                    text = '"' + ' '.join(tokens) + '"'
                    outfile.write(','.join([line[1], line[2], line[5], text]) + '\n')
                i += 1
    return out_file

# %%
#This reads all notes, selects only the discharge summaries, and tokenizes them, returning the output filename
disch_full_file = write_discharge_summaries(out_file=f"{MIMIC_4_SAVE_DIR}/disch_10_full.csv",data_dir= MIMIC_4_DIR)

# %% [markdown]
# Let's read this in and see what kind of data we're working with

# %%
df = pd.read_csv(f"{MIMIC_4_SAVE_DIR}/disch_10_full.csv", encoding='utf-8', engine='python')

# %%
n=5
df.head(n)

# %%
#How many admissions?
len(df['hadm_id'].unique())

# %%
print(len(df['subject_id'].unique()))
print(len(df))

# %%
#Tokens and types
types = set()
num_tok = 0
for row in df.itertuples():
    for w in row[4].split():
        types.add(w)
        num_tok += 1

# %%
print("Num types", len(types))
print("Num tokens", str(num_tok))

# %%
#Let's sort by SUBJECT_ID and HADM_ID to make a correspondence with the MIMIC-3 label file
df = df.sort_values(['subject_id', 'hadm_id'])

# %%
#Sort the label file by the same
dfl = pd.read_csv(f'{MIMIC_4_SAVE_DIR}/ALL_CODES.csv',dtype={"ICD10_CODE": str})
dfl = dfl.sort_values(['subject_id', 'hadm_id'])

# %%
len(df['hadm_id'].unique()), len(dfl['hadm_id'].unique())

# %% [markdown]
# ## Consolidate labels with set of discharge summaries

# %% [markdown]
# Looks like there were some HADM_ID's that didn't have discharge summaries, so they weren't included with our notes

# %%
#Let's filter out these HADM_ID's
hadm_ids = set(df['hadm_id'])
with open(f'{MIMIC_4_SAVE_DIR}/ALL_CODES.csv', 'r') as lf:
    with open(f'{MIMIC_4_SAVE_DIR}/ALL_CODES_filtered.csv','w') as of:
        w = csv.writer(of)
        w.writerow(['subject_id', 'hadm_id', 'icd10_code', 'admittime', 'dischtime'])
        r = csv.reader(lf)
        #header
        next(r)
        for i,row in enumerate(r):
            hadm_id = int(row[1])
            #print(hadm_id)
            #break
            if hadm_id in hadm_ids:
                w.writerow(row[:2] + [row[-1], '', ''])

# %% [markdown]
# There are also some HADM_ID's that didn't have labels, so they weren't included with our notes.
# We need to remove the note.

# %%
dfl = pd.read_csv(f'{MIMIC_4_SAVE_DIR}/ALL_CODES_filtered.csv',dtype={"icd10_code": str}, index_col=None)

# %%
len(dfl['hadm_id'].unique())

# %%
#Let's filter out these HADM_ID in the note but not in the label
hadm_lids = set(dfl['hadm_id'])
with open(f'{MIMIC_4_SAVE_DIR}/disch_10_full.csv', 'r', encoding='utf-8') as lf:
    with open(f'{MIMIC_4_SAVE_DIR}/disch_10_filtered.csv','w', encoding='utf-8') as of:
        w = csv.writer(of)
        w.writerow(['subject_id', 'hadm_id', 'charttime', 'text'])
        r = csv.reader(lf)
        #header
        next(r)
        for i,row in enumerate(r):
            hadm_id = int(row[1])
            #print(hadm_id)
            #break
            if hadm_id in hadm_lids:
                w.writerow(row)

# %%
n=5
dfl.head(n)

# %%
#we still need to sort it by HADM_ID
dfl = dfl.sort_values(['subject_id', 'hadm_id'])
dfl.to_csv(f'{MIMIC_4_SAVE_DIR}/ALL_CODES_filtered.csv', index=False)

# %% [markdown]
# ## Append labels to notes in a single file

# %%
#Now let's append each instance with all of its codes
#this is pretty non-trivial so let's use this script I wrote, which requires the notes to be written to file
df = pd.read_csv(f'{MIMIC_4_SAVE_DIR}/disch_10_filtered.csv', index_col=None, encoding='utf-8', engine='python')
df = df.sort_values(['subject_id', 'hadm_id'])
sorted_file = f'{MIMIC_4_SAVE_DIR}/disch_10_filtered.csv'
df.to_csv(sorted_file, index=False, encoding='utf-8')

# %%
print(len(df['hadm_id'].unique()))
print(len(dfl['hadm_id'].unique()))
set(dfl['hadm_id'].unique()).issubset(set(df['hadm_id'].unique()))

# %%
print(len(df['subject_id'].unique()))
print(len(dfl['subject_id'].unique()))
set(dfl['subject_id'].unique()).issubset(set(df['subject_id'].unique()))

# %%
"""
    Concatenate the labels with the notes data and split using the saved splits
"""
import csv
from datetime import datetime
import random


import pandas as pd

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

def concat_data(labelsfile, notes_file):
    """
        INPUTS:
            labelsfile: sorted by hadm id, contains one label per line
            notes_file: sorted by hadm id, contains one note per line
    """
    with open(labelsfile, 'r', encoding='utf-8') as lf:
        print("CONCATENATING")
        with open(notes_file, 'r', encoding='utf-8') as notesfile:
            outfilename = f'{MIMIC_4_SAVE_DIR}/notes_labeled_icd10_filtered.csv'
            with open(outfilename, 'w', encoding='utf-8') as outfile:
                w = csv.writer(outfile)
                w.writerow(['subject_id', 'hadm_id', 'text', 'labels'])

                labels_gen = next_labels(lf)
                notes_gen = next_notes(notesfile)

                for i, (subj_id, text, hadm_id) in enumerate(notes_gen):
                    if i % 10000 == 0:
                        print(str(i) + " done")
                    cur_subj, cur_labels, cur_hadm = next(labels_gen)

                    if cur_hadm == hadm_id:
                        w.writerow([subj_id, str(hadm_id), text, ';'.join(cur_labels)])
                    else:
                        print("couldn't find matching hadm_id. data is probably not sorted correctly")
                        break
                    
    return outfilename

def next_labels(labelsfile):
    """
        Generator for label sets from the label file
    """
    labels_reader = csv.reader(labelsfile)
    #header
    next(labels_reader)

    first_label_line = next(labels_reader)

    cur_subj = int(first_label_line[0])
    cur_hadm = int(first_label_line[1])
    cur_labels = [first_label_line[2]]

    for row in labels_reader:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        code = row[2]
        #keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_labels, cur_hadm
            cur_labels = [code]
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            #add to the labels and move on
            cur_labels.append(code)
    yield cur_subj, cur_labels, cur_hadm

def next_notes(notesfile):
    """
        Generator for notes from the notes file
        This will also concatenate discharge summaries and their addenda, which have the same subject and hadm id
    """
    nr = csv.reader(notesfile)
    #header
    next(nr)

    first_note = next(nr)

    cur_subj = int(first_note[0])
    cur_hadm = int(first_note[1])
    cur_text = first_note[3]
    
    for row in nr:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        text = row[3]
        #keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_text, cur_hadm
            cur_text = text
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            #concatenate to the discharge summary and move on
            cur_text += " " + text
    yield cur_subj, cur_text, cur_hadm


# %%
df.head(10)

# %%
dfl.head(10)

# %%
#For this cell, I do not recommend to run this cell directly.
#You can run through the file data_mimic_IV_concate_note_label.py
#Remember to change the directories there: MIMIC_4_SAVE_DIR, labelsfile,notes_file, output_note_labeled_file
labeled = concat_data(f'{MIMIC_4_SAVE_DIR}/ALL_CODES_filtered.csv', f'{MIMIC_4_SAVE_DIR}/disch_10_filtered.csv')

# %% [markdown]
# Let's sanity check the combined data we just made. Do we have all hadm id's accounted for, and the same vocab stats?

# %%
dfnl = pd.read_csv(f'{MIMIC_4_SAVE_DIR}/notes_labeled_icd10_filtered.csv', encoding='utf-8', engine='python')
#Tokens and types
types = set()
num_tok = 0
for row in dfnl.itertuples():
    for w in row[3].split():
        types.add(w)
        num_tok += 1

# %%
print("num types", len(types), "num tokens", num_tok)

# %%
len(dfnl['hadm_id'].unique())

# %%
len(dfnl)

# %%
len(dfnl['subject_id'].unique())

# %% [markdown]
# ## Create train/dev/test splits

# %%
def split_data(labeledfile, base_name):
    print("SPLITTING")
    #create and write headers for train, dev, test
    train_name = '%s_train_split.csv' % (base_name)
    dev_name = '%s_dev_split.csv' % (base_name)
    test_name = '%s_test_split.csv' % (base_name)
    train_file = open(train_name, 'w', encoding='utf-8')
    dev_file = open(dev_name, 'w', encoding='utf-8')
    test_file = open(test_name, 'w', encoding='utf-8')
    train_file.write(','.join(['subject_id', 'hadm_id', 'text', 'labels']) + "\n")
    dev_file.write(','.join(['subject_id', 'hadm_id', 'text', 'labels']) + "\n")
    test_file.write(','.join(['subject_id', 'hadm_id', 'text', 'labels']) + "\n")

    hadm_ids = {}

    #read in train, dev, test splits
    for splt in ['train', 'dev', 'test']:
        hadm_ids[splt] = set()
        with open('%s/%s_full_hadm_ids.csv' % (MIMIC_4_SAVE_DIR, splt), 'r') as f:
            for line in f:
                hadm_ids[splt].add(line.rstrip())

    with open(labeledfile, 'r', encoding='utf-8') as lf:
        reader = csv.reader(lf)
        next(reader)
        i = 0
        cur_hadm = 0
        for row in reader:
            #filter text, write to file according to train/dev/test split
            if i % 10000 == 0:
                print(str(i) + " read")

            hadm_id = row[1]

            if hadm_id in hadm_ids['train']:
                train_file.write(','.join(row) + "\n")
            elif hadm_id in hadm_ids['dev']:
                dev_file.write(','.join(row) + "\n")
            elif hadm_id in hadm_ids['test']:
                test_file.write(','.join(row) + "\n")
            else:
                print("Error")

            i += 1

        train_file.close()
        dev_file.close()
        test_file.close()
    return train_name, dev_name, test_name
    

# %%
print(MIMIC_4_SAVE_DIR)

# %%
# MIMIC_4_SAVE_DIR
# /home/justjosh/Projects/PLM-ICD/data/mimic4/mimic4_icd10/notes_labeled_icd10_filtered.csv
base_name = "%s/disch" % MIMIC_4_SAVE_DIR #for output
fname = f'{MIMIC_4_SAVE_DIR}/notes_labeled_icd10_filtered.csv'
tr, dv, te = split_data(fname, base_name=base_name)
# tr, dv, te = split_data(fname, base_name=base_name)

# %%
import os
import csv

def ensure_split_files(labeledfile, save_dir):
    """
    Ensure that train/dev/test split ID files exist.
    If missing, create them by splitting hadm_ids from the labeled file.
    """
    split_files = {
        'train': os.path.join(save_dir, 'train_full_hadm_ids.csv'),
        'dev': os.path.join(save_dir, 'dev_full_hadm_ids.csv'),
        'test': os.path.join(save_dir, 'test_full_hadm_ids.csv')
    }

    # Check if all exist
    if all(os.path.exists(path) for path in split_files.values()):
        return split_files

    print("Split ID files missing — generating them from labeled file...")

    # Example: naive split by index (replace with your real logic)
    hadm_ids = {'train': set(), 'dev': set(), 'test': set()}
    with open(labeledfile, 'r', encoding='utf-8') as lf:
        reader = csv.reader(lf)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            hadm_id = row[1]
            if i % 10 < 8:
                hadm_ids['train'].add(hadm_id)
            elif i % 10 == 8:
                hadm_ids['dev'].add(hadm_id)
            else:
                hadm_ids['test'].add(hadm_id)

    # Write them out
    for split, path in split_files.items():
        with open(path, 'w', encoding='utf-8') as f:
            for hid in sorted(hadm_ids[split]):
                f.write(hid + "\n")

    print("Split ID files created.")
    return split_files


def split_data(labeledfile, base_name, save_dir):
    print("SPLITTING")

    # Ensure split files exist
    split_files = ensure_split_files(labeledfile, save_dir)

    # Create and write headers for train, dev, test
    train_name = f'{base_name}_train_split.csv'
    dev_name = f'{base_name}_dev_split.csv'
    test_name = f'{base_name}_test_split.csv'

    with open(train_name, 'w', encoding='utf-8') as train_file, \
         open(dev_name, 'w', encoding='utf-8') as dev_file, \
         open(test_name, 'w', encoding='utf-8') as test_file:

        header = ['subject_id', 'hadm_id', 'text', 'labels']
        train_file.write(','.join(header) + "\n")
        dev_file.write(','.join(header) + "\n")
        test_file.write(','.join(header) + "\n")

        # Load hadm_ids
        hadm_ids = {}
        for splt, path in split_files.items():
            with open(path, 'r', encoding='utf-8') as f:
                hadm_ids[splt] = {line.strip() for line in f}

        # Process labeled file
        with open(labeledfile, 'r', encoding='utf-8') as lf:
            reader = csv.reader(lf)
            next(reader)
            for i, row in enumerate(reader):
                if i % 10000 == 0:
                    print(f"{i} read")

                hadm_id = row[1]
                if hadm_id in hadm_ids['train']:
                    train_file.write(','.join(row) + "\n")
                elif hadm_id in hadm_ids['dev']:
                    dev_file.write(','.join(row) + "\n")
                elif hadm_id in hadm_ids['test']:
                    test_file.write(','.join(row) + "\n")
                else:
                    print(f"Warning: hadm_id {hadm_id} not found in any split.")

    return train_name, dev_name, test_name


# Usage
MIMIC_4_SAVE_DIR = "/home/justjosh/Projects/PLM-ICD/data/mimic4/mimic4_icd10"
fname = os.path.join(MIMIC_4_SAVE_DIR, "notes_labeled_icd10_filtered.csv")
base_name = os.path.join(MIMIC_4_SAVE_DIR, "disch")

tr, dv, te = split_data(fname, base_name=base_name, save_dir=MIMIC_4_SAVE_DIR)
print("Created:", tr, dv, te)


# %%
train_df=pd.read_csv('%s/disch_train_split.csv' % (MIMIC_4_SAVE_DIR), encoding='utf-8',engine='python',dtype={"icd_code": str})
dev_df=pd.read_csv('%s/disch_dev_split.csv' % (MIMIC_4_SAVE_DIR), encoding='utf-8',engine='python',dtype={"icd_code": str})
test_df=pd.read_csv('%s/disch_test_split.csv' % (MIMIC_4_SAVE_DIR), encoding='utf-8',engine='python',dtype={"icd_code": str})
# train_df=pd.read_csv(tr, encoding='utf-8',engine='python')
# dev_df=pd.read_csv(dv, encoding='utf-8',engine='python')
# test_df=pd.read_csv(te, encoding='utf-8',engine='python')

# %%
print(len(dfnl['subject_id']))
print(len(train_df['subject_id']))
print(len(dev_df['subject_id']))
print(len(test_df['subject_id']))

# %%
print(len(dfnl['subject_id'].unique()))
print(len(train_df['subject_id'].unique()))
print(len(dev_df['subject_id'].unique()))
print(len(test_df['subject_id'].unique()))

# %%
print(len(dfnl['subject_id'].unique()))
print(len(train_df['subject_id'].unique()))
print(len(dev_df['subject_id'].unique()))
print(len(test_df['subject_id'].unique()))

# %% [markdown]
# ## Build vocabulary from training data

# %%
import csv
import numpy as np
import operator

from collections import defaultdict
from scipy.sparse import csr_matrix

def build_vocab(vocab_min, infile, vocab_filename):
    """
        INPUTS:
            vocab_min: how many documents a word must appear in to be kept
            infile: (training) data file to build vocabulary from
            vocab_filename: name for the file to output
    """
    with open(infile, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        #header
        next(reader)

        #0. read in data
        print("reading in data...")
        #holds number of terms in each document
        note_numwords = []
        #indices where notes start
        note_inds = [0]
        #indices of discovered words
        indices = []
        #holds a bunch of ones
        data = []
        #keep track of discovered words
        vocab = {}
        #build lookup table for terms
        num2term = {}
        #preallocate array to hold number of notes each term appears in
        note_occur = np.zeros(400000, dtype=int)
        i = 0
        for row in reader:
            text = row[2]
            numwords = 0
            for term in text.split():
                #put term in vocab if it's not there. else, get the index
                index = vocab.setdefault(term, len(vocab))
                indices.append(index)
                num2term[index] = term
                data.append(1)
                numwords += 1
            #record where the next note starts
            note_inds.append(len(indices))
            indset = set(indices[note_inds[-2]:note_inds[-1]])
            #go thru all the word indices you just added, and add to the note occurrence count for each of them
            for ind in indset:
                note_occur[ind] += 1
            note_numwords.append(numwords)
            i += 1
        #clip trailing zeros
        note_occur = note_occur[note_occur>0]

        #turn vocab into a list so indexing doesn't get fd up when we drop rows
        vocab_list = np.array([word for word,ind in sorted(vocab.items(), key=operator.itemgetter(1))])

        #1. create sparse document matrix
        C = csr_matrix((data, indices, note_inds), dtype=int).transpose()
        #also need the numwords array to be a sparse matrix
        note_numwords = csr_matrix(1. / np.array(note_numwords))
        
        #2. remove rows with less than 3 total occurrences
        print("removing rare terms")
        #inds holds indices of rows corresponding to terms that occur in < 3 documents
        inds = np.nonzero(note_occur >= vocab_min)[0]
        print(str(len(inds)) + " terms qualify out of " + str(C.shape[0]) + " total")
        #drop those rows
        C = C[inds,:]
        note_occur = note_occur[inds]
        vocab_list = vocab_list[inds]

        print("writing output")
        with open(vocab_filename, 'w', encoding='utf-8') as vocab_file:
            for word in vocab_list:
                vocab_file.write(word + "\n")

# %%
vocab_min = 3
vname = '%s/vocab.csv' % MIMIC_4_SAVE_DIR
build_vocab(vocab_min, '%s/disch_train_split.csv' % (MIMIC_4_SAVE_DIR), vname)

# %% [markdown]
# ## Sort each data split by length for batching

# %%
for splt in ['train', 'dev', 'test']:
    filename = '%s/disch_%s_split.csv' % (MIMIC_4_SAVE_DIR, splt)
    df = pd.read_csv(filename, encoding='utf-8', engine='python',dtype={"icd_code": str})
    df['length'] = df.apply(lambda row: len(str(row['text']).split()), axis=1)
    df = df.sort_values(['length'])
    df.to_csv('%s/%s_full.csv' % (MIMIC_4_SAVE_DIR, splt), index=False, encoding='utf-8')

# %% [markdown]
# ## Filter each split to the top 50 diagnosis/procedure codes

# %%
Y = 50

# %%
#first calculate the top k
counts = Counter()
dfnl = pd.read_csv('%s/notes_labeled.csv' % MIMIC_4_DIR)
for row in dfnl.itertuples():
    for label in str(row[4]).split(';'):
        counts[label] += 1
codes_50 = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
codes_50 = [code[0] for code in codes_50[:Y]]

# %%
with open(f"{MIMIC_4_SAVE_DIR}/top50_icd10_code_list.txt", "r") as fd:
    codes_50=[x.strip() for x in fd.readlines()]
print(codes_50)

with open('%s/TOP_%s_CODES.csv' % (MIMIC_4_SAVE_DIR, str(Y)), 'w') as of:
    w = csv.writer(of)
    for code in codes_50:
        w.writerow([code])

# %%
for splt in ['train', 'dev', 'test']:
    print(splt)
    hadm_ids = set()
    with open('%s/%s_50_hadm_ids.csv' % (MIMIC_4_SAVE_DIR, splt), 'r') as f:
        for line in f:
            hadm_ids.add(line.rstrip())
    with open('%s/notes_labeled_icd10_filtered.csv' % MIMIC_4_SAVE_DIR, 'r', encoding='utf-8') as f:
        with open('%s/%s_%s.csv' % (MIMIC_4_SAVE_DIR, splt, str(Y)), 'w', encoding='utf-8') as of:
            r = csv.reader(f)
            w = csv.writer(of)
            #header
            w.writerow(next(r))
            i = 0
            for row in r:
                hadm_id = row[1]
                if hadm_id not in hadm_ids:
                    continue
                codes = set(str(row[3]).split(';'))
                filtered_codes = codes.intersection(set(codes_50))
                if len(filtered_codes) > 0:
                    w.writerow(row[:3] + [';'.join(filtered_codes)])
                    i += 1

# %%
Y=50
for splt in ['train', 'dev', 'test']:
    filename = '%s/%s_%s.csv' % (MIMIC_4_SAVE_DIR, splt, str(Y))
    df = pd.read_csv(filename, encoding='utf-8', engine='python')
    df['length'] = df.apply(lambda row: len(str(row['text']).split()), axis=1)
    df = df.sort_values(['length'])
    df.to_csv('%s/%s_%s.csv' % (MIMIC_4_SAVE_DIR, splt, str(Y)), index=False, encoding='utf-8')


