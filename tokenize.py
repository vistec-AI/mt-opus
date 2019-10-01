#python tokenize.py data/opensubtitles/OpenSubtitles.en-th.en data/opensubtitles/OpenSubtitles.en-th.th data/opensubtitles_tok/

from pythainlp.tokenize import word_tokenize
# from pythainlp.ulmfit import *
import random
import sys

#open file
with open(sys.argv[0],'r') as f:
    en = f.readlines()
print('English raw:', len(en), en[:3])

with open(sys.argv[1],'r') as f:
    th = f.readlines()
print('Thai raw:', len(th), th[:3])


#tokenize
en_tok = []
for e in tqdm_notebook(en):
    en_tok.append(' '.join(word_tokenize(e,keep_whitespace=False)))

th_tok = []
for t in tqdm_notebook(th):
    th_tok.append(' '.join(word_tokenize(t)))
#     th_tok.append(' '.join(process_thai(t)))

#train-valid-test split 80/10/10
n = len(th_tok)
idx = list(range(n))
random.shuffle(idx)
train_idx, valid_idx, test_idx = idx[:int(n*0.8)], idx[int(n*0.8):int(n*0.9)], idx[int(n*0.9):]
print('train/valid/test:', len(train_idx),len(valid_idx),len(test_idx))

#save tokenized
th_train = [th_tok[i] for i in train_idx]
print('English tokenized train', len(th_train), th_train[:10])
en_train = [en_tok[i] for i in train_idx]
print('Thai tokenized train', len(en_train), en_train[:10])

with open(f'{sys.argv[2]}/train.en','w') as f:
    for e in en_train:
        f.write(e)
with open(f'{sys.argv[2]}/train.th','w') as f:
    for t in th_train:
        f.write(t)
with open(f'{sys.argv[2]}/valid.en','w') as f:
    for e in en_valid:
        f.write(e)
with open(f'{sys.argv[2]}/valid.th','w') as f:
    for t in th_valid:
        f.write(t)
with open(f'{sys.argv[2]}/test.en','w') as f:
    for e in en_test:
        f.write(e)
with open(f'{sys.argv[2]}/test.th','w') as f:
    for t in th_test:
        f.write(t)
print(f'saved to {sys.argv[2]}')