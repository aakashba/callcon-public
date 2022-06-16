import sys
import pickle
import argparse
import re
import random

random.seed(1337)

import statistics

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score

from scipy.stats import ttest_rel

import numpy as np

#from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word

def corpus_meteor(expected, predicted):

    scores = list()
    
    for e, p in zip(expected, predicted):
        e = [' '.join(x) for x in e]
        p = ' '.join(p)
        m = meteor_score(e, p)
        scores.append(m)
        
    return scores, np.mean(scores)
    

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.asarray(x)
    x = x.astype(np.float)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def meteor_so_far_m_only(refs, preds):
    scores, m = corpus_meteor(refs, preds)
    m = round(m*100, 2)
    return m

def meteor_so_far(refs, preds):
    
    scores, m = corpus_meteor(refs, preds)
    m = round(m*100, 2)
    
    ret = ''
    ret += ('for %s functions\n' % (len(preds)))
    ret += ('M %s\n' % (m))
    return scores, m, ret

def bleu_so_far_ba_only(refs, preds):
    Ba = corpus_bleu(refs, preds)
    Ba = round(Ba * 100, 2)
    return Ba

def bleu_so_far(refs, preds):
    Ba = corpus_bleu(refs, preds)
    B1 = corpus_bleu(refs, preds, weights=(1,0,0,0))
    B2 = corpus_bleu(refs, preds, weights=(0,1,0,0))
    B3 = corpus_bleu(refs, preds, weights=(0,0,1,0))
    B4 = corpus_bleu(refs, preds, weights=(0,0,0,1))

    Ba = round(Ba * 100, 2)
    B1 = round(B1 * 100, 2)
    B2 = round(B2 * 100, 2)
    B3 = round(B3 * 100, 2)
    B4 = round(B4 * 100, 2)

    ret = ''
    ret += ('for %s functions\n' % (len(preds)))
    ret += ('Ba %s\n' % (Ba))
    ret += ('B1 %s\n' % (B1))
    ret += ('B2 %s\n' % (B2))
    ret += ('B3 %s\n' % (B3))
    ret += ('B4 %s\n' % (B4))
    
    return Ba, ret

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputA', type=str, default=None)
    parser.add_argument('inputB', type=str, default=None)
    parser.add_argument('--data', dest='dataprep', type=str, default='../javastmt')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--challenge', action='store_true', default=False)
    parser.add_argument('--obfuscate', action='store_true', default=False)
    parser.add_argument('--sbt', action='store_true', default=False)
    parser.add_argument('--not-diffonly', dest='diffonly', action='store_false', default=True)
    parser.add_argument('--shuffles', type=int, default=100)

    args = parser.parse_args()
    outdir = args.outdir
    dataprep = args.dataprep
    inputA_file = args.inputA
    inputB_file = args.inputB
    challenge = args.challenge
    obfuscate = args.obfuscate
    sbt = args.sbt
    diffonly = args.diffonly
    R = args.shuffles

    if challenge:
        dataprep = '../data/challengeset/output'

    if obfuscate:
        dataprep = '../data/obfuscation/output'

    if sbt:
        dataprep = '../data/sbt/output'

    if inputA_file is None:
        print('Please provide an input file to test with --input')
        exit()

    sys.path.append(dataprep)
    import tokenizer

    #prep('preparing predictions list A... ')
    predsA = dict()
    predictsA = open(inputA_file, 'r')
    for c, line in enumerate(predictsA):
        (fid, pred) = line.split('\t')
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        predsA[fid] = pred
    predictsA.close()
    #drop()

    #prep('preparing predictions list B... ')
    predsB = dict()
    predictsB = open(inputB_file, 'r')
    for c, line in enumerate(predictsB):
        (fid, pred) = line.split('\t')
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        predsB[fid] = pred
    predictsB.close()
    #drop()

    refs = list()
    refsd = dict()
    newpredsA = list()
    newpredsB = list()
    samesPreds = list()
    samesRefs = list()
    worddiff = dict()
    bleusA = dict()
    bleusB = dict()
    fidbd = dict()
    smlnd = dict()
    d = 0
    targets = open('%s/output/coms.test' % (dataprep), 'r')
    for line in targets:
        (fid, com) = line.split('<SEP>')
        fid = int(fid)
        com = com.split()
        com = fil(com)        
        if len(com) < 1:
            continue

        try:
            if(diffonly):
                if(predsA[fid] == predsB[fid]):
                    samesPreds.append(predsA[fid])
                    samesRefs.append([com])
                    continue

            newpredsA.append(predsA[fid])
            newpredsB.append(predsB[fid])

        except Exception as ex:
            #newpreds.append([])
            continue
        
        refsd[fid] = com
        refs.append([com])

    c = 0

    scoresA, SA, ret = meteor_so_far(refs, newpredsA)
    print(ret)
    print()

    scoresB, SB, ret = meteor_so_far(refs, newpredsB)
    print(ret)
    print()

    if diffonly:
        scoresS, SAMESBLEU, ret = meteor_so_far(samesRefs, samesPreds)
        print(ret)
        print()

    ttest = ttest_rel(scoresA, scoresB, alternative='greater')
    print(ttest)

