import sys
import pickle
import argparse
import re
import random

random.seed(1337)

import statistics

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

import numpy as np

#from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word

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
           # print(ex)
            continue
        
        refsd[fid] = com
        refs.append([com])

#Set c = 0
#Compute actual statistic of score differences SX −SY
#on test data
#For random shuffles r = 1,...,R
#For sentences in test set
#Shuffle variable tuples between systems X and Y
#with probability 0.5
#Compute pseudo-statistic SXr −SYr on shuffled data
#If SXr −SYr ≥SX −SY
#c = c + 1
#If c/R ≤α
#Reject the null hypothesis

    c = 0

    SA, ret = bleu_so_far(refs, newpredsA)
    print(ret)
    print()

    SB, ret = bleu_so_far(refs, newpredsB)
    print(ret)
    print()

    if diffonly:
        SAMESBLEU, ret = bleu_so_far(samesRefs, samesPreds)
        print(ret)
        print()

    #R = 100

    doublerefs = refs+refs
    allpreds = newpredsA+newpredsB
    ndat = int(len(refs)/2)

    #for r in range(0, R):
    if 0 == 1:   
        #temp = list(zip(doublerefs, allpreds))
        #random.shuffle(temp)
        #trefs, tpreds = zip(*temp)

        #trefsB = trefs[ndat:]
        #tpredsB = tpreds[ndat:]

        #trefsA = trefs[:ndat]
        #tpredsA = tpreds[:ndat]

        temp = list(zip(refs, newpredsA))
        random.shuffle(temp)
        trefsR1, tpredsR1 = zip(*temp)

        temp = list(zip(refs, newpredsB))
        random.shuffle(temp)
        trefsR2, tpredsR2 = zip(*temp)

        trefsA = trefsR1[ndat:] + trefsR2[ndat:]
        trefsB = trefsR1[:ndat] + trefsR2[:ndat]

        tpredsA = tpredsR1[ndat:] + tpredsR2[ndat:]
        tpredsB = tpredsR1[:ndat] + tpredsR2[:ndat]

        SAr = bleu_so_far_ba_only(trefsA, tpredsA)
        SBr = bleu_so_far_ba_only(trefsB, tpredsB)

        if (SAr - SBr) >= (SA - SB):
            c = c + 1

        print(r, c, SAr, SBr, SAr-SBr, SA, SB, SA-SB, end='')
        if r > 0:
            print(' %f' % round(c/r, 2), end='')
        print()


print('c {}\tc/r {}'.format(c, c/R))

