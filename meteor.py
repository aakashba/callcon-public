import sys
import pickle
import argparse
import re
import os

import numpy as np

from nltk.translate.meteor_score import meteor_score

from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

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
    #return scores, m, ret
    return ret

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
    parser.add_argument('input', type=str, default=None)
    parser.add_argument('--data', dest='dataprep', type=str, default='../javastmt/output')  
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--challenge', action='store_true', default=False)
    parser.add_argument('--obfuscate', action='store_true', default=False)
    parser.add_argument('--sbt', action='store_true', default=False)
    parser.add_argument('--lim-overlap', dest='limoverlap', type=int, default=-1)
    parser.add_argument('--tdats-filename', dest='tdatsfilename', type=str, default='tdats.test')
    parser.add_argument('--coms-filename', dest='comsfilename', type=str, default='coms.test')
    parser.add_argument('--sentence-bleus', dest='sentencebleus', action='store_true', default=False)
    
    args = parser.parse_args()
    outdir = args.outdir
    dataprep = args.dataprep
    input_file = args.input
    lim_overlap = args.limoverlap
    tdatsfilename = args.tdatsfilename
    comsfilename = args.comsfilename
    sentencebleus = args.sentencebleus

    if input_file is None:
        print('Please provide an input file to test')
        exit()

    if lim_overlap != -1:
        prep('preparing tdats list... ')
        tdats = dict()
        tdatsf = open('%s/%s' % (dataprep, tdatsfilename), 'r')
        for c, line in enumerate(tdatsf):
            (fid, tdat) = line.split('<SEP>')
            fid = int(fid)
            tdat = tdat.split()
            tdat = fil(tdat)
            tdats[fid] = tdat
        tdatsf.close()
        drop()

    prep('preparing predictions list... ')
    preds = dict()
    predicts = open(input_file, 'r')
    for c, line in enumerate(predicts):
        (fid, pred) = line.split('\t')
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        preds[fid] = pred
    predicts.close()
    drop()

    re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

    if(sentencebleus):
        bfn = os.path.basename(input_file)
        bfn = os.path.splitext(bfn)[0]
        bleusf = open('{}/bleus/{}.tsv'.format(outdir, bfn), 'w')

    refs = list()
    newpreds = list()
    d = 0
    targets = open('%s/%s' % (dataprep, comsfilename), 'r')
    for line in targets:
        (fid, com) = line.split('<SEP>')
        fid = int(fid)
        com = com.split()
        com = fil(com)
        
        if len(com) < 1:
            continue

        if lim_overlap != -1:
            t = list(set(com) & set(tdats[fid][:12]))
            overlap = len(t) #/ len(set(com))
            
            if overlap != lim_overlap:
                continue

        try:
            newpreds.append(preds[fid])
            
            if(sentencebleus):
                
                Bas = corpus_bleu([[com]], [preds[fid]])
                B1s = corpus_bleu([[com]], [preds[fid]], weights=(1,0,0,0))
                B2s = corpus_bleu([[com]], [preds[fid]], weights=(0,1,0,0))
                B3s = corpus_bleu([[com]], [preds[fid]], weights=(0,0,1,0))
                B4s = corpus_bleu([[com]], [preds[fid]], weights=(0,0,0,1))

                Bas = round(Bas * 100, 4)
                B1s = round(B1s * 100, 4)
                B2s = round(B2s * 100, 4)
                B3s = round(B3s * 100, 4)
                B4s = round(B4s * 100, 4)
                
                bleusf.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(fid, Bas, B1s, B2s, B3s, B4s))
            
        except Exception as ex:
            #newpreds.append([])
            continue

        refs.append([com])
    
    if(sentencebleus):
        bleusf.close()

    print('final status')
    print(meteor_so_far(refs, newpreds))

