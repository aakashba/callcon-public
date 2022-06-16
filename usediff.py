import sys
import pickle
import numpy as np
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from scipy.stats import ttest_rel

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def use(reflist, predlist, batchsize):
	import tensorflow_hub as tfhub

	module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
	model = tfhub.load(module_url)
	refs = list()
	preds = list()
	count = 0
	for ref, pred in zip(reflist, predlist):
	    #print(ref)        
	    ref = ' '.join(ref[0]).strip()
	    pred = ' '.join(pred).strip()
	    if pred == '':
	        pred = ' <s> '
			#count+=1
			#continue
	    refs.append(ref)
	    preds.append(pred)

	#total_csd = np.zeros(count)
	scores = list()
	for i in range(0, len(refs), batchsize):
            print(i)
            ref_emb = model(refs[i:i+batchsize])
            pred_emb = model(preds[i:i+batchsize])
            csm = cosine_similarity_score(ref_emb, pred_emb)
            csd = csm.diagonal()
            total_csd = csd #np.concatenate([total_csd, csd])
            scores = total_csd.tolist()
	#print(count)
	avg_css = np.average(total_csd)

	corpuse = (round(avg_css*100, 2))
	ret = ('for %s functions\n' % (len(predlist)))
	ret+= 'Cosine similarity score with universal sentence encoder embedding is %s\n' % corpuse
	return scores, corpuse, ret

def cosine_similarity_score(x, y):
	from sklearn.metrics.pairwise import cosine_similarity
	cosine_similarity_matrix = cosine_similarity(x, y)
	return cosine_similarity_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('inputA', type=str, default=None)
    parser.add_argument('inputB', type=str, default=None)
    parser.add_argument('--data', dest='dataprep', type=str, default='../javastmt')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--sbt', action='store_true', default=False)
    parser.add_argument('--not-diffonly', dest='diffonly', action='store_false', default=True)
    parser.add_argument('--shuffles', type=int, default=100)

    parser.add_argument('--coms-filename', dest='comsfilename', type=str, default='../javastmt/output/coms.test')
    parser.add_argument('--batchsize', dest='batchsize', type=int, default=50000)
    # parser.add_argument('--data', dest='datapath', type=str, default='/nfs/projects/simmetrics/data/standard/output')
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--vmem-limit', dest='vmemlimit', type=int, default=0)

    args = parser.parse_args()
    comsfile = args.comsfilename
    batchsize = args.batchsize
    # datapath = args.datapath
    gpu = args.gpu
    vmemlimit = args.vmemlimit

    outdir = args.outdir
    dataprep = args.dataprep
    inputA_file = args.inputA
    inputB_file = args.inputB
    sbt = args.sbt
    diffonly = args.diffonly
    R = args.shuffles

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
                   # print(predsA[fid])
                   # print(predsB[fid])
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

    scoresA, SA, ret = use(refs, newpredsA, batchsize)
    print(ret)
    print()

    #print(scoresA)

    scoresB, SB, ret = use(refs, newpredsB, batchsize)
    print(ret)
    print()

    if diffonly:
        scoresS, SAMESSCORE, ret = use(samesRefs, samesPreds, batchsize)
        print(ret)
        print()

    ttest = ttest_rel(scoresA, scoresB, alternative='greater')
    print(ttest)
