import os
import sys
import traceback
import pickle
import h5py
import argparse
import collections
from tensorflow.keras import metrics
import random
import tensorflow as tf
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
import multiprocessing
from itertools import product

from multiprocessing import Pool

from timeit import default_timer as timer

from model import create_model
from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from custom.graphlayers import GCNLayer
from custom.qstransformer_layers import TransformerBlock, TokenAndPositionEmbedding, MultiHeadAttentionBlock
from custom.qs_loss import use_prep, attendgru_prep, custom_use_loss,  custom_attendgru_loss, custom_cce_loss, custom_dist_cce_loss

def gendescr(model,model2, batch, batch2, badfids, comseqpos, comseqpos2, comstok, batchsize, config):
    
    comlen = config['comlen']
    
    fiddats = list(zip(*batch.values()))
    fiddats2 = list(zip(*batch2.values()))
    #tdats = np.array(tdats)
    #coms = np.array(coms)
    #fiddats = [ tdats, coms ]
    nfiddats = list()
    nfiddats2 = list()
    
    for fd in fiddats:
        fd = np.array(fd)
        nfiddats.append(fd)

    for d in fiddats2:
        d = np.array(d)
        nfiddats2.append(d)
    
    #print(comlen)

    for i in range(1, comlen):
        #fiddats[comseqpos] = coms
        results = model.predict(nfiddats, batch_size=batchsize)
        results2 = model2.predict(nfiddats2, batch_size=batchsize)
        for c, (t,a) in enumerate(zip(results,results2)):
            m = np.argmax(np.mean([t,a], axis=0))
            nfiddats[comseqpos][c][i] = m
            nfiddats2[comseqpos2][c][i] = m
            #print(c, i, np.argmax(s))

    final_data = {}
    for fid, com in zip(batch.keys(), nfiddats[comseqpos]):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def load_model_from_weights(modelpath, modeltype, datvocabsize, comvocabsize, smlvocabsize, datlen, comlen, smllen):
    config = dict()
    config['datvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['datlen'] = datlen # length of the data
    config['comlen'] = comlen # comlen sent us in workunits
    config['smlvocabsize'] = smlvocabsize
    config['smllen'] = smllen

    model = create_model(modeltype, config)
    model.load_weights(modelpath)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('modelfile', type=str, default=None)
    parser.add_argument('modelfile2', type=str, default=None)
    parser.add_argument('--num-procs', dest='numprocs', type=int, default='4')
    parser.add_argument('--gpu', dest='gpu', type=str, default='0')
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/funcom/data/javastmt/q90')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=200)
    parser.add_argument('--with-graph', dest='withgraph', action='store_true', default=False)
    parser.add_argument('--with-calls', dest='withcalls', action='store_true', default=False)
    parser.add_argument('--model-type', dest='modeltype', type=str, default=None)
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--datfile', dest='datfile', type=str, default='dataset.pkl')
    parser.add_argument('--testval', dest='testval', type=str, default='test')
    parser.add_argument('--vmem-limit', dest='vmemlimit', type=int, default=0)

    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    modelfile = args.modelfile
    modelfile2 = args.modelfile2
    numprocs = args.numprocs
    gpu = args.gpu
    batchsize = args.batchsize
    modeltype = args.modeltype
    outfile = args.outfile
    datfile = args.datfile
    testval = args.testval
    withgraph = args.withgraph
    withcalls = args.withcalls
    vmemlimit = args.vmemlimit
    
    if outfile is None:
        outfile = modelfile.split('/')[-1].split(".")[0]+"-"+modelfile2.split('/')[-1].split(".")[0]

    #K.set_floatx(args.dtype)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    if(vmemlimit > 0):
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vmemlimit)])
            except RuntimeError as e:
                print(e)

    prep('loading sequences... ')
    extradata = pickle.load(open('%s/dataset_short.pkl' % (dataprep), 'rb'))
    h5data = h5py.File('%s/dataset_seqs.h5' % (dataprep), 'r')
    
    seqdata = dict()
    seqdata['dt%s' % testval] = h5data.get('/dt%s' % testval)
    seqdata['ds%s' % testval] = h5data.get('/ds%s' % testval)
    seqdata['s%s' % testval] = h5data.get('/s%s' % testval)
    seqdata['c%s' % testval] = h5data.get('/c%s' % testval)
    
    drop()

    if withgraph:
        prep('loading graph data... ')
        graphdata = pickle.load(open('%s/dataset_graph.pkl' % (dataprep), 'rb'))
        for k, v in extradata.items():
            graphdata[k] = v
        extradata = graphdata
        drop()


    if withcalls:
        prep('loading call data... ')
        callnodes = pickle.load(open('%s/callsnodes.pkl' % (dataprep), 'rb'))
        calledges = pickle.load(open('%s/callsedges.pkl' % (dataprep), 'rb'))
        callnodesdata = pickle.load(open('%s/callsnodedata.pkl' % (dataprep), 'rb'))
        extradata['callnodes'] = callnodes
        extradata['calledges'] = calledges
        extradata['callnodedata'] = callnodesdata
        drop()


    prep('loading tokenizers... ')
    comstok = extradata['comstok']
    tdatstok = extradata['tdatstok']
    sdatstok = tdatstok
    smlstok = extradata['smlstok']
    if withgraph:
        graphtok = extradata['graphtok']
    drop()

    prep('loading config... ')
    print(modelfile)
    modeltype = modelfile.split('/')[-1]
    modeltype2 = modelfile2.split('/')[-1]
    (modeltype, mid, timestart) = modeltype.split('_')
    (modeltype2, mid, timestart2) = modeltype2.split("_")
    (timestart, ext) = timestart.split('.')
    (timestart2,ext) = timestart2.split('.')
   # modeltype = modeltype.split('/')[-1]
   # modeltype2 = modeltype2.split('/')[-1]
    config = pickle.load(open(outdir+'/histories/'+modeltype+'_conf_'+timestart+'.pkl', 'rb'))
    config2 = pickle.load(open(outdir+'/histories/'+modeltype2+'_conf_'+timestart2+'.pkl', 'rb'))

    comlen = config['comlen']
    #fid2loc = config['fidloc']['c'+testval] # fid2loc[fid] = loc
    loc2fid = config['locfid']['c'+testval] # loc2fid[loc] = fid
    #allfids = list(fid2loc.keys())
    allfidlocs = list(loc2fid.keys())

    drop()

    prep('loading model... ')
    config, model = create_model(modeltype, config)
    model.load_weights(modelfile)
    print(model.summary())
    config2, model2 = create_model(modeltype2, config2)
    model2.load_weights(modelfile2)
    print(model2.summary())
    drop()

    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    outdir = '.' #remove if access to outdir/predictions 
    outfn = outdir+"/ensemblepreds/predict-{}.txt".format(outfile.split('.')[0])
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)
    batch_sets = [allfidlocs[i:i+batchsize] for i in range(0, len(allfidlocs), batchsize)]
 
    prep("computing predictions...\n")
    for c, fidloc_set in enumerate(batch_sets):
        st = timer()
        
        #pcomseqs = list()
        #for fidloc in fidloc_set:
        #    pcomseqs.append(comstart)
        #    seqdata['c'+testval][fidloc] = comstart
            
        bg = batch_gen(h5data, extradata, testval, config, training=False)
        bg2 = batch_gen(h5data,extradata, testval, config2, training=False) 
        (batch, badfids, comseqpos) = bg.make_batch(fidloc_set)
        (batch2, badfids2, comseqpos2) = bg2.make_batch(fidloc_set)
        
        batch_results = gendescr(model, model2, batch, batch2, badfids, comseqpos, comseqpos2, comstok, batchsize, config)

        for key, val in batch_results.items():
            #print("{}\t{}\n".format(key, val))
            #quit()
            outf.write("{}\t{}\n".format(key, val))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, batchsize/(end-st)))

    outf.close()        
    drop()
