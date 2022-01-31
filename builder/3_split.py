import pickle
import sys

splitfile = "trainvaltest_ids.pkl"
mainfile = pickle.load(open("callsedges.pkl","rb"))


spliter = pickle.load(open(splitfile,"rb"))

trainfid = spliter['trainfid']

valfid = spliter['valfid']

testfid = spliter['testfid']

train = dict((fid, mainfile[fid]) for fid in trainfid if fid in mainfile.keys())
pickle.dump(train,open("cedgetrain.pkl","wb"))

val = dict((fid, mainfile[fid]) for fid in valfid if fid in mainfile.keys()) 
pickle.dump(val,open("cedgeval.pkl","wb"))

test = dict((fid, mainfile[fid]) for fid in testfid if fid in mainfile.keys()) 
pickle.dump(test,open("cedgetest.pkl","wb"))

