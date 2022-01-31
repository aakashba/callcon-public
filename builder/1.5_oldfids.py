import pickle

mapping = pickle.load(open("finalmap.pkl","rb"))
cdict = pickle.load(open("cnodesnew.pkl","rb"))
ndict = dict()
for ofid in mapping:
    try:
        nfid = mapping[ofid]
        ndict[ofid] = cdict[nfid]
    except:
        continue

pickle.dump(ndict,open("q75cnodes.pkl","wb"))
    




