import pickle
import gc

fnames = pickle.load(open("q75newfnames.pkl","rb"))
callset = pickle.load(open("q75newcalls.pkl","rb"))
projfid = pickle.load(open("q75newpidfid.pkl","rb"))
fidproj = pickle.load(open("q75newfidpid.pkl","rb"))
newset = dict()
q90set = pickle.load(open("bylist.pkl","rb"))
print("dataload done")

count = 0

for fid in q90set:
    pid = fidproj[fid]
    newset[fid] = list()
    for pfid in projfid[pid]:
        if len(newset[fid]) >=5:
            break
        try:
            if fnames[fid] in callset[pfid]:
                callset[pfid].remove(fnames[fid]) # removes 1 to many mapping for overloaded and inherited functions, first sample linked
                newset[fid].append(pfid)
        except:
            continue
    count += 1
    if count % 10000==0:
        print(count)
        
del callset
gc.collect()

pickle.dump(newset,open("q75calledby2.pkl","wb"))


