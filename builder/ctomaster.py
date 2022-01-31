import pickle

fnames = pickle.load(open("q75newfnames.pkl","rb"))
callset = pickle.load(open("q75newcalls.pkl","rb"))
projfid = pickle.load(open("q75newpidfid.pkl","rb"))
fidproj = pickle.load(open("q75newfidpid.pkl","rb"))
newset = dict()

print("dataload done")

count = 0
q90set = pickle.load(open("q75newfids.pkl","rb"))

for fid in q90set:
    pid = fidproj[fid]
    newset[fid] = list()
    for pfid in projfid[pid]:
        if len(newset[fid]) >= 5:
            break
        try:
            if fnames[pfid] in callset[fid]:
                callset[fid].remove(fnames[pfid]) # negate overloaded, clones or inherited methods from taking all nodes. first one kept.
                newset[fid].append(pfid)
        except:
            continue

    count += 1
    if count % 10000 == 0:
        print(count)

pickle.dump(newset,open("q75callsto.pkl","wb"))


