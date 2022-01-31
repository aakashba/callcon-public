import pickle

initial = pickle.load(open("initialmap.pkl","rb"))
newlines = pickle.load(open("q90newfilefidlines.pkl","rb"))
oldlines = pickle.load(open("../q90oldfidlines.pkl","rb"))

revend = dict()

for fileid in newlines:
    revend[fileid] = dict()
    for fid in newlines[fileid]:
        end = newlines[fileid][fid][1]
        revend[fileid][end] = fid

final = dict()

tester = list()
count = 0
print("finished dataload")
print(len(initial))

for ofid in initial:
    count +=1
    if count % 10000 == 0:
        print(count)
    candidates = initial[ofid]
    if len(candidates) == 1:
        final[ofid] = candidates[0]
    elif len(candidates) == 0:
        [fileid,lbegin,lend] = oldlines[ofid]
        try:
            nfid = revend[fileid][lend]
            final[ofid] = nfid
        except:
            try:
                nfid = revend[fileid][min(revend[fileid].keys(), key = lambda key: abs(key-lend))] 
                tester.append([ofid,nfid])
                final[ofid] = nfid
            except:
                continue)


    else:
        [fileid,lbegin,lend] = oldlines[ofid]
        sub = {key:value for key,value in revend[fileid].items() if value in candidates}
        try:
            nfid = sub[lend]
            final[ofid]=nfid
        except:
            try:
                nfid = sub[min(sub.keys(),key = lambda key: abs(key-lend))]
                tester.append([ofid,nfid])
                final[ofid] = nfid
            except:
                continue

        
pickle.dump(final,open("finalmap2.pkl","wb"))
pickle.dump(tester,open("manualtest.pkl","wb"))        
