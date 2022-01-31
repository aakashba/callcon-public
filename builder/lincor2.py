import pickle

initial = pickle.load(open("initialmap.pkl","rb"))
newlines = pickle.load(open("q75newfilefidlines.pkl","rb"))
oldlines = pickle.load(open("../q75oldfidlines.pkl","rb"))

revstart = dict()

for fileid in newlines:
    revstart[fileid] = dict()
    for fid in newlines[fileid]:
        start = newlines[fileid][fid][0]
        revstart[fileid][start] = fid

final = dict()
bad = list()
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
    else:
        [fileid,lbegin,lend] = oldlines[ofid]
        try: 
            nfid = revstart[fileid][lbegin]
            final[ofid] = nfid
        except:
            try:
                m = 0
                mfid = 0
                for start in revstart[fileid]:
                    tmp = abs(lbegin-start)
                    if m == 0:
                        m = tmp
                        mfid = revstart[fileid][start]

                    if tmp < m:
                        m = tmp
                        mfid = revstart[fileid][start]
                if mfid != 0:
                    tester.append([ofid,mfid])
                    final[ofid] = mfid
            except:
                bad.append(ofid)


print(len(final))
            
        
pickle.dump(final,open("finalmap.pkl","wb"))
pickle.dump(tester,open("manualtest.pkl","wb"))
pickle.dump(bad,open("oldfidsnofile.pkl","wb"))
