import pickle
import sqlite3
import numpy as np

filedb = sqlite3.connect('calltdats.sqlite')
filedbcur = filedb.cursor()

callnodes = pickle.load(open("q75cnodes.pkl","rb")) 
newcalls = dict()
q75fids = pickle.load(open("../q75fids.pkl","rb"))
count = 0 

tdatstok = pickle.load(open("/nfs/projects/funcom/data/ccppstmt/q75/tdats.tok","rb"))


for fid in q75fids:
    count +=1
    if count % 10000 == 0:
        print(count)
    newcalls[fid] = []
    p = 0
    try:
        for cfid in callnodes[fid]:
            filedbcur.execute('select tdat from fundats where fid={}'.format(cfid))
            tdat = filedbcur.fetchall()[0][0]
            newcalls[fid].append(tdatstok.texts_to_sequences([tdat],maxlen=200)[0])
    except:
        newcalls[fid] = [np.zeros((200))]


filedbcur.close()
filedb.close()

pickle.dump(newcalls,open("q75cnodesdata.pkl","wb"))


