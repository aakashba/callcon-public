import pickle
import networkx as nx
import gc

callsto = pickle.load(open("q75callstofull.pkl","rb"))
calledby = pickle.load(open("q75calledbyfull.pkl","rb"))
fids = pickle.load(open("q75newfids.pkl","rb"))

print("data loading finished")

breadth= 5
depth= 2

class builder:
    def __init__(self):
        self.nid = 0
        self.graph = nx.DiGraph()



    def adderto(self,fid,d,b):
        if d >= 1:
            p = self.nid #parent node
            self.graph.add_node(p,text=fid)
            for pfid in callsto[fid][:b]:
                self.nid += 1
                self.graph.add_node(self.nid,text=pfid)
                self.graph.add_edge(p,self.nid)
                graph = self.adderto(pfid,d-1,b)

    def adderby(self,fid,d,b,start):
        if d>= 1:
            if start == 0:
                child = 0    # starting fid is always 0th node
            else:
                child = self.nid #child node
            self.graph.add_node(child,text=fid)
            for pfid in calledby[fid][:b]:
                self.nid += 1
                self.graph.add_node(self.nid, text=pfid)
                self.graph.add_edge(self.nid,child)
                graph = self.adderby(pfid,d-1,b,1)


    def build(self,fid,d,b):
        self.adderto(fid,d,b)
        self.adderby(fid,d,b,0)
        return(self.graph)

c = 0
finalnodes = dict()
finaledges = dict()
for fid in fids:
    c += 1
    if c % 10000 == 0:
        print(c)
        gc.collect()

    try:
        b = builder()
        graph = b.build(fid,depth,breadth)
        nodes = [x[1]['text'] for x in list(graph.nodes.data())]
        edges = nx.adjacency_matrix(graph)
    except:
        continue # no need for blank fids with no calls!
    
    finalnodes[fid] = nodes
    finaledges[fid] = edges



pickle.dump(finalnodes,open("cnodesnew.pkl","wb"))
pickle.dump(finaledges,open("cedgesnew.pkl","wb"))


    


