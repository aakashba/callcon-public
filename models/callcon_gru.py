import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax
import tensorflow.keras as keras
import tensorflow.keras.utils
import tensorflow as tf
from tensorflow.keras import metrics

from custom.graphlayers import GCNLayer


class CallconGRUModel:
    def __init__(self, config):
        
        
        self.fmodelconfig = pickle.load(open('/nfs/projects/callcon/pretrain/histories/attendgru_conf_1631889114.pkl', 'rb'))
        self.fmodelfname = '/nfs/projects/callcon/pretrain/models/attendgru_E12_1631889114.h5'
        self.fmodel = keras.models.load_model(self.fmodelfname, custom_objects={"tf":tf, "keras":keras})
        
        config['tdatlen'] = 50
        config['maxcallnodes'] = 61
        config['gnnhops'] = config['hops']
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.tdatlen = config['tdatlen']
        self.comlen = config['comlen']
        self.cconlen = config['maxcallnodes']

        self.embdims = 100
        self.recdims = 100
        self.tdddims = 100

        self.config['batch_config'] = [ ['tdat', 'com', 'callnode', 'calledge'], ['comout'] ]

    def create_model(self):
       
        tdat_input = Input(shape=(self.tdatlen,))
        com_input = Input(shape=(self.comlen,))
        callnode_input = Input(shape=(self.cconlen,self.tdatlen,))  # each node is tdats of the fid
        calledge_input = Input(shape=(self.cconlen,self.cconlen))
        
        # load attendgru base model
        tdel = self.fmodel.get_layer('embedding')
        tdel._name = 'att_embedding'
        tdel.trainable = True
        tde = tdel(tdat_input)

        tenc = self.fmodel.get_layer('gru')
        tenc._name = 'att_gru'
        tenc.trainable = True
        tencout, tstate_h = tenc(tde)
        
        de = self.fmodel.get_layer('embedding_1')(com_input)
        de._name = 'att_embedding_1'
        de.trainable = True
        dec = self.fmodel.get_layer('gru_1')
        dec._name = 'att_gru_1'
        dec.trainable = True
        decout = dec(de, initial_state=tstate_h)

        tattn = dot([decout, tencout], axes=[2, 2])
        tattn = Activation('softmax')(tattn)
        tcontext = dot([tattn, tencout], axes=[2, 1])
        
        attcontext = concatenate([tcontext, decout])
        attouttd = self.fmodel.get_layer('time_distributed')
        attouttd._name = 'att_time_distributed'
        attouttd.trainable = True
        attouttd = attouttd(attcontext)
        attout = Flatten()(attouttd)

        cemb = TimeDistributed(tdel)
        cde = cemb(callnode_input)
        igru = GRU(int(self.recdims))
        igru._weights = tenc.get_weights()[0]
        igru.trainable = True
        cenc = TimeDistributed(igru)
        callwork = cenc(cde)

        # graph layer for call-chain context
        for i in range(self.config['gnnhops']): # 2 hops for call chain
            callwork = GCNLayer(self.embdims)([callwork, calledge_input])
        # callwork = GRU(self.recdims, return_sequences=True)(callwork)
        # attend decoder words to nodes in ast

        cattn = dot([decout, callwork], axes=[2, 2])
        cattn = Activation('softmax')(cattn)
        ccontext = dot([cattn, callwork], axes=[2, 1])

        #context = concatenate([ccontext, decout])

        #out = ccontext
        out = TimeDistributed(Dense(self.tdddims, activation="relu"))(ccontext)

        out = Flatten()(out)
        
        out = concatenate([out, attout])
        #out = Dense(2600)(out)
        
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, com_input, callnode_input, calledge_input], outputs=out1)

        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
        return self.config, model
 
