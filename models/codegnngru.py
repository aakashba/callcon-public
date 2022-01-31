from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax, Adam
import tensorflow.keras as keras
import tensorflow.keras.utils
import tensorflow.keras.backend as K
import tensorflow as tf

from custom.graphlayers import GCNLayer

# codegnngru baseline from ICPC'20 LeClair et al.
# configuration set to best performing approach in the paper's experiment

# sometimes called ast-attendgru-gnn

class CodeGNNGRUModel:
    def __init__(self, config):
        
        self.config = config

        self.config['tdatlen'] = 50

        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.astvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']

        self.config['maxastnodes'] = config['smllen']
        
        self.config['batch_config'] = [ ['tdat', 'com', 'smlnode', 'smledge'], ['comout'] ]

        self.config['asthops'] = 2

        self.embdims = 100
        self.smldims = 100
        self.recdims = 100
        self.tdddims = 100

    def create_model(self):
        
        tdat_input = Input(shape=self.tdatlen)
        com_input = Input(shape=self.comlen)
        smlnode_input = Input(shape=self.smllen)
        smledge_input = Input(shape=(self.smllen, self.smllen))
        
        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)

        #sdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        se = tdel(smlnode_input)

        tenc = GRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde)
        
        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = GRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=tstate_h)

        tattn = dot([decout, tencout], axes=[2, 2])
        tattn = Activation('softmax')(tattn)
        tcontext = dot([tattn, tencout], axes=[2, 1])

        astwork = se
        for k in range(self.config['asthops']):
            astwork = GCNLayer(self.embdims)([astwork, smledge_input])
        
        astwork = GRU(self.recdims, return_sequences=True)(astwork, initial_state=tstate_h)

        # attend decoder words to nodes in ast
        aattn = dot([decout, astwork], axes=[2, 2])
        aattn = Activation('softmax')(aattn)
        acontext = dot([aattn, astwork], axes=[2, 1])

        context = concatenate([tcontext, decout, acontext])

        out = TimeDistributed(Dense(self.tdddims, activation="relu"))(context)

        out = Flatten()(out)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, com_input, smlnode_input, smledge_input], outputs=out1)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, clipnorm=20.), metrics=['accuracy'])
        return self.config, model
