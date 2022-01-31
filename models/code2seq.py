from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax
import tensorflow.keras as keras
import tensorflow.keras.utils
import tensorflow as tf
from tensorflow.keras import metrics

# code2seq baseline from Alon et al.

class Code2SeqModel:
    def __init__(self, config):
        
        # data length in dataset is 20+ functions per file, but we can elect to reduce
        # that length here, since myutils reads this length when creating the batches
        config['sdatlen'] = 10
        config['stdatlen'] = 50

        config['tdatlen'] = 50

        config['smllen'] = 100
        config['3dsmls'] = False

        config['pathlen'] = 8
        config['maxpaths'] = 100
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.sdatlen = config['sdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']
        
        self.config['maxastnodes'] = config['maxpaths']

        self.config['batch_config'] = [ ['tdat', 'com', 'smlpath'], ['comout'] ]

        self.embdims = 100
        self.recdims = 100
        self.tdddims = 100

    def create_model(self):
        
        tdat_input = Input(shape=(self.tdatlen,))
        astp_input = Input(shape=(self.config['maxpaths'], self.config['pathlen']))
        com_input = Input(shape=(self.comlen,))
        
        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)

        tenc = GRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde)
        
        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = GRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=tstate_h)

        tattn = dot([decout, tencout], axes=[2, 2])
        tattn = Activation('softmax')(tattn)

        tcontext = dot([tattn, tencout], axes=[2, 1])

        aemb = TimeDistributed(tdel)
        ade = aemb(astp_input)
        
        aenc = TimeDistributed(GRU(int(self.recdims)))
        aenc = aenc(ade)

        aattn = dot([decout, aenc], axes=[2, 2])
        aattn = Activation('softmax')(aattn)

        acontext = dot([aattn, aenc], axes=[2, 1])

        context = concatenate([tcontext, acontext, decout])

        out = TimeDistributed(Dense(self.tdddims, activation="relu"))(context)

        out = Flatten()(out)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, com_input, astp_input], outputs=out1)

        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
        return self.config, model
