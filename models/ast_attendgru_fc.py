from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax
import tensorflow.keras as keras
import tensorflow.keras.utils
import tensorflow as tf
from tensorflow.keras import metrics

# ast-attendgru ICSE'19 LeClair et al. model with file context added
# used to show enhancement over attendgru baseline in MSR'20 Haque et al.

class AstAttentionGRUFCModel:
    def __init__(self, config):
        
        # data length in dataset is 20+ functions per file, but we can elect to reduce
        # that length here, since myutils reads this length when creating the batches
        config['sdatlen'] = 10
        config['stdatlen'] = 50
        
        config['tdatlen'] = 50

        config['smllen'] = 100
        config['3dsmls'] = False
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.sdatlen = config['sdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']

        self.config['batch_config'] = [ ['tdat', 'sdat', 'com', 'smlseq'], ['comout'] ]

        self.embdims = 100
        self.smldims = 100
        self.recdims = 100
        self.tdddims = 100

    def create_model(self):
        
        tdat_input = Input(shape=(self.tdatlen,))
        sdat_input = Input(shape=(self.sdatlen, self.config['stdatlen']))
        sml_input = Input(shape=(self.smllen,))
        com_input = Input(shape=(self.comlen,))
        
        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)

        tenc = GRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde)
        
        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = GRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=tstate_h)

        se = Embedding(output_dim=self.smldims, input_dim=self.smlvocabsize, mask_zero=False)(sml_input)
        se_enc = GRU(self.recdims, return_state=True, return_sequences=True)
        seout, state_sml = se_enc(se)

        ast_attn = dot([decout, seout], axes=[2, 2])
        ast_attn = Activation('softmax')(ast_attn)
        
        acontext = dot([ast_attn, seout], axes=[2, 1])

        tattn = dot([decout, tencout], axes=[2, 2])
        tattn = Activation('softmax')(tattn)

        tcontext = dot([tattn, tencout], axes=[2, 1])

        # Adding file context information to ast-attendgru model
        # shared embedding between tdats and sdats
        semb = TimeDistributed(tdel)
        sde = semb(sdat_input)
        
        # sdats encoder
        senc = TimeDistributed(GRU(int(self.recdims)))
        senc = senc(sde)

        # attention to sdats
        sattn = dot([decout, senc], axes=[2, 2])
        sattn = Activation('softmax')(sattn)

        scontext = dot([sattn, senc], axes=[2, 1])

        # context vector has teh result of attention to sdats along with ast, tdats and decoder output vectors
        context = concatenate([scontext, tcontext, acontext, decout])

        out = TimeDistributed(Dense(self.tdddims, activation="relu"))(context)

        out = Flatten()(out)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, sdat_input, com_input, sml_input], outputs=out1)
        
        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
        return self.config, model
