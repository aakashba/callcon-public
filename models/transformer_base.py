import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, Concatenate, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, GlobalAveragePooling1D
from tensorflow.compat.v1.keras.layers import Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax
from tensorflow.keras import utils, metrics

from custom.qstransformer_layers import TransformerBlock, TokenAndPositionEmbedding, MultiHeadAttentionBlock
from custom.qs_loss import use_prep, attendgru_prep, custom_use_loss,  custom_attendgru_loss, custom_cce_loss, custom_dist_cce_loss

class TransformerBase:
    def __init__(self, config):
        
        config['tdatlen'] = 50
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.datlen = config['tdatlen']
        self.comlen = config['comlen']
        
        self.embdims = 100
        self.attheads = 2 # number of attention heads
        self.recdims = 100 
        self.ffdims = 100 # hidden layer size in feed forward network inside transformer

        self.config['batch_config'] = [ ['tdat', 'com'], ['comout'] ]
        self.config['loss_type'] = config['loss_type']
        if self.config['loss_type'] == 'use':
            self.index_tensor, self.comwords_tensor = use_prep(self.config['comstok'])
        elif self.config['loss_type'] == 'attendgru':
            self.fmodel = attendgru_prep()
        elif self.config['loss_type'] == 'use-dist':
            self.dist = config['target_dist']

    def create_model(self):
        
        dat_input = Input(shape=(self.datlen,))
        com_input = Input(shape=(self.comlen,))
        
        ee = TokenAndPositionEmbedding(self.datlen, self.tdatvocabsize, self.embdims)
        eeout = ee(dat_input)
        etransformer_block = TransformerBlock(self.embdims, self.attheads, self.ffdims)
        encout = etransformer_block(eeout, eeout)

        de = TokenAndPositionEmbedding(self.comlen, self.comvocabsize, self.embdims)
        deout = de(com_input)
        de_mha1 = MultiHeadAttentionBlock(self.embdims, self.attheads)
        de_mha1_out = de_mha1(deout, deout)
        dtransformer_block = TransformerBlock(self.embdims, self.attheads, self.ffdims)
        decout = dtransformer_block(de_mha1_out, encout)

        context = decout
        out = context
        # out = TimeDistributed(Dense(self.recdims, activation="tanh"))(context)
        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, com_input], outputs=out)
        lossf = custom_cce_loss()
        if self.config['loss_type'] == 'use':
            lossf = custom_use_loss(self.index_tensor, self.comwords_tensor)
        elif self.config['loss_type'] == 'attendgru':
            lossf = custom_attendgru_loss(self.fmodel)
        elif self.config['loss_type'] == 'use-dist':
            lossf = custom_dist_cce_loss(self.dist)

        model.compile(loss=lossf, optimizer='adam', metrics=['accuracy'])
        return self.config, model
