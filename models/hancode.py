import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import MultiHeadAttention,GRUCell,Input, Dense, Embedding, Reshape, GRU, Flatten, TimeDistributed, Activation, dot, concatenate,Bidirectional
from tensorflow.keras.models import Model
import numpy as np

#from custom.qstransformer_layers import MultiHeadAttentionBlock as HANSelfAttention

# This dynamic memory networks can use either the input module in "Ask me anything" or the input module in "End-to-End"
# The rest of the architecture follows the architecture in "Ask me anything"

class HANcode:
    def __init__(self, config):

        config['tdatlen'] = 50

        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.datlen = config['tdatlen']
        self.comlen = config['comlen']
        self.batch_size = config['batch_size']

        self.embdims = 100
        self.recdims = 100
        self.innerdim = 25

        self.sentence_cnt = config['max_sentence_cnt']
        self.sentence_size = config['max_sentence_len']
        self.config['batch_config'] = [['tdat_sent', 'com'], ['comout']]

    def create_model(self):
        print("sentence cnt: {}, sentence len: {}".format(self.sentence_cnt, self.sentence_size))
        dat_input = Input(shape=(self.sentence_cnt, self.sentence_size,),)
        print("comlen: {}".format(self.comlen))
        com_input = Input(shape=(self.comlen,))

    
        dat_input_reshaped = Reshape((self.sentence_cnt * self.sentence_size,), )(dat_input)
        ee = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)(dat_input_reshaped)
        statements = Reshape((self.sentence_cnt, self.sentence_size, self.embdims), )(ee)
        print(statements.shape)
        wGRUforward= GRU(self.innerdim,return_sequences=True)
        wGRUbackward= GRU(self.innerdim,return_sequences=True,go_backwards=True)

        wbigru = Bidirectional(wGRUforward,backward_layer=wGRUbackward)
        wencode = TimeDistributed(wbigru)
        weout = wencode(statements)
       
        wordattend = MultiHeadAttention(num_heads=3, key_dim=2, value_dim=2, attention_axes=(2,3))
        wattend = wordattend(weout,weout,weout)

        #wattend = tf.reduce_sum(wattend,axis=2) #reduce sum is done at the end of attention on the original HAN paper
        wattend_reshaped = Reshape((wattend.shape[1],wattend.shape[2]*wattend.shape[3]),)(wattend)
        wattend = TimeDistributed(Dense(int(self.innerdim*2), activation="tanh"))(wattend_reshaped) #fix dimension for concatenation

        sGRUforward= GRU(self.innerdim,return_sequences=True)
        sGRUbackward= GRU(self.innerdim,return_sequences=True,go_backwards=True)
        sencoder = Bidirectional(sGRUforward,backward_layer=sGRUbackward,)

        encout = sencoder(wattend)
        print(encout.shape)
        encout = concatenate([encout,wattend])

        #traditional tdats encoder
        #enct = GRU(self.recdims, return_state=True, return_sequences=True)
        #encoutt, state_ht = enct(ee)

        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = GRU(self.recdims, return_sequences=True)
        decout = dec(de)

        attn = dot([decout,encout], axes=[2, 2])
        attn = Activation('softmax')(attn)
        context = dot([attn, encout], axes=[2, 1])

        #attnt = dot([decout, encoutt], axes=[2, 2])
        #attnt = Activation('softmax')(attnt)
        #contextt = dot([attnt, encoutt], axes=[2, 1])

        #context = concatenate([context, contextt, decout])
        
        context = concatenate([context,decout])

        #squash into original dims
        out = TimeDistributed(Dense(self.recdims, activation="tanh"))(context)
        
        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)

        model = Model(inputs=[dat_input, com_input], outputs=out)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.config, model


if __name__ == "__main__":
    config = dict()

    config['tdatvocabsize'] = 260
    config['comvocabsize'] = 260

    try:
        config['comlen'] = 200
        config['tdatlen'] = 200
    except KeyError:
        pass

    config['batch_size'] = 200
    config['memorynetwork_input'] = 'positional-encoding'
    # config['memorynetwork_input'] = 'eos-embedding'
    config['max_sentence_len'] = 30
    config['max_sentence_cnt'] = 70

    memorynetwork_input = config['memorynetwork_input']
    if memorynetwork_input == 'eos-embedding':
        config['batch_config'] = [['tdat', 'tdat_sent', 'com'], ['comout']]
    else:
        config['batch_config'] = [['tdat_sent', 'com'], ['comout']]


    # dmn = DynamicMemoryNetworkInputModuleOnly(config)

    # dmn = DynamicMemoryNetworkInputModuleOnly(config)
    dmn = DynamicMemoryNetworks(config)
    config, model = dmn.create_model()
