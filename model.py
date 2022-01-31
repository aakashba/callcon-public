import tensorflow.keras as keras
import tensorflow as tf

from models.attendgru import AttentionGRUModel as attendgru
from models.code2seq import Code2SeqModel as code2seq
from models.ast_attendgru_fc import AstAttentionGRUFCModel as ast_attendgru_fc
from models.codegnngru import CodeGNNGRUModel as codegnngru
from models.callcon_gru import CallconGRUModel as callcon_gru
from models.transformer_base import TransformerBase as xformer_base

def create_model(modeltype, config):
    mdl = None

    if modeltype == 'attendgru':
        mdl = attendgru(config)
    elif modeltype == 'codegnngru':
        mdl = codegnngru(config)
    elif modeltype == 'ast-attendgru-fc':
        mdl = ast_attendgru_fc(config)
    elif modeltype == 'code2seq':
        mdl = code2seq(config)
    elif modeltype == 'callcon-gru':
        mdl = callcon_gru(config)
    elif modeltype == 'transformer-base':
        mdl = xformer_base(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
