import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
import numpy as np
import tensorflow_hub as tfhub
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import cosine_similarity, categorical_crossentropy
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import parallel_sort

sys.path.append(os.path.abspath('../'))
import tokenizer
from myutils import index2word

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = tfhub.load(module_url)

def use_prep(comstok):
	index_tensor = tf.Variable(list(comstok.i2w.keys()), dtype=tf.int64)
	comwords = list(comstok.i2w.values())
	comwords_tensor = tf.Variable(comwords)
	return index_tensor, comwords_tensor

def custom_use_loss(index_tensor, comwords_tensor):
	def qs_use_loss(y_true, y_pred):
		y_true_arg = tf.argmax(y_true, axis=1)
		y_pred_arg = tf.argmax(y_pred, axis=1)
		i2wtable = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(index_tensor, comwords_tensor), default_value='<UNK>')

		y_true_comwords = i2wtable.lookup(y_true_arg)
		y_pred_comwords = i2wtable.lookup(y_pred_arg)
		y_true_emb = model(y_true_comwords)
		y_pred_emb = model(y_pred_comwords)

		y_true = y_true[:, :y_true_emb.shape[1]]
		zs = tf.zeros([y_true_emb.shape[0], y_true_emb.shape[1]], dtype=tf.float32)
		y_true *= zs
		y_true += 1
		y_true *= y_true_emb
		y_pred = y_pred[:, :y_pred_emb.shape[1]]
		zs = tf.zeros([y_pred_emb.shape[0], y_pred_emb.shape[1]], dtype=tf.float32)
		y_pred *= zs
		y_pred += 1
		y_pred *= y_pred_emb

		axis=1
		css = keras.losses.cosine_similarity(y_true, y_pred)
		css = tf.abs(css)
		css = (css+1)/2
		css = 1-css
		# print(css)
		return css
	return qs_use_loss

def attendgru_prep():
	datapath = '/nfs/projects/funcom/data/standard/aakashfiltered'
	sys.path.append(datapath)
	import tokenizer

	fmodelfname = '/nfs/projects/funcom/data/outdir/models/attendgru_E01_1612205848.h5'
	fmodel = keras.models.load_model(fmodelfname, custom_objects={"tf":tf, "keras":keras})
	return fmodel

def custom_attendgru_loss(fmodel):
	def qs_attendgru_loss(y_true, y_pred):
		com_input = fmodel.get_layer('input_2')
		dec_emb = fmodel.get_layer('embedding_1')
		dec_gru = fmodel.get_layer('gru_1')
		ref_input = com_input(y_true)
		pred_input = com_input(y_pred)
		ref_emb = dec_emb(ref_input)
		ref_gru = dec_gru(ref_emb)
		pred_emb = dec_emb(pred_input)
		pred_gru = dec_gru(pred_emb)
		y_pred_flat = tf.keras.layers.Flatten()(pred_gru)
		y_true_flat = tf.keras.layers.Flatten()(ref_gru)

		zs = tf.zeros([y_true_flat.shape[0], y_true_flat.shape[1]], dtype=tf.float32)
		d11 = y_true_flat.shape[1]-y_true.shape[-1]
		p = tf.constant([[0, 0], [0, d11]])
		y_true=tf.pad(y_true, p, "CONSTANT")
		y_true *= zs
		y_true += 1
		y_true *= y_true_flat

		zs = tf.zeros([y_pred_flat.shape[0], y_pred_flat.shape[1]], dtype=tf.float32)
		d11 = y_pred_flat.shape[1]-y_pred.shape[-1]
		p = tf.constant([[0, 0], [0, d11]])
		y_pred=tf.pad(y_pred, p, "CONSTANT")
		y_pred *= zs
		y_pred += 1
		y_pred *= y_pred_flat

		css = keras.losses.cosine_similarity(y_true, y_pred)
		return css
	return qs_attendgru_loss


def custom_cce_loss():
	def qs_cce_loss(y_true, y_pred):
		cce = keras.losses.categorical_crossentropy(y_true, y_pred)
		return cce
	return qs_cce_loss

def custom_dist_cce_loss(dist):
	def qs_dist_cce_loss(y_true, y_pred):
		y_true_arg = tf.argmax(y_true, axis=1)
		y_true *= 0
		y_true += 1
		y_true_soft = [dist[i] for i in y_true_arg]
		y_true *= y_true_soft
		cce = keras.losses.categorical_crossentropy(y_true, y_pred)
		return cce
	return qs_dist_cce_loss

def custom_dist_cce_loss_cmc_5(dist):
    def qs_dist_cce_loss_cmc5(y_true, y_pred):
        y_true_arg = tf.argmax(y_true, axis=1)
        y_true_soft = [dist[i] for i in y_true_arg]
        
        y_true_soft = np.asarray(y_true_soft)
        y_true_soft = y_true_soft.astype("float32")
        
        #for v in y_true_soft[1]:
        #    print(v, end=' ')
        #quit()
        
        cce = keras.losses.categorical_crossentropy(y_true_soft, y_pred)
        return cce
    return qs_dist_cce_loss_cmc5
