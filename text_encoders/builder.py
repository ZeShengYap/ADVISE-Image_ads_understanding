
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import text_encoders_pb2

from text_encoders.bow_encoder import BOWEncoder
from text_encoders.rnn_encoder import RNNEncoder
from text_encoders.bi_rnn_encoder import BiRNNEncoder
from text_encoders.roberta_encoder import RoBERTaEncoder
from text_encoders.bert_encoder import BERTEncoder

def build(config, is_training=False):
  """Build a text encoder from config.

  Args:
    config: an instance of TextEncoder proto.
    is_training: if True, build training graph.

  Raises:
    ValueError: if config is invalid.

  Returns:
    text_encoder: an instance of TextEncoder.
  """
  if not isinstance(config, text_encoders_pb2.TextEncoder):
    raise ValueError('The config has to be an instance of TextEncoder.')

  encoder = config.WhichOneof('text_encoder')
  if 'bow_encoder' == encoder:
    return BOWEncoder(config.bow_encoder, is_training)
  if 'rnn_encoder' == encoder:
    return RNNEncoder(config.rnn_encoder, is_training)
  if 'bi_rnn_encoder' == encoder:
    return BiRNNEncoder(config.bi_rnn_encoder, is_training)
  if 'roberta_encoder' == encoder:
    return RoBERTaEncoder(config.roberta_encoder, is_training)
  if 'bert_encoder' == encoder:
    return BERTEncoder(config.bert_encoder, is_training)

  raise ValueError('Invalid text encoder %s.' % (encoder))
