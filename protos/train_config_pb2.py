# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/train_config.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19protos/train_config.proto\"\x9e\x03\n\x0bTrainConfig\x12\x1d\n\toptimizer\x18\x01 \x01(\x0b\x32\n.Optimizer\x12\x15\n\rlearning_rate\x18\x02 \x01(\x02\x12\'\n\x19learning_rate_decay_steps\x18\x03 \x01(\x02:\x04\x32\x30\x30\x30\x12#\n\x18learning_rate_decay_rate\x18\x04 \x01(\x02:\x01\x31\x12%\n\x17learning_rate_staircase\x18\x05 \x01(\x08:\x04true\x12\x17\n\x0fnumber_of_steps\x18\x0b \x01(\x05\x12\x1d\n\x11log_every_n_steps\x18\x0c \x01(\x05:\x02\x31\x30\x12\x1f\n\x12save_interval_secs\x18\r \x01(\x05:\x03\x36\x30\x30\x12 \n\x13save_summaries_secs\x18\x0e \x01(\x05:\x03\x36\x30\x30\x12\x1d\n\x0emoving_average\x18\x0f \x01(\x08:\x05\x66\x61lse\x12\x30\n\x13gradient_multiplier\x18\x10 \x03(\x0b\x32\x13.GradientMultiplier\x12\x18\n\x10\x65xclude_variable\x18\x11 \x03(\t\"7\n\x12GradientMultiplier\x12\r\n\x05scope\x18\x01 \x01(\t\x12\x12\n\nmultiplier\x18\x02 \x01(\x02\"\x84\x01\n\tOptimizer\x12\x1e\n\x04\x61\x64\x61m\x18\x01 \x01(\x0b\x32\x0e.AdamOptimizerH\x00\x12$\n\x07\x61\x64\x61grad\x18\x02 \x01(\x0b\x32\x11.AdagradOptimizerH\x00\x12$\n\x07rmsprop\x18\x03 \x01(\x0b\x32\x11.RMSPropOptimizerH\x00\x42\x0b\n\toptimizer\"\x0f\n\rAdamOptimizer\"\x12\n\x10\x41\x64\x61gradOptimizer\";\n\x10RMSPropOptimizer\x12\x12\n\x05\x64\x65\x63\x61y\x18\x01 \x01(\x02:\x03\x30.9\x12\x13\n\x08momentum\x18\x02 \x01(\x02:\x01\x30')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.train_config_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _TRAINCONFIG._serialized_start=30
  _TRAINCONFIG._serialized_end=444
  _GRADIENTMULTIPLIER._serialized_start=446
  _GRADIENTMULTIPLIER._serialized_end=501
  _OPTIMIZER._serialized_start=504
  _OPTIMIZER._serialized_end=636
  _ADAMOPTIMIZER._serialized_start=638
  _ADAMOPTIMIZER._serialized_end=653
  _ADAGRADOPTIMIZER._serialized_start=655
  _ADAGRADOPTIMIZER._serialized_end=673
  _RMSPROPOPTIMIZER._serialized_start=675
  _RMSPROPOPTIMIZER._serialized_end=734
# @@protoc_insertion_point(module_scope)
