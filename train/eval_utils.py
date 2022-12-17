
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.compat.v1 import logging

def evaluate(results, groundtruths):
  """Evaluate the statement ranking task.

  Args:
    results: a dict mapping from image_id to 15 float numbers denoting distances.

  Returns:
    metrics: a dict mapping from metric name to score, involving:
      accuracy: the ratio of correct top-1 prediction.
      rank-med: the medain value of the groundtruths' ranks.
      rank-avg: the average value of the groundtruths' ranks.
      rank-min: the minimum value of the groundtruths' ranks.
  """
  #logging.info('THIS IS THE NEW ONE LOADED')
  if len(results) != len(groundtruths):
    logging.warn(
        'size of gts: %i, size of res: %i', len(groundtruths), len(results))

  #logging.info('get to gts')
  all_accuracy, all_recall_at_3 = [], []
  all_rank_min, all_rank_avg, all_rank_med = [], [], []

  #print(results.items())
  #logging.info('get past items')
  for image_id, result in results.items():
    #logging.info('in the loooop')
    image_id = image_id.decode('utf-8')
    #logging.info(f'help what is tis {result.keys()}')
    #assert image_id in groundtruths
    #logging.info(str(type(groundtruths)))
    distances = result['distances']
    #logging.info(f'help what is this {distances}')

    #logging.info('gets hereeeee')
    pos_examples = groundtruths[image_id]['pos_examples']
    all_examples = groundtruths[image_id]['all_examples']
    
    #logging.info('gets to before np distances')
    distances = np.array(distances)

    #logging.info('gets to after np distances')
    ranking_r = distances.argsort()
    #logging.info('argsort no prob')
    ranking = np.array(ranking_r)
    #logging.info('ranking no prob')
    #logging.info(f'ranking_r is {ranking_r}')
    #logging.info(f'ranking is {ranking}')
    for i, rank in enumerate(ranking_r):
      ranking[rank] = i

    #logging.info(f'apa in sial {[all_examples.index(example) for example in pos_examples]}')
    positions = ranking[[all_examples.index(example) for example in pos_examples]]
    positions = np.sort(positions)
    
    #logging.info('gets past positions') 
    all_accuracy.append(positions[0] == 0)
    #logging.info('gets past first all_accuracy')
    all_recall_at_3.append(sum([1 for pos in positions if pos < 3]))
    all_rank_min.append(1 + positions[0])
    all_rank_avg.append(1 + np.mean(positions))
    all_rank_med.append(1 + np.median(positions))

  mean_func = lambda x: round(np.array(x).astype(np.float).mean(), 4)
  eval_results = {
    'accuracy': mean_func(all_accuracy),
    'recall_at_3': mean_func(all_recall_at_3),
    'rank_min': mean_func(all_rank_min),
    'rank_avg': mean_func(all_rank_avg),
    'rank_med': mean_func(all_rank_med),
  }
  return eval_results
