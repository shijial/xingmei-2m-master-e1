# 
# Author: He-Da Wang
# Email : whd.thu@gmail.com
# 
# This code is for the movie going prediction of xingmei data
# This code comes with NO WARRANTIES OR CONDITIONS OF ANY KIND
# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# 

import tensorflow as tf
import numpy
import numpy as np
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("candidates_file", "data/xingmei-2m/other/screens",
                    "File that map the week number to the candidate movies.")
flags.DEFINE_string("features_file", "data/xingmei-2m/other/features",
                    "File that map the movie id to features.")

def get_candidates_by_week(label_week, num_classes):
    with open(FLAGS.candidates_file) as F:
        lines = map(lambda x: map(int, x.strip().split()), F.readlines())
        max_week_num = max(map(lambda x: x[0], lines))
        weekly_screens = [[0]*num_classes for i in xrange(max_week_num+1)]
        for screens in lines:
            week_num = screens[0]
            screens = screens[2:]
            for i in xrange(0, len(screens), 2):
                mid = screens[i]
                weekly_screens[week_num][mid] = 1.0
        weekly_screens = np.array(weekly_screens)
        weekly_candidates = tf.get_variable("weekly_candidates", 
            initializer=tf.constant_initializer(weekly_screens),
            trainable=False, shape=[max_week_num+1,num_classes])
        candidates = tf.nn.embedding_lookup(weekly_candidates, label_week)
        return candidates
            
    
def get_features_variable(num_classes):
    def flat(list2d):
        list1d = []
        for l in list2d:
            list1d.extend(l)
        return list1d
    with open(FLAGS.features_file) as F:
        lines = map(lambda x: map(int, x.strip().split()), F.readlines())
        feature_length = max(flat(map(lambda x: x[1:], lines))) + 1
        feature_array = [[0]*feature_length for i in xrange(num_classes)]
        for features in lines:
            mid = features[0]
            features = features[1:]
            for f in features:
                feature_array[mid][f] = 1.0
        feature_array = np.array(feature_array)
        movie_features = tf.get_variable("feature_array", 
            initializer=tf.constant_initializer(feature_array),
            trainable=False, shape=[num_classes,feature_length])
        return movie_features

def get_mask_map(max_range):
    mask_array = [[0.0]*max_range for _ in xrange(max_range+1)]
    for i in xrange(max_range+1):
        for j in xrange(i):
            mask_array[i][j] = 1.0
    mask_array = np.array(mask_array)
    mask_array = tf.get_variable("mask_array", 
        initializer=tf.constant_initializer(mask_array),
        trainable=False, shape=[max_range+1,max_range])
    return mask_array

def MakeSummary(name, value):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  return summary


def AddGlobalStepSummary(summary_writer,
                         global_step_val,
                         global_step_info_dict,
                         summary_scope="Eval"):
  """Add the global_step summary to the Tensorboard.

  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    global_step_info_dict: a dictionary of the evaluation metrics calculated for
      a mini-batch.
    summary_scope: Train or Eval.

  Returns:
    A string of this global_step summary
  """
  this_hit_at_one = global_step_info_dict["hit_at_one"]
  this_perr = global_step_info_dict["perr"]
  this_loss = global_step_info_dict["loss"]
  examples_per_second = global_step_info_dict.get("examples_per_second", -1)

  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@1", this_hit_at_one),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Perr", this_perr),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Loss", this_loss),
      global_step_val)

  if examples_per_second != -1:
    summary_writer.add_summary(
        MakeSummary("GlobalStep/" + summary_scope + "_Example_Second",
                    examples_per_second), global_step_val)

  summary_writer.flush()
  info = ("global_step {0} | Batch Hit@1: {1:.3f} | Batch PERR: {2:.3f} | Batch Loss: {3:.3f} "
          "| Examples_per_sec: {4:.3f}").format(
              global_step_val, this_hit_at_one, this_perr, this_loss,
              examples_per_second)
  return info


def AddEpochSummary(summary_writer,
                    global_step_val,
                    epoch_info_dict,
                    summary_scope="Eval"):
  """Add the epoch summary to the Tensorboard.

  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    epoch_info_dict: a dictionary of the evaluation metrics calculated for the
      whole epoch.
    summary_scope: Train or Eval.

  Returns:
    A string of this global_step summary
  """
  epoch_id = epoch_info_dict["epoch_id"]
  avg_hit_at_one = epoch_info_dict["avg_hit_at_one"]
  avg_perr = epoch_info_dict["avg_perr"]
  avg_loss = epoch_info_dict["avg_loss"]
  aps = epoch_info_dict["aps"]
  gap = epoch_info_dict["gap"]
  mean_ap = numpy.mean(aps)

  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@1", avg_hit_at_one),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Perr", avg_perr),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Loss", avg_loss),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_MAP", mean_ap),
          global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_GAP", gap),
          global_step_val)
  summary_writer.flush()

  info = ("epoch/eval number {0} | Avg_Hit@1: {1:.3f} | Avg_PERR: {2:.3f} "
          "| MAP: {3:.3f} | GAP: {4:.3f} | Avg_Loss: {5:3f}").format(
          epoch_id, avg_hit_at_one, avg_perr, mean_ap, gap, avg_loss)
  return info

