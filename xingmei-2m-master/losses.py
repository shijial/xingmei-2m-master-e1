# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 
# Author: He-Da Wang
# Email : whd.thu@gmail.com
# 
# This code is for the movie going prediction of xingmei data
# This code comes with NO WARRANTIES OR CONDITIONS OF ANY KIND
# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# 

"""Provides definitions for non-regularized training or test losses."""
import tensorflow as tf

class BaseLoss(object):
  """Inherit from this class when implementing new losses."""
  def calculate_loss(self, unused_predictions, unused_labels, unused_candidates, **unused_params):
    raise NotImplementedError()


class CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """
  def calculate_loss(self, predictions, labels, candidates, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      float_candidates = tf.cast(candidates, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      cross_entropy_loss = cross_entropy_loss * float_candidates
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


