# 
# Author: He-Da Wang
# Email : whd.thu@gmail.com
# 
# This code is for the movie going prediction of xingmei data
# This code comes with NO WARRANTIES OR CONDITIONS OF ANY KIND
# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# 

import tensorflow as tf
import utils

class BaseReader(object):
    def prepare_reader(self, unused_filename_queue, **unused_params):
        raise NotImplementedError()

class XM2MFeatureReader(BaseReader):
    """Decode xingmei feature"""
    def __init__(self,
               num_classes=588,
               max_weeks=16,
               max_movies=13):
        self.max_weeks = max_weeks
        self.max_movies = max_movies 
        self.num_classes = num_classes

    def prepare_reader(self, filename_queue, batch_size=128):
        reader = tf.TFRecordReader()
        _, serialized_examples = reader.read_up_to(filename_queue, batch_size)
        max_weeks = self.max_weeks
        max_movies = self.max_movies
        feature_map = {
            "labels": tf.VarLenFeature(tf.int64),
            "uid": tf.FixedLenFeature([], tf.int64),
            "num_weeks": tf.FixedLenFeature([], tf.int64),
            "label_week": tf.FixedLenFeature([], tf.int64),
            "weeks": tf.FixedLenFeature([max_weeks], tf.int64),
            "num_movies": tf.FixedLenFeature([max_weeks], tf.int64),
            "movies": tf.FixedLenFeature([max_weeks, max_movies], tf.int64)
        }
        features = tf.parse_example(serialized_examples, features=feature_map)
        labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
        labels.set_shape([None, self.num_classes])
        return features["uid"], features["num_weeks"], features["weeks"], features["num_movies"], features["movies"], \
               labels, features["label_week"]
 

