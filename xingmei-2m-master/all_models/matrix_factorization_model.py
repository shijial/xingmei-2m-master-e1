
import math
import tensorflow as tf
from tensorflow import flags
FLAGS = flags.FLAGS

import models
import utils

class MatrixFactorizationModel(models.BaseModel):
    """Inherit from this class when implementing new models."""

    def create_model(self, 
                     user_id, num_weeks=None, weeks=None, num_movies=None, movies=None, 
                     num_classes=588, candidates=None, label_week=None, labels=None, 
                     l2_penalty=1e-8,
                     **unused_params):
        embedding_size = FLAGS.mf_embedding_size
        movie_features = utils.get_features_variable(num_classes)
        num_features = movie_features.get_shape().as_list()[-1]
        num_users = FLAGS.num_users
        user_emb_var = tf.get_variable("user_emb_var",
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=math.sqrt(1.0 / embedding_size)),
            regularizer=tf.contrib.layers.l2_regularizer(l2_penalty),
            shape=[num_users, embedding_size])
        feature_emb_var = tf.get_variable("feature_emb_var",
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=math.sqrt(1.0 / embedding_size)),
            regularizer=tf.contrib.layers.l2_regularizer(l2_penalty),
            shape=[num_features, embedding_size])
        user_bias_var = tf.get_variable("user_bias_var",
            initializer=tf.zeros_initializer(),
            regularizer=None,
            shape=[num_users, 1])
        movie_bias_var = tf.get_variable("movie_bias_var",
            initializer=tf.zeros_initializer(),
            regularizer=None,
            shape=[1, num_classes])

        user_emb = tf.nn.embedding_lookup(user_emb_var, user_id)
        movie_emb = tf.matmul(movie_features, feature_emb_var)
        user_bias = tf.nn.embedding_lookup(user_bias_var, user_id)
        movie_bias = movie_bias_var
        print user_emb, movie_emb, user_bias
        ratings = tf.einsum("ij,kj->ik", user_emb, movie_emb) + user_bias + movie_bias
        predictions = tf.nn.sigmoid(ratings)
        return {"predictions": predictions}
