
import math
import tensorflow as tf
from tensorflow import flags
FLAGS = flags.FLAGS

import models
import utils

class BasicLstmModel(models.BaseModel):
    """Inherit from this class when implementing new models."""

    def create_model(self, 
                     user_id, num_weeks, weeks, num_movies, movies, 
                     num_classes=588, candidates=None, label_week=None, labels=None, 
                     l2_penalty=1e-8,
                     **unused_params):
        """
        user_id:    batch_size tensor, the user id in this example
        num_weeks:  batch_size tensor, the number of weeks valid in weeks, num_movies and movies tensor
        weeks:      batch_size x max_weeks tensor, the week numbers
        num_movies: batch_size x max_weeks tensor, the number of movies that user watched in the week
                                                   corresponds to the week in week numbers
        movies:     batch_size x max_weeks x max_movies tensor, the movie id watched by the user in the 
                                                                correspoding week
        """
        lstm_sizes = FLAGS.lstm_cells
        lstm_layers = FLAGS.lstm_layers 
        num_users = FLAGS.num_users
        embedding_size = FLAGS.rnn_embedding_size 

        # Feature Map: num_classes x num_features
        feature_map = utils.get_features_variable(num_classes)
        num_features = feature_map.get_shape().as_list()[-1]
        max_movies = movies.get_shape().as_list()[-1]
        movie_mask_map = utils.get_mask_map(max_movies)

        # user embedding lookup table: num_users x 
        user_emb_var = tf.get_variable("user_emb_var",
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=math.sqrt(1.0 / embedding_size)),
            regularizer=tf.contrib.layers.l2_regularizer(l2_penalty),
            shape=[num_users, embedding_size])

        feature_emb_var = tf.get_variable("feature_emb_var",
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=math.sqrt(1.0 / embedding_size)),
            regularizer=tf.contrib.layers.l2_regularizer(l2_penalty),
            shape=[num_features, embedding_size])

        # num_classes x embedding_size
        movie_emb = tf.matmul(feature_map, feature_emb_var)
        # batch_size x max_weeks x max_movies
        movie_mask = tf.nn.embedding_lookup(movie_mask_map, num_movies)
        # batch_size x max_weeks x max_movies x embedding_size
        movie_feature_emb = tf.nn.embedding_lookup(movie_emb, movies)
        movie_feature_emb = tf.einsum("ijkl,ijk->ijkl", movie_feature_emb, movie_mask)
        # batch_size x max_weeks x embedding_size
        lstm_inputs = tf.reduce_sum(movie_feature_emb, axis=2)
        
        with tf.variable_scope("RNN"):

            lstm_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.BasicLSTMCell(lstm_sizes, state_is_tuple=True)
                    for _ in xrange(lstm_layers)],
                state_is_tuple=True)

            outputs, state = tf.nn.dynamic_rnn(lstm_cell, lstm_inputs, 
                                         sequence_length=num_weeks, 
                                         initial_state=None, # could be user-specific
                                         dtype=tf.float32,
                                         swap_memory=FLAGS.rnn_swap_memory)

        final_state = tf.concat(map(lambda x: x.h, state), axis=1)
        predictions = tf.contrib.layers.fully_connected(
            inputs = final_state,
            num_outputs = num_classes,
            activation_fn = tf.nn.sigmoid,
            weights_regularizer = tf.contrib.layers.l2_regularizer(l2_penalty),
            trainable = True)

        return {"predictions": predictions}
