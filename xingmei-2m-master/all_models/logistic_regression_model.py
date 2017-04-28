
import math
import tensorflow as tf
from tensorflow import flags
FLAGS = flags.FLAGS

import models
import utils

class LogisticRegressionModel(models.BaseModel):
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
         
         
        num_users = FLAGS.num_users
        max_weeks = FLAGS.max_weeks
        embedding_size = FLAGS.lr_embedding_size 

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
        inputs = tf.reduce_sum(movie_feature_emb, axis=2)
        
        
        log_reg_weights = tf.get_variable("log_reg_weights",
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=math.sqrt(1.0 / embedding_size)),
            regularizer=tf.contrib.layers.l2_regularizer(l2_penalty),
            shape=[max_weeks,embedding_size,num_classes])
            
        log_reg_bias=tf.get_variable("log_reg_bias",
            initializer=tf.zeros_initializer(),
            regularizer=None,
            shape=[1,num_classes])
        
        ratings=tf.einsum("ijk,jkl->il",inputs,log_reg_weights)+log_reg_bias
        predictions=tf.nn.sigmoid(ratings)
       
        return {"predictions": predictions}
