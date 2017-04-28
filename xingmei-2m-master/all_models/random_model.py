import math
import tensorflow as tf
from tensorflow import flags
FLAGS=flags.FLAGS

import models
import utils

class RandomModel(models.BaseModel):
    """Inherit from this class when implementing new models."""
    def create_model(self, 
                     user_id, num_weeks, weeks, num_movies, movies, 
                     num_classes=588, candidates=None, label_week=None, labels=None, 
                     l2_penalty=1e-8,
                     **unused_params):
        
        num_users = FLAGS.num_users
        embedding_size = FLAGS.rm_embedding_size
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
        
        user_emb = tf.nn.embedding_lookup(user_emb_var, user_id)
        movie_emb = tf.matmul(movie_features, feature_emb_var)
        
        random_matrix=tf.truncated_normal(movie_emb.get_shape(),mean=0.0,stddev=math.sqrt(1.0/embedding_size))
        
        ratings=tf.einsum("ij,kj->ik",user_emb,random_matrix)
        predictions=tf.nn.sigmoid(ratings)
        return{"predictions":predictions}
