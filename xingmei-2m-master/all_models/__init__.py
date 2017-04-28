
from tensorflow import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("mf_embedding_size", 64, 
                     "The embedding size in matrix factorization model.")

flags.DEFINE_integer("lstm_cells", 64, 
                     "The number of cell units per layer in rnn model.")
flags.DEFINE_integer("lstm_layers", 2, 
                     "The number of layers in rnn model.")
flags.DEFINE_integer("rnn_embedding_size", 64, 
                     "The embedding size in rnn model.")
flags.DEFINE_bool("rnn_swap_memory", True, 
                  "Use swap memory option in dynamic rnn that saves memory usage.")

flags.DEFINE_integer("rm_embedding_size", 64, 
                     "The embedding size in random model.")  

flags.DEFINE_integer("lr_embedding_size", 64, 
                     "The embedding size in logistic regression model.")                      
                  
                  
from matrix_factorization_model import *
from basic_lstm_model import *
from random_model import *
from logistic_regression_model import *

