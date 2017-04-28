# 
# Author: He-Da Wang
# Email : whd.thu@gmail.com
# 
# This code is for the movie going prediction of xingmei data
# This code comes with NO WARRANTIES OR CONDITIONS OF ANY KIND
# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# 

import sys
from os.path import dirname
if dirname(__file__) not in sys.path:
  sys.path.append(dirname(__file__))
import json
import os
import time

import losses
import readers
import utils
import all_models
import eval_util

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile 
from tensorflow import logging

FLAGS = flags.FLAGS

if __name__ == "__main__":
    # Dataset flags
    flags.DEFINE_string("train_dir", "xm2m_model/", 
                        "The directory to save the model files in.")
    flags.DEFINE_string("train_data_pattern", "", 
                        "The file glob for the training dataset.")
    flags.DEFINE_integer("num_classes", 588, 
                        "The num of classes in the dataset.")
    flags.DEFINE_integer("max_weeks", 16, 
                        "The maximum num of weeks in a sequence of behaviours.")
    flags.DEFINE_integer("max_movies", 13, 
                        "The maximum num of movies watched in a week.")
    flags.DEFINE_integer("num_users", 304941, 
                        "The total number of users.")

    # Model tag
    flags.DEFINE_string("model", "MatrixFactorizationModel", 
                        "Which model to use.")
    flags.DEFINE_bool("start_new_model", False, 
                      "If set, this will not resume from a checkpoint and will instead create a"
                      " new model instance")

    # Training flags
    flags.DEFINE_integer("batch_size", 128, 
                        "How many examples to process per batch.")
    flags.DEFINE_string("label_loss", "CrossEntropyLoss", 
                        "Which loss function used to train the model.")
    flags.DEFINE_float("regularization_penalty", 1, 
                       "How much weight given to regularization loss (the weight for "
                       "label loss is 1.")
    flags.DEFINE_float("base_learning_rate", 0.01, 
                       "Learning rate to start with.")
    flags.DEFINE_float("learning_rate_decay", 0.95, 
                       "Learning rate decay rate applied to current learning rate "
                       "every #learning_rate_decay_examples.")
    flags.DEFINE_float("learning_rate_decay_examples", 2000000, 
                       "Multiply by current learning rate by #learning_rate_decay "
                       "every #learning_rate_decay_examples.")
    flags.DEFINE_string("optimizer", "AdamOptimizer", 
                         "Which optimizer to use in training.")
    flags.DEFINE_float("clip_gradient_norm", 1.0,
                       "Gradients larger than the norm will be clipped.")
    flags.DEFINE_integer("num_epochs", 50, 
                         "How many passes to make over the dataset before halting training.")
    flags.DEFINE_float("keep_checkpoint_every_n_hours", 0.05, 
                         "How many hours before keeping a checkpoint.")

    # Other flags
    flags.DEFINE_integer("num_readers", 2, 
                        "How many passes to make over the dataset before halting training.")
    flags.DEFINE_bool("log_device_placement", False, 
                      "Whether to write the device on which every op will run into the "
                      "logs on startup.")

# utils functions
def find_class_by_name(name, modules):
    modules = [getattr(module, name, None) for module in modules]
    return next(m for m in modules if m)

def clip_gradient_norms(gradients_to_variables, max_norm):
    clipped_grads_and_vars = []
    for grad, var in gradients_to_variables:
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                tmp = tf.clip_by_norm(grad.values, max_norm)
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad = tf.clip_by_norm(grad, max_norm)
        clipped_grads_and_vars.append((grad, var))
    return clipped_grads_and_vars

# get input tensors
def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=128,
                           num_epochs=None,
                           num_readers=1):
    
    logging.info("Using batch size of %d for training."%batch_size)
    with tf.name_scope("train_input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find any matching files for pattern %s."%data_pattern)
        logging.info("Number of training files: %d."%len(files))
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs, shuffle=True)
        training_data = [reader.prepare_reader(filename_queue) for _ in range(num_readers)]

        return tf.train.shuffle_batch_join(
            training_data,
            batch_size=batch_size,
            capacity=batch_size*5,
            min_after_dequeue=batch_size,
            allow_smaller_final_batch=True,
            enqueue_many=True)


# build graph
def build_graph(reader, model, train_data_pattern,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=128,
                base_learning_rate=0.01,
                learning_rate_decay_examples=2000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None):

    global_step = tf.Variable(0, trainable=False, name="global_step")

    learning_rate = tf.train.exponential_decay(
        learning_rate=base_learning_rate, 
        global_step=global_step*batch_size, 
        decay_steps=learning_rate_decay_examples,
        decay_rate=learning_rate_decay, 
        staircase=True)
    tf.summary.scalar("learning_rate", learning_rate)

    user_id, num_weeks, weeks, num_movies, movies, labels, label_week = \
        get_input_data_tensors(
            reader=reader,
            data_pattern=train_data_pattern,
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_readers=num_readers)
    candidates = utils.get_candidates_by_week(label_week, FLAGS.num_classes)
    tf.summary.histogram("model_input/num_weeks", num_weeks)
    tf.summary.histogram("model_input/weeks", weeks)
    tf.summary.histogram("model_input/num_movies", num_movies)
    tf.summary.histogram("model_input/movies", movies)

    with tf.name_scope("model"):
        result = model.create_model(
            user_id=user_id,
            num_weeks=num_weeks,
            weeks=weeks,
            num_movies=num_movies,
            movies=movies,
            num_classes=reader.num_classes,
            candidates=candidates,
            label_week=label_week,
            labels=labels)

        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

        predictions = result["predictions"]
        # only predict the available movies
        predictions = predictions * tf.cast(candidates, tf.float32)
        if "loss" in result.keys():
            label_loss = result["loss"]
        else:
            label_loss = label_loss_fn.calculate_loss(
                predictions=predictions,
                labels=labels,
                candidates=candidates)
        tf.summary.scalar("label_loss", label_loss)

        if "regulation_loss" in result.keys():
            reg_loss = result["regulation_loss"]
        else:
            reg_loss = tf.constant(0.0)
        reg_losses = tf.losses.get_regularization_losses()
        if reg_losses:
            reg_loss += tf.add_n(reg_losses)
        if regularization_penalty != 0:
            tf.summary.scalar("reg_loss", reg_loss)

        final_loss = regularization_penalty * reg_loss + label_loss

        optimizer = optimizer_class()
        gradients = optimizer.compute_gradients(final_loss, 
            colocate_gradients_with_ops=False)
        if clip_gradient_norm > 0:
            with tf.name_scope("clip_grads"):
                gradients = clip_gradient_norms(gradients, clip_gradient_norm)
        train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        tf.add_to_collection("global_step", global_step)
        tf.add_to_collection("loss", label_loss)
        tf.add_to_collection("user_id", user_id)
        tf.add_to_collection("num_weeks", num_weeks)
        tf.add_to_collection("weeks", weeks)
        tf.add_to_collection("num_movies", num_movies)
        tf.add_to_collection("movies", movies)
        tf.add_to_collection("labels", labels)
        tf.add_to_collection("predictions", predictions)
        tf.add_to_collection("candidates", candidates)
        tf.add_to_collection("train_op", train_op)

class Trainer(object):
    def __init__(self, cluster, task, train_dir, log_device_placement=True):
        """Creates a trainer"""
        self.cluster = cluster
        self.task = task
        self.is_master = (task.type == "master" and task.index == 0)
        self.train_dir = train_dir
        self.config = tf.ConfigProto(log_device_placement=log_device_placement)
        if self.is_master and self.task.index > 0:
            raise StandardError("%s: Only one replica of master expected",
                                task_as_string(self.task))

    def run(self, start_new_model=False):
        if self.is_master and start_new_model:
            self.remove_training_directory(self.train_dir)
        target, device_fn = self.start_server_if_distributed()
        meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

        with tf.Graph().as_default() as graph:
            if meta_filename:
                saver = self.recover_model(meta_filename)

            with tf.device(device_fn):
                if not meta_filename:
                    saver = self.build_model()

                global_step = tf.get_collection("global_step")[0]
                loss = tf.get_collection("loss")[0]
                predictions = tf.get_collection("predictions")[0]
                labels = tf.get_collection("labels")[0]
                candidates = tf.get_collection("candidates")[0]
                train_op = tf.get_collection("train_op")[0]
                init_op = tf.global_variables_initializer()

        sv = tf.train.Supervisor(
            graph,
            logdir=self.train_dir,
            init_op=init_op,
            is_chief=self.is_master,
            global_step=global_step,
            save_model_secs=3 * 60,
            save_summaries_secs=120,
            saver=saver)

        logging.info("%s: Starting managed session.", task_as_string(self.task))
        with sv.managed_session(target, config=self.config) as sess:
            try:
                logging.info("%s: Entering training loop.", task_as_string(self.task))
                while not sv.should_stop():

                    batch_start_time = time.time()
                    _, global_step_val, loss_val, predictions_val, labels_val = sess.run(
                        [train_op, global_step, loss, predictions, labels])
                    seconds_per_batch = time.time() - batch_start_time

                    if self.is_master:
                        examples_per_second = labels_val.shape[0] / seconds_per_batch
                        hit_at_one = eval_util.calculate_hit_at_one(predictions_val,
                                                                labels_val)
                        perr = eval_util.calculate_precision_at_equal_recall_rate(
                            predictions_val, labels_val)
                        gap = eval_util.calculate_gap(predictions_val, labels_val)

                        logging.info(
                            "%s: training step " + str(global_step_val) + "| Hit@1: " +
                            ("%.2f" % hit_at_one) + " PERR: " + ("%.2f" % perr) + " GAP: " +
                            ("%.2f" % gap) + " Loss: " + str(loss_val),
                            task_as_string(self.task))
    
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_Hit@1", hit_at_one),
                            global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_Perr", perr), global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_GAP", gap), global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("global_step/Examples/Second",
                                            examples_per_second), global_step_val)
                        sv.summary_writer.flush()
    
            except tf.errors.OutOfRangeError:
                logging.info("%s: Done training -- epoch limit reached.",
                             task_as_string(self.task))

        logging.info("%s: Exited training loop.", task_as_string(self.task))
        sv.Stop()

    def start_server_if_distributed(self):
        """Starts a server if the execution is distributed."""
        if self.cluster:
            logging.info("%s: Starting trainer within cluster %s.",
                         task_as_string(self.task), self.cluster.as_dict())
            server = start_server(self.cluster, self.task)
            target = server.target
            device_fn = tf.train.replica_device_setter(
                ps_device="/job:ps",
                worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
                cluster=self.cluster)
        else:
            target = ""
            device_fn = ""
        return (target, device_fn)

    def remove_training_directory(self, train_dir):
        """Removes the training directory."""
        try:
            logging.info(
                "%s: Removing existing train directory.",
                task_as_string(self.task))
            gfile.DeleteRecursively(train_dir)
        except:
            logging.error(
                "%s: Failed to delete directory " + train_dir +
                " when starting a new model. Please delete it manually and" +
                " try again.", task_as_string(self.task))

    def get_meta_filename(self, start_new_model, train_dir):
        if start_new_model:
            logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                         task_as_string(self.task))
            return None
    
        latest_checkpoint = tf.train.latest_checkpoint(train_dir)
        if not latest_checkpoint:
            logging.info("%s: No checkpoint file found. Building a new model.",
                         task_as_string(self.task))
            return None
    
        meta_filename = latest_checkpoint + ".meta"
        if not gfile.Exists(meta_filename):
            logging.info("%s: No meta graph file found. Building a new model.",
                         task_as_string(self.task))
            return None
        else:
            return meta_filename
  
    def recover_model(self, meta_filename):
        logging.info("%s: Restoring from meta graph file %s",
                     task_as_string(self.task), meta_filename)
        return tf.train.import_meta_graph(meta_filename)
  
    def build_model(self):
        """Find the model and build the graph."""
    
        reader = readers.XM2MFeatureReader(
            num_classes=FLAGS.num_classes,
            max_weeks=FLAGS.max_weeks,
            max_movies=FLAGS.max_movies)
    
        model = find_class_by_name(FLAGS.model, [all_models])()
        label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
        optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])
    
        build_graph(reader=reader,
                    model=model,
                    optimizer_class=optimizer_class,
                    clip_gradient_norm=FLAGS.clip_gradient_norm,
                    train_data_pattern=FLAGS.train_data_pattern,
                    label_loss_fn=label_loss_fn,
                    base_learning_rate=FLAGS.base_learning_rate,
                    learning_rate_decay=FLAGS.learning_rate_decay,
                    learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                    regularization_penalty=FLAGS.regularization_penalty,
                    num_readers=FLAGS.num_readers,
                    batch_size=FLAGS.batch_size,
                    num_epochs=FLAGS.num_epochs)
    
        logging.info("%s: Built graph.", task_as_string(self.task))
        return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)

class ParameterServer(object):
  """A parameter server to serve variables in a distributed execution."""

  def __init__(self, cluster, task):
    """Creates a ParameterServer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task

  def run(self):
    """Starts the parameter server."""

    logging.info("%s: Starting parameter server within cluster %s.",
                 task_as_string(self.task), self.cluster.as_dict())
    server = start_server(self.cluster, self.task)
    server.join()


def start_server(cluster, task):
  """Creates a Server.

  Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed.
      None otherwise.
    task: A TaskSpec describing the job type and the task index.
  """

  if not task.type:
    raise ValueError("%s: The task type must be specified." %
                     task_as_string(task))
  if task.index is None:
    raise ValueError("%s: The task index must be specified." %
                     task_as_string(task))

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol="grpc",
      job_name=task.type,
      task_index=task.index)

def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)

def main(unused_argv):
  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  # Logging the version.
  logging.set_verbosity(tf.logging.INFO)
  logging.info("%s: Tensorflow version: %s.",
               task_as_string(task), tf.__version__)

  # Dispatch to a master, a worker, or a parameter server.
  if not cluster or task.type == "master" or task.type == "worker":
    Trainer(cluster, task, FLAGS.train_dir, FLAGS.log_device_placement).run(
            start_new_model=FLAGS.start_new_model)
  elif task.type == "ps":
    ParameterServer(cluster, task).run()
  else:
    raise ValueError("%s: Invalid task_type: %s." %
                     (task_as_string(task), task.type))

if __name__ == "__main__":
  app.run()
