import os, sys, random

import numpy as np
import tensorflow as tf

from models.nn.prediction_model_nn import PredictionModelNN

from models.bd_nn.process_data_bd_nn import ProcessDataBDNN

class PredictionModelBDNN(PredictionModelNN):

    def __init__(self, exp_folder, data_folder, params):
        self.process_data = ProcessDataBDNN(data_folder, params)
        super(PredictionModelNN, self).__init__(exp_folder, data_folder, params)

        self._graph_inference = self._get_old_graph_inference(graph_type=params['graph_type'])

        self._graph_setup_eval()

    #############
    ### Files ###
    #############

    @property
    def _this_file(self):
        return os.path.abspath(__file__.replace('.pyc', '.py'))

    #####################
    ### Graph methods ###
    #####################

    def _graph_input_output_from_file(self, is_train, shuffle=True):
        with tf.name_scope('train_input_output_from_file' if is_train else 'val_input_output_from_file'):
            ### create file queue
            fnames = self.process_data.all_train_data_files if is_train else self.process_data.all_val_data_files
            random.shuffle(fnames)
            filename_queue = tf.train.string_input_producer(fnames, num_epochs=None, shuffle=shuffle, capacity=8)

            ### read and decode
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            features = {
                'fname': tf.FixedLenFeature([], tf.string)
            }
            for b in xrange(self.params['bootstrap']):
                features['input_b{0}'.format(b)] = tf.FixedLenFeature([np.prod(self.process_data.input_shape)], tf.float32)
                features['output_b{0}'.format(b)] = tf.FixedLenFeature([np.prod(self.process_data.output_shape)], tf.float32)

            parsed_example = tf.parse_single_example(serialized_example, features=features)

            fname = parsed_example['fname']
            input = [tf.reshape(parsed_example['input_b{0}'.format(b)], self.process_data.input_shape)
                     for b in xrange(self.params['bootstrap'])]
            output = [tf.reshape(parsed_example['output_b{0}'.format(b)], self.process_data.output_shape)
                      for b in xrange(self.params['bootstrap'])]

            ### shuffle and put into batches
            if shuffle:
                batched = tf.train.shuffle_batch(
                    [fname] + input + output, batch_size=self.params['batch_size'], num_threads=2,
                    capacity=1000 + 3 * self.params['batch_size'],
                    min_after_dequeue=1000)
            else:
                batched = tf.train.batch(
                    [fname] + input + output, batch_size=self.params['batch_size'], num_threads=2,
                    capacity=1000 + 3 * self.params['batch_size'])

        fnames = batched[0]
        inputs = batched[1:1+self.params['bootstrap']]
        outputs = batched[1+self.params['bootstrap']:]

        return fnames, inputs, outputs, filename_queue

    def _graph_input_output_from_placeholders(self):
        with tf.name_scope('from_feed'):
            inputs = [tf.placeholder('float', [None] + self.process_data.input_shape, name='input_b{0}'.format(b))
                      for b in xrange(self.params['bootstrap'])]
            outputs = [tf.placeholder('float', [None] + self.process_data.output_shape, name='output_b{0}'.format(b))
                       for b in xrange(self.params['bootstrap'])]

        return inputs, outputs

    def _get_old_graph_inference(self, graph_type):
        self.logger.info('Graph type: {0}'.format(graph_type))
        sys.path.append(os.path.dirname(self._code_file))
        exec ('from {0} import {1} as OldPredictionModel'.format(
            os.path.basename(self._code_file).split('.')[0], 'PredictionModelBDNN'))

        if graph_type == 'fc':
            return OldPredictionModel._graph_inference_fc
        else:
            raise Exception('graph_type {0} is not valid'.format(graph_type))

    @staticmethod
    def _graph_inference_fc(name, input, input_shape, output_shape, reuse=False, random_seed=None):
        tf.set_random_seed(random_seed)

        pred_output = []
        with tf.name_scope(name + '_inference'):

            for b, input_b in enumerate(input):
                ### define variables
                with tf.variable_scope('inference_vars_b{0}'.format(b), reuse=reuse):
                    weights_b = [
                        tf.get_variable('w_hidden_0_b{0}'.format(b), [np.prod(input_shape), 40],
                                        initializer=tf.contrib.layers.xavier_initializer()),
                        tf.get_variable('w_hidden_1_b{0}'.format(b), [40, 40],
                                        initializer=tf.contrib.layers.xavier_initializer()),
                        tf.get_variable('w_output_b{0}'.format(b), [40, np.prod(output_shape)],
                                        initializer=tf.contrib.layers.xavier_initializer()),
                    ]
                    biases_b = [
                        tf.get_variable('b_hidden_0_b{0}'.format(b), [40], initializer=tf.constant_initializer(0.)),
                        tf.get_variable('b_hidden_1_b{0}'.format(b), [40], initializer=tf.constant_initializer(0.)),
                        tf.get_variable('b_output_b{0}'.format(b), [np.prod(output_shape)], initializer=tf.constant_initializer(0.)),
                    ]

                ### weight decays
                for v in weights_b + biases_b:
                    tf.add_to_collection('weight_decays', 1.0 * tf.nn.l2_loss(v))

                ### fully connected relus
                layer = tf.reshape(input_b, [-1, np.prod(input_shape)])
                for i, (weight_b, bias_b) in enumerate(zip(weights_b, biases_b)):
                    with tf.name_scope('hidden_{0}_b{1}'.format(i, b)):
                        layer = tf.add(tf.matmul(layer, weight_b), bias_b)
                        if i < len(weights_b) - 1:
                            layer = tf.nn.relu(layer)

                pred_output_b = tf.reshape(layer, [-1] + output_shape)
                pred_output.append(pred_output_b)

        return pred_output

    def _graph_cost(self, name, pred_output, output):
        """
        pred_output and output lists of length bootstrap
        """
        with tf.name_scope(name + '_cost_and_err'):
            cost = tf.reduce_mean([tf.reduce_mean((pred_output_b - output_b) * (pred_output_b - output_b))
                                   for pred_output_b, output_b in zip(pred_output, output)])
            weight_decay = self.params['reg'] * tf.add_n(tf.get_collection('weight_decays'))
            total_cost = cost + weight_decay

        return total_cost, cost

    ##################
    ### Evaluating ###
    ##################

    def eval(self, input):
        feed_dict = dict([(input_b, input) for input_b in self.input])
        pred_output = None
        for _ in xrange(self.params['samples']):
            pred_output_s = self.sess.run(self.output_pred, feed_dict=feed_dict)
            pred_output_s = [np.expand_dims(po, 1) for po in pred_output_s]
            pred_output_s = np.concatenate(pred_output_s, axis=1)
            if pred_output is None:
                pred_output = pred_output_s
            else:
                pred_output = np.concatenate((pred_output, pred_output_s), axis=1)

        assert(len(pred_output.shape) == 4)
        assert(pred_output.shape[1] == self.params['bootstrap'] * self.params['samples'] * len(input))
        assert(pred_output.shape[2] == self.params['H'])
        assert(pred_output.shape[3] == 2)

        return pred_output
