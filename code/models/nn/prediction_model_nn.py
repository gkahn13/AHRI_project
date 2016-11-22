import os, sys, time, random
from collections import defaultdict

import numpy as np
import tensorflow as tf

from models.nn.process_data_nn import ProcessDataNN

from general.models.prediction_model import PredictionModel, MLPlotter

class PredictionModelNN(PredictionModel):

    def __init__(self, exp_folder, data_folder, params):
        self.process_data = ProcessDataNN(data_folder, params)
        PredictionModel.__init__(self, exp_folder, data_folder, params)

        self._graph_inference = self._get_old_graph_inference(graph_type=params['graph_type'])

        self._graph_setup_eval_called = False

    #############
    ### Files ###
    #############

    @property
    def _this_file(self):
        return os.path.abspath(__file__.replace('.pyc', '.py'))

    @property
    def _model_file(self):
        return os.path.join(self.exp_folder, 'model.ckpt')

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
                'fname': tf.FixedLenFeature([], tf.string),
                'input': tf.FixedLenFeature([np.prod(self.process_data.input_shape)], tf.float32),
                'output': tf.FixedLenFeature([np.prod(self.process_data.output_shape)], tf.float32),
            }

            parsed_example = tf.parse_single_example(serialized_example, features=features)

            fname = parsed_example['fname']
            input = tf.reshape(parsed_example['input'], self.process_data.input_shape)
            output = tf.reshape(parsed_example['output'], self.process_data.output_shape)

            ### shuffle and put into batches
            if shuffle:
                fnames, inputs, outputs = tf.train.shuffle_batch(
                    [fname, input, output], batch_size=self.params['batch_size'], num_threads=2,
                    capacity=1000 + 3 * self.params['batch_size'],
                    min_after_dequeue=1000)
            else:
                fnames, inputs, outputs = tf.train.batch(
                    [fname, input, output], batch_size=self.params['batch_size'], num_threads=2,
                    capacity=1000 + 3 * self.params['batch_size'])

        return fnames, inputs, outputs, filename_queue

    def _graph_input_output_from_placeholders(self):
        with tf.name_scope('from_feed'):
            inputs = tf.placeholder('float', [None] + self.process_data.input_shape)
            outputs = tf.placeholder('float', [None] + self.process_data.output_shape)

        return inputs, outputs

    def _get_old_graph_inference(self, graph_type):
        self.logger.info('Graph type: {0}'.format(graph_type))
        sys.path.append(os.path.dirname(self._code_file))
        exec('from {0} import {1} as OldPredictionModel'.format(
            os.path.basename(self._code_file).split('.')[0], 'PredictionModelNN'))

        if graph_type == 'fc':
            return OldPredictionModel._graph_inference_fc
        elif graph_type == 'fc_sub':
            return OldPredictionModel._graph_inference_fc_sub
        else:
            raise Exception('graph_type {0} is not valid'.format(graph_type))

    @staticmethod
    def _graph_inference_fc(name, input, input_shape, output_shape, params, reuse=False, random_seed=None):
        tf.set_random_seed(random_seed)

        with tf.name_scope(name + '_inference'):

            ### define variables
            with tf.variable_scope('inference_vars', reuse=reuse):
                weights = [
                    tf.get_variable('w_hidden_0', [np.prod(input_shape), 40],
                                    initializer=tf.contrib.layers.xavier_initializer()),
                    tf.get_variable('w_hidden_1', [40, 40],
                                    initializer=tf.contrib.layers.xavier_initializer()),
                    tf.get_variable('w_output', [40, np.prod(output_shape)],
                                    initializer=tf.contrib.layers.xavier_initializer()),
                ]
                biases = [
                    tf.get_variable('b_hidden_0', [40], initializer=tf.constant_initializer(0.)),
                    tf.get_variable('b_hidden_1', [40], initializer=tf.constant_initializer(0.)),
                    tf.get_variable('b_output', [np.prod(output_shape)], initializer=tf.constant_initializer(0.)),
                ]

            ### weight decays
            for v in weights + biases:
                tf.add_to_collection('weight_decays', 1.0 * tf.nn.l2_loss(v))

            ### fully connected relus
            layer = tf.reshape(input, [-1, np.prod(input_shape)])
            for i, (weight, bias) in enumerate(zip(weights, biases)):
                with tf.name_scope('hidden_{0}'.format(i)):
                    layer = tf.add(tf.matmul(layer, weight), bias)
                    if i < len(weights) - 1:
                        layer = tf.nn.relu(layer)

            pred_output = tf.reshape(layer, [-1] + output_shape)

        return pred_output

    @staticmethod
    def _graph_inference_fc_sub(name, input, input_shape, output_shape, params, reuse=False, random_seed=None):
        tf.set_random_seed(random_seed)
        assert(output_shape[1] == 2)

        pos_input_0 = tf.expand_dims(input[:, 0, :2], 1)
        input = tf.concat(2, (input[:, :, :2] - tf.tile(pos_input_0, (1, input_shape[0], 1)), input[:, :, 2:]))

        with tf.name_scope(name + '_inference'):

            ### define variables
            with tf.variable_scope('inference_vars', reuse=reuse):
                weights = [
                    tf.get_variable('w_hidden_0', [np.prod(input_shape), 40],
                                    initializer=tf.contrib.layers.xavier_initializer()),
                    tf.get_variable('w_hidden_1', [40, 40],
                                    initializer=tf.contrib.layers.xavier_initializer()),
                    tf.get_variable('w_output', [40, np.prod(output_shape)],
                                    initializer=tf.contrib.layers.xavier_initializer()),
                ]
                biases = [
                    tf.get_variable('b_hidden_0', [40], initializer=tf.constant_initializer(0.)),
                    tf.get_variable('b_hidden_1', [40], initializer=tf.constant_initializer(0.)),
                    tf.get_variable('b_output', [np.prod(output_shape)], initializer=tf.constant_initializer(0.)),
                ]

            ### weight decays
            for v in weights + biases:
                tf.add_to_collection('weight_decays', 1.0 * tf.nn.l2_loss(v))

            ### fully connected relus
            layer = tf.reshape(input, [-1, np.prod(input_shape)])
            for i, (weight, bias) in enumerate(zip(weights, biases)):
                with tf.name_scope('hidden_{0}'.format(i)):
                    layer = tf.add(tf.matmul(layer, weight), bias)
                    if i < len(weights) - 1:
                        layer = tf.nn.relu(layer)

            pred_output = tf.reshape(layer, [-1] + output_shape) + tf.tile(pos_input_0, (1, output_shape[0], 1))

        return pred_output

    def _graph_cost(self, name, pred_output, output):
        with tf.name_scope(name + '_cost_and_err'):
            cost = tf.reduce_mean((pred_output - output) * (pred_output - output))
            weight_decay = self.params['reg'] * tf.add_n(tf.get_collection('weight_decays'))
            total_cost = cost + weight_decay

        return total_cost, cost

    def _graph_optimize(self, cost):
        opt = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'])
        return opt.minimize(cost), opt.compute_gradients(cost)

    def _graph_initialize(self):
        # The op for initializing the variables.
        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.params['device'])
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params['gpu_fraction'])
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                log_device_placement=False,
                                                allow_soft_placement=True))
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Set logs writer into folder /tmp/tensorflow_logs
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(self.exp_folder, graph_def=sess.graph_def)

        saver = tf.train.Saver(max_to_keep=None)

        return sess, coord, threads, saver

    ################
    ### Training ###
    ################

    def _get_epoch(self, fnames_dict, fnames_value):
        for fname in fnames_value:
            fnames_dict[fname] += 1

        epoch = max(fnames_dict.values()) - 1
        return epoch

    def train(self):
        ### process data
        self.process_data.process()

        tf.reset_default_graph()
        ### prepare for trainining
        train_fnames, train_input, train_output, train_queue = self._graph_input_output_from_file(is_train=True,
                                                                                                  shuffle=True)
        train_pred_output = self._graph_inference('train', train_input,
                                                  self.process_data.input_shape, self.process_data.output_shape,
                                                  self.params,
                                                  reuse=False, random_seed=self.params['random_seed'])
        train_total_cost, train_cost = self._graph_cost('train', train_pred_output, train_output)
        train_optimizer, train_grads = self._graph_optimize(train_total_cost)
        train_fnames_dict = defaultdict(int)

        ### create validation
        val_fnames, val_input, val_output, val_queue = self._graph_input_output_from_file(is_train=False,
                                                                                          shuffle=False)
        val_pred_output = self._graph_inference('val', val_input,
                                                self.process_data.input_shape, self.process_data.output_shape,
                                                self.params,
                                                reuse=True, random_seed=self.params['random_seed'])
        val_total_cost, val_cost = self._graph_cost('val', val_pred_output, val_output)
        val_fnames_dict = defaultdict(int)

        ### initialize
        self.sess, coord, threads, self.saver = self._graph_initialize()

        ### create plotter
        plotter = MLPlotter(self.exp_folder,
                            {
                                'total_cost': {
                                    'Title': 'Total Cost',
                                    'subplot': 0,
                                    'color': 'k',
                                    'ylabel': 'MSE + weight decay'
                                },
                                'cost': {
                                    'Title': 'Cost',
                                    'subplot': 1,
                                    'color': 'k',
                                    'ylabel': 'MSE'
                                }
                            })

        ### train
        try:
            train_total_cost_values = []
            train_cost_values = []
            train_epoch = -1
            new_train_epoch = 0
            val_epoch = 0
            step = 0
            epoch_start = time.time()
            while not coord.should_stop() and step < self.params['max_step']:
                if step == 0:
                    for _ in xrange(10): print('')

                ### validation
                if step % self.params['val_step'] == 0:
                    val_total_cost_values = []
                    val_cost_values = []
                    self.logger.debug('\tComputing validation...')
                    while True:
                        val_total_cost_value, val_cost_value, val_fnames_value = \
                            self.sess.run([val_total_cost, val_cost, val_fnames])

                        val_total_cost_values.append(val_total_cost_value)
                        val_cost_values.append(val_cost_value)

                        new_val_epoch = self._get_epoch(val_fnames_dict, val_fnames_value)
                        if new_val_epoch != val_epoch:
                            val_epoch = new_val_epoch
                            break

                    plotter.add_val('total_cost', np.mean(val_total_cost_values))
                    plotter.add_val('cost', np.mean(val_cost_values))
                    plotter.plot()

                    self.logger.debug(
                        'Epoch: {0:04d}, cost val = {1:.3f}, total cost val = {2:.3f} ({3:.2f} s per {4:04d} samples)'.format(
                            train_epoch + 1, np.mean(val_cost_values), np.mean(val_total_cost_values),
                            time.time() - epoch_start,
                            step * self.params['batch_size'] / (train_epoch + 1) if train_epoch >= 0 else 0))
                    epoch_start = time.time()

                ### save
                if step % self.params['save_step'] == 0:
                    self.save()  # save every epoch
                    plotter.save(self.exp_folder)

                train_epoch = new_train_epoch

                ### train
                _, train_total_cost_value, train_cost_value, train_fnames_value = \
                    self.sess.run([train_optimizer, train_total_cost, train_cost, train_fnames])

                train_total_cost_values.append(train_total_cost_value)
                train_cost_values.append(train_cost_value)

                new_train_epoch = self._get_epoch(train_fnames_dict, train_fnames_value)

                # Print an overview fairly often.
                if step % self.params['display_step'] == 0 and step > 0:
                    plotter.add_train('total_cost', step * self.params['batch_size'], np.mean(train_total_cost_values))
                    plotter.add_train('cost', step * self.params['batch_size'], np.mean(train_cost_values))
                    plotter.plot()

                    self.logger.debug(
                        '\tstep {0:05d}, epoch {1:d}, cost: {2:6.2f}, total cost: {3:6.2f}'.format(
                            step,
                            train_epoch,
                            np.mean(train_cost_values),
                            np.mean(train_total_cost_values)))

                    train_total_cost_values = []
                    train_cost_values = []

                step += 1
        except tf.errors.OutOfRangeError:
            self.logger.debug('Done training')
        finally:
            coord.request_stop()

        coord.join(threads)
        self.save()
        plotter.save(self.exp_folder)
        self.sess.close()

        plotter.close()

        ### prepare for feeding
        self._graph_setup_eval()
        self.load()

    ##################
    ### Evaluating ###
    ##################

    def _graph_setup_eval(self):
        self._graph_setup_eval_called = True

        tf.reset_default_graph()
        self.input, self.output = self._graph_input_output_from_placeholders()
        self.output_pred = self._graph_inference('eval', self.input,
                                                 self.process_data.input_shape, self.process_data.output_shape,
                                                 self.params,
                                                 reuse=False)
        self.sess, _, _, self.saver = self._graph_initialize()

    def eval(self, input):
        if not self._graph_setup_eval_called:
            self._graph_setup_eval()

        pred_output = np.expand_dims(self.sess.run(self.output_pred, feed_dict={self.input: input}), 1)
        assert(len(pred_output.shape) == 4)
        assert(pred_output.shape[2] == self.params['H'])
        assert(pred_output.shape[3] == 2)
        return pred_output

    #############################
    ### Load/save/reset/close ###
    #############################

    def load(self):
        if not self._graph_setup_eval_called:
            self._graph_setup_eval()
        self.saver.restore(self.sess, self._model_file)

    def save(self):
        self.saver.save(self.sess, self._model_file)

    def close(self):
        """ Release tf session """
        self.sess.close()
        self.sess = None

