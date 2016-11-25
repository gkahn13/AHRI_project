import os, pickle
import itertools

import numpy as np
import tensorflow as tf
import GPflow

from models.gp.process_data_gp import ProcessDataGP

from general.models.prediction_model import PredictionModel

class PredictionModelGP(PredictionModel):

    def __init__(self, exp_folder, data_folder, params):
        self.process_data = ProcessDataGP(data_folder, params)
        PredictionModel.__init__(self, exp_folder, data_folder, params)

    #############
    ### Files ###
    #############

    @property
    def _this_file(self):
        return os.path.abspath(__file__.replace('.pyc', '.py'))

    @property
    def _model_file(self):
        return os.path.join(self.exp_folder, 'model.pkl')

    #############
    ### Model ###
    #############

    def get_inputs_outputs(self, is_train, reshape):
        train_pedestrians, val_pedestrians = self.process_data.load_pedestrians()
        pedestrians = train_pedestrians if is_train else val_pedestrians

        inputs = np.array(list(itertools.chain(*[p.inputs for p in pedestrians])))
        outputs = np.array(list(itertools.chain(*[p.outputs for p in pedestrians])))
        N = len(inputs)

        if reshape:
            input_dim = np.prod(self.process_data.input_shape)
            output_dim = np.prod(self.process_data.output_shape)
            inputs = inputs.reshape((N, input_dim))
            outputs = outputs.reshape((N, output_dim))

        idxs = np.arange(N)
        np.random.shuffle(idxs)
        inputs = inputs[idxs]
        outputs = outputs[idxs]

        return inputs, outputs

    def create_model(self, inputs, outputs):
        # inputs, outputs = inputs[:100], outputs[:100]

        kernel = GPflow.kernels.Matern52(inputs.shape[1], lengthscales=0.3, ARD=False)
        # A = 0.01 * np.ones((inputs.shape[1], outputs.shape[1]))
        # b = np.zeros(outputs.shape[1])
        # mean_function = GPflow.mean_functions.Linear(A=A, b=b)

        self.gp_model = GPflow.gpr.GPR(inputs, outputs, kern=kernel)
        self.gp_model.likelihood.variance = 0.1

    ################
    ### Training ###
    ################

    def train(self):
        self.logger.info('Creating model...')
        self.process_data.process()
        train_inputs, train_outputs = self.get_inputs_outputs(is_train=True, reshape=True)
        if self.params['training_samples']:
            train_inputs = train_inputs[:self.params['training_samples']]
            train_outputs = train_outputs[:self.params['training_samples']]
        self.create_model(train_inputs, train_outputs)

        self.logger.info('Optimizing model...')
        if self.params['opt'] is None:
            method = 'L-BFGS-B'
        elif self.params['opt'] == 'adam':
            method = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'])
        self.gp_model.optimize(method=method, maxiter=self.params['maxiter'])

        self.logger.info('Evaluating model...')
        train_inputs, train_outputs = self.get_inputs_outputs(is_train=True,  reshape=False)
        val_inputs, val_outputs = self.get_inputs_outputs(is_train=False, reshape=False)

        train_pred_outputs = self.eval(train_inputs).mean(axis=1)
        val_pred_outputs = self.eval(val_inputs).mean(axis=1)

        train_costs = np.linalg.norm(train_pred_outputs - train_outputs, axis=1)
        val_costs = np.linalg.norm(val_pred_outputs - val_outputs, axis=1)

        self.logger.info('\tTrain cost: {0:.3f} +- {1:.3f}'.format(np.mean(train_costs), np.std(train_costs)))
        self.logger.info('\tVal cost: {0:.3f} +- {1:.3f}'.format(np.mean(val_costs), np.std(val_costs)))

        self.save()

    ##################
    ### Evaluating ###
    ##################

    def eval(self, input):
        """
        :return np.array [batch_size, num_sample, H, 2]
        """
        mean, std = self.gp_model.predict_f(np.reshape(input, (len(input), np.prod(self.process_data.input_shape))))

        if self.params['std_max'] is not None:
            std *= self.params['std_max'] / std.max()

        pred_output = np.concatenate(((mean + std)[:, np.newaxis, :], (mean - std)[:, np.newaxis, :]), axis=1)
        pred_output = pred_output.reshape([len(input), 2] + self.process_data.output_shape)

        return pred_output

    #############################
    ### Load/save/reset/close ###
    #############################

    def load(self):
        with open(self._model_file, 'r') as f:
            self.gp_model = pickle.load(f)['model']

    def save(self):
        with open(self._model_file, 'w') as f:
            pickle.dump({'model': self.gp_model}, f)

    def close(self):
        pass
