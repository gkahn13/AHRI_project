import os, pickle, itertools

import numpy as np

from general.models.prediction_model import PredictionModel
from models.lr.process_data_lr import ProcessDataLR

class PredictionModelLR(PredictionModel):

    def __init__(self, exp_folder, data_folder, params):
        self.process_data = ProcessDataLR(data_folder, params)
        PredictionModel.__init__(self, exp_folder, data_folder, params)

        self.w = None

    #############
    ### Files ###
    #############

    @property
    def _this_file(self):
        return os.path.abspath(__file__.replace('.pyc', '.py'))

    @property
    def _model_file(self):
        return os.path.join(self.exp_folder, 'model.pkl')

    ################
    ### Training ###
    ################

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

    def train(self):
        self.logger.info('Processing data...')
        self.process_data.process()
        X, Y = self.get_inputs_outputs(is_train=True, reshape=True)

        self.logger.info('Optimizing model...')
        self.w = np.linalg.lstsq(X, Y)[0]

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
        input = np.reshape(input, (len(input), np.prod(self.process_data.input_shape)))

        pred_output = input.dot(self.w)

        pred_output = np.reshape(pred_output, [len(pred_output)] + self.process_data.output_shape)
        pred_output = np.expand_dims(pred_output, 1)

        return pred_output

    #############################
    ### Load/save/reset/close ###
    #############################

    def load(self):
        with open(self._model_file, 'r') as f:
            self.w = pickle.load(f)['w']

    def save(self):
        with open(self._model_file, 'w') as f:
            pickle.dump({'w': self.w}, f)

    def close(self):
        pass