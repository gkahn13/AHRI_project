import os
import itertools

import numpy as np
import GPflow

from models.gp.process_data_gp import ProcessDataGP

from general.models.prediction_model import PredictionModel, MLPlotter

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

    ################
    ### Training ###
    ################

    def train(self):
        ### load data/model
        self.process_data.process()
        self.load()
        train_inputs, train_outputs = self.get_inputs_outputs(True)
        val_inputs, val_outputs = self.get_inputs_outputs(False)


        train_pred_outputs, _ = self.gp_model.predict_f(train_inputs)
        val_pred_outputs, _ = self.gp_model.predict_f(val_inputs)

        train_cost = np.linalg.norm(train_pred_outputs - train_outputs, axis=1).mean()
        val_cost = np.linalg.norm(val_pred_outputs - val_outputs, axis=1).mean()

        import IPython; IPython.embed()

    ##################
    ### Evaluating ###
    ##################

    def eval(self, input):
        """
        :return np.array [batch_size, num_sample, H, 2]
        """
        raise NotImplementedError('Implement in subclass')

    #############################
    ### Load/save/reset/close ###
    #############################

    def get_inputs_outputs(self, is_train):
        train_pedestrians, val_pedestrians = self.process_data.load_pedestrians()
        pedestrians = train_pedestrians if is_train else val_pedestrians

        inputs = np.array(list(itertools.chain(*[p.inputs for p in pedestrians])))
        outputs = np.array(list(itertools.chain(*[p.outputs for p in pedestrians])))

        N = len(inputs)
        input_dim = np.prod(self.process_data.input_shape)
        output_dim = np.prod(self.process_data.output_shape)
        inputs = inputs.reshape((N, input_dim))
        outputs = outputs.reshape((N, output_dim))

        return inputs, outputs

    def load(self):
        ### load data
        train_inputs, train_outputs = self.get_inputs_outputs(True)

        ### create GP model
        kernel = GPflow.kernels.Matern52(train_inputs.shape[1], lengthscales=0.3)
        self.gp_model = GPflow.gpr.GPR(train_inputs, train_outputs, kern=kernel)
        self.gp_model.likelihood.variance = 0.1

    def save(self):
        raise NotImplementedError('Implement in subclass')

    def close(self):
        raise NotImplementedError('Implement in subclass')
