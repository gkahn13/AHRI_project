import os, pickle

import GPflow

from models.gp.prediction_model_gp import PredictionModelGP
from models.gp_nn.process_data_gp_nn import ProcessDataGPNN
from models.gp_nn.manifold_gpr import ManifoldGPR

from general.models.prediction_model import PredictionModel

class PredictionModelGPNN(PredictionModelGP):

    def __init__(self, exp_folder, data_folder, params):
        self.process_data = ProcessDataGPNN(data_folder, params)
        PredictionModel.__init__(self, exp_folder, data_folder, params)

    #############
    ### Files ###
    #############

    @property
    def _this_file(self):
        return os.path.abspath(__file__.replace('.pyc', '.py'))

    #############
    ### Model ###
    #############

    def create_model(self, inputs, outputs):
        kernel = GPflow.kernels.Matern52(self.params['kernel_size'], lengthscales=0.3, ARD=False)
        # A = 0.01 * np.ones((self.params['kernel_size'], outputs.shape[1]))
        # b = np.zeros(outputs.shape[1])
        # mean_function = GPflow.mean_functions.Linear(A=A, b=b)

        self.gp_model = ManifoldGPR(inputs, outputs, kern=kernel,
                                    graph_type=self.params['graph_type'])

    #############################
    ### Load/save/reset/close ###
    #############################

    def load(self):
        with open(self._model_file, 'r') as f:
            d = pickle.load(f)
        self.create_model(d['inputs'], d['outputs'])
        self.gp_model.set_parameter_dict(d['parameter_dict'])

    def save(self):
        with open(self._model_file, 'w') as f:
            pickle.dump({'parameter_dict': self.gp_model.get_parameter_dict(),
                         'inputs': self.gp_model.X.value,
                         'outputs': self.gp_model.Y.value},
                        f)

