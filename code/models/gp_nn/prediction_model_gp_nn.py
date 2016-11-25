import os, pickle

import GPflow

from models.gp.prediction_model_gp import PredictionModelGP
from models.gp_nn.manifold_gpr import ManifoldGPR

class PredictionModelGPNN(PredictionModelGP):

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
        self.gp_model = ManifoldGPR(inputs, outputs, kern=kernel, graph_type=self.params['graph_type'])

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

