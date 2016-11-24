import os

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

