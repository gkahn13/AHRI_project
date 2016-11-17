import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class AnalyzeModel(object):
    """
    Open model, evaluate it, close it
    """
    def __init__(self, prediction_model):
        self.prediction_model = prediction_model
        self.prediction_model.load()
        self.train_pedestrians, self.val_pedestrians = self.prediction_model.process_data.load_pedestrians()

        self.train_inputs, self.train_outputs, self.train_pred_outputs = \
            AnalyzeModel.get_inputs_outputs_preds(self.prediction_model, self.train_pedestrians)
        self.val_inputs, self.val_outputs, self.val_pred_outputs = \
            AnalyzeModel.get_inputs_outputs_preds(self.prediction_model, self.train_pedestrians)

        self.prediction_model.close()

    ##################
    ### Processing ###
    ##################

    @property
    def name(self):
        return self.prediction_model.exp_folder.split('exps/')[1]

    @staticmethod
    def get_inputs_outputs_preds(prediction_model, pedestrians):
        inputs = np.array(list(itertools.chain(*[p.inputs for p in pedestrians])))
        pred_outputs = prediction_model.eval(inputs)
        outputs = np.array(list(itertools.chain(*[p.outputs for p in pedestrians])))

        return inputs, outputs, pred_outputs


class AnalyzeModels(object):

    def __init__(self, prediction_models):
        self.analyze_models = [AnalyzeModel(pm) for pm in prediction_models]

    ###############
    ### Metrics ###
    ###############

    @staticmethod
    def average_displacement_metric(outputs, pred_outputs, average_samples):
        if average_samples:
            pred_outputs_avg = pred_outputs.mean(axis=1)
            return np.linalg.norm(outputs - pred_outputs_avg, axis=2).sum(axis=1) # norm each timestep, sum over timesteps
        else:
            num_samples = pred_outputs.shape[1]
            outputs_tiled = np.tile(np.expand_dims(outputs, 1), (1, num_samples, 1, 1))
            return np.linalg.norm(outputs_tiled - pred_outputs, axis=3).sum(axis=2).mean(axis=1) # norm each timestep, sum over timesteps, mean over samples

    @staticmethod
    def final_displacement_metric(outputs, pred_outputs, average_samples):
        if average_samples:
            pred_outputs_avg = pred_outputs.mean(axis=1)
            return np.linalg.norm(outputs[:, -1, :] - pred_outputs_avg[:, -1, :], axis=1)
        else:
            num_samples = pred_outputs.shape[1]
            outputs_tiled = np.tile(np.expand_dims(outputs, 1), (1, num_samples, 1, 1))
            return np.linalg.norm(outputs_tiled[:, :, -1, :] - pred_outputs[:, :, -1, :], axis=2).mean(axis=1)

    # @staticmethod
    # def nonlinear_displacement_metric(outputs, pred_outputs, average_samples):
    #     outputs_filt = []
    #     pred_outputs_filt = []
    #     for output, pred_output in zip(outputs, pred_outputs):
    #         deriv1 = np.array([output[i+1] - pred_output[i] for i in xrange(len(output)-1)])
    #         deriv2 = 0.5*np.array([deriv1[i+1] - deriv1[i] for i in xrange(len(deriv1)-1)])
    #         if np.linalg.norm(deriv2) / len(deriv2) > 1.0: # TODO thresh
    #             outputs_filt.append(deriv1)
    #             pred_outputs_filt.append(deriv2)
    #
    #     N = len(outputs_filt)
    #     return np.linalg.norm((outputs_filt - pred_outputs_filt).reshape((N, -1)), axis=1)

    ################
    ### Plotting ###
    ################

    def plot_cost_histogram(self, title, displacement_metric, average_samples, num_bins=11):
        f, ax = plt.subplots(1, 1)

        ### get prediction distance error
        l2_dists = []
        labels = []
        for am in self.analyze_models:
            inputs, outputs, pred_outputs = am.train_inputs, am.train_outputs, am.train_pred_outputs
            l2_dists.append(displacement_metric(outputs, pred_outputs, average_samples))
            labels.append(am.name)

        ### plot histogram
        bins = np.linspace(np.concatenate(l2_dists).min(), np.concatenate(l2_dists).max(), num_bins)
        hists = []
        for l2_dist in l2_dists:
            N = len(l2_dist)
            hists.append(np.histogram(l2_dist, bins, weights=(1. / N) * np.ones(N))[0])
        bar_width = 0.8 * (bins[1] - bins[0]) / len(hists)
        for i, (pcts, label) in enumerate(zip(hists, labels)):
            ax.bar(bins[:-1] + i * bar_width, pcts, bar_width,
                   color=cm.jet(i / float(len(hists))), label=label)

        ### formatting
        ax.set_title(title)
        ax.set_xlabel('L2 distance (meters)')
        ax.set_ylabel('Fraction')
        ax.legend()
        ax.set_xticks(bins)

        plt.show(block=False)
        plt.pause(0.05)

    def run(self):
        self.plot_cost_histogram('Average displacement', AnalyzeModels.average_displacement_metric, True)
        self.plot_cost_histogram('Final displacement', AnalyzeModels.final_displacement_metric, True)

        raw_input('Press enter to exit')



