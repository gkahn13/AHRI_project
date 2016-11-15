import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class AnalyzeModel(object):

    def __init__(self, prediction_model):
        self.prediction_model = prediction_model
        self.prediction_model.load()
        self.train_pedestrians, self.val_pedestrians = self.prediction_model.process_data.load_pedestrians()

    ##################
    ### Processing ###
    ##################

    @property
    def name(self):
        return self.prediction_model.exp_folder.split('exps/')[1]

    def get_inputs_outputs_preds(self, pedestrians):
        inputs = np.array(list(itertools.chain(*[p.inputs for p in pedestrians])))
        outputs = np.array(list(itertools.chain(*[p.outputs for p in pedestrians])))
        pred_outputs = self.prediction_model.eval(inputs)[0]

        return inputs, outputs, pred_outputs


class AnalyzeModels(object):

    def __init__(self, prediction_models):
        self.analyze_models = [AnalyzeModel(pm) for pm in prediction_models]

    ################
    ### Plotting ###
    ################

    def plot_cost_histogram(self, num_bins=21):
        f, ax = plt.subplots(1, 1)

        ### get prediction distance error
        l2_dists = []
        labels = []
        for am in self.analyze_models:
            inputs, outputs, pred_outputs = am.get_inputs_outputs_preds(am.train_pedestrians)
            N = len(inputs)
            l2_dists.append(np.linalg.norm((outputs - pred_outputs).reshape((N, -1)), axis=1))
            labels.append(am.name)

        ### plot histogram
        bins = np.linspace(np.min(l2_dists), np.max(l2_dists), num_bins)
        hists = []
        for l2_dist in l2_dists:
            N = len(l2_dist)
            hists.append(np.histogram(l2_dist, bins, weights=(1. / N) * np.ones(N))[0])
        bar_width = 0.8 / (len(hists) * num_bins)
        for i, (pcts, label) in enumerate(zip(hists, labels)):
            ax.bar(bins[:-1] + i * bar_width, pcts, bar_width,
                   color=cm.jet(i / float(len(hists))), label=label)

        ### formatting
        ax.set_title('L2 distance histogram')
        ax.set_xlabel('L2 distance (meters)')
        ax.set_ylabel('Fraction')
        ax.legend()
        ax.set_xticks(bins)

        plt.show(block=False)
        plt.pause(0.05)

        import IPython; IPython.embed()

    def run(self):
        self.plot_cost_histogram()



