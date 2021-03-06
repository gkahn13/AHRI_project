import itertools, os

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 32})
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

from matplotlib import cm
import general.utils.colormaps as cmaps

class AnalyzeModel(object):
    """
    Open model, evaluate it, close it
    """
    def __init__(self, prediction_model):
        self.params = prediction_model.params
        self.prediction_model = prediction_model
        self.prediction_model.load()
        self.train_pedestrians, self.val_pedestrians = self.prediction_model.process_data.load_pedestrians()

        self.train_inputs, self.train_outputs, self.train_pred_outputs = \
            AnalyzeModel.get_inputs_outputs_preds(self.prediction_model, self.train_pedestrians)
        self.val_inputs, self.val_outputs, self.val_pred_outputs = \
            AnalyzeModel.get_inputs_outputs_preds(self.prediction_model, self.val_pedestrians)

        self.prediction_model.close()

    ##################
    ### Processing ###
    ##################

    @property
    def name(self):
        if 'name' in self.prediction_model.params:
            return self.prediction_model.params['name']
        else:
            return self.prediction_model.exp_folder.split('exps/')[1]

    @staticmethod
    def get_inputs_outputs_preds(prediction_model, pedestrians):
        inputs = np.array(list(itertools.chain(*[p.inputs for p in pedestrians])))
        outputs = np.array(list(itertools.chain(*[p.outputs for p in pedestrians])))
        pred_outputs = prediction_model.eval(inputs)

        ped_idxs = np.hstack([[i] * len(p.inputs) for i, p in enumerate(pedestrians)])

        for i, ped in enumerate(pedestrians):
            mask = (ped_idxs == i)
            if np.sum(mask) == 0:
                continue

            ped.inputs = inputs[mask]
            ped.outputs = outputs[mask]
            ped.pred_outputs = pred_outputs[mask]

        return inputs, outputs, pred_outputs

    def get_pedestrian(self, id):
        for p in self.train_pedestrians + self.val_pedestrians:
            if p.id == id:
                return p

        return None

    @property
    def max_pedestrian_id(self):
        return max([p.id for p in self.train_pedestrians + self.val_pedestrians])


class AnalyzeModels(object):

    def __init__(self, prediction_models):
        self.analyze_models = [AnalyzeModel(pm) for pm in prediction_models]

    ###############
    ### Metrics ###
    ###############

    @staticmethod
    def average_displacement_metric(outputs, pred_outputs, average_samples):
        pred_outputs_avg = pred_outputs.mean(axis=1)
        num_samples = pred_outputs.shape[1]

        if average_samples:
            dists_mean = np.linalg.norm(outputs - pred_outputs_avg, axis=2).mean(axis=1) # norm each timestep, mean over timesteps
        else:
            outputs_tiled = np.tile(np.expand_dims(outputs, 1), (1, num_samples, 1, 1))
            dists_mean = np.linalg.norm(outputs_tiled - pred_outputs, axis=3).mean(axis=2).mean(axis=1) # norm each timestep, mean over timesteps, mean over samples

        ### std is distance from mean prediction (so no ground truth involved)
        # pred_outputs_avg_tiled = np.tile(np.expand_dims(pred_outputs.mean(axis=1), 1), (1, num_samples, 1, 1))
        # dists_std = np.linalg.norm(pred_outputs - pred_outputs_avg_tiled, axis=3).sum(axis=2).std(axis=1)

        dists_std = pred_outputs.std(axis=1).sum(axis=2).mean(axis=1)

        return dists_mean, dists_std

    @staticmethod
    def final_displacement_metric(outputs, pred_outputs, average_samples):
        raise Exception('Not correct now')
        if average_samples:
            pred_outputs_avg = pred_outputs.mean(axis=1)
            dists_mean = np.linalg.norm(outputs[:, -1, :] - pred_outputs_avg[:, -1, :], axis=1)
            dists_std = np.zeros(dists_mean.shape)
        else:
            num_samples = pred_outputs.shape[1]
            outputs_tiled = np.tile(np.expand_dims(outputs, 1), (1, num_samples, 1, 1))
            dists_mean = np.linalg.norm(outputs_tiled[:, :, -1, :] - pred_outputs[:, :, -1, :], axis=2).mean(axis=1)
            dists_std = np.linalg.norm(outputs_tiled[:, :, -1, :] - pred_outputs[:, :, -1, :], axis=2).std(axis=1)

        return dists_mean, dists_std

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

    def plot_predictions_on_images(self):
        ### must have same K and H
        K = self.analyze_models[0].params['K']
        H = self.analyze_models[0].params['H']
        assert(np.all([am.params['K'] == K for am in self.analyze_models]))
        assert(np.all([am.params['H'] == H for am in self.analyze_models]))

        labels = [am.name for am in self.analyze_models]
        homography = self.analyze_models[0].prediction_model.process_data.homography

        f, ax = plt.subplots(1, 1)
        ax_im = None

        def perform_homography(hom, x):
            pred_output = np.hstack((x, np.ones((len(x), 1))))
            coords = np.linalg.solve(hom, pred_output.T)
            coords /= coords[-1]
            coords = coords[:-1].T
            return coords

        plot_dict = {}
        def plot_memoized(key, ax, x, y, **kwargs):
            if key in plot_dict.keys():
                plot_dict[key].set_xdata(x)
                plot_dict[key].set_ydata(y)
            else:
                plot_dict[key] = ax.plot(x, y, **kwargs)[0]
                ax.legend()

        ### choose which ids to look at
        all_ids = range(max([am.max_pedestrian_id for am in self.analyze_models]))
        train_ids = np.array([sorted([p.id for p in am.train_pedestrians]) for am in self.analyze_models])
        val_ids = np.array([sorted([p.id for p in am.val_pedestrians]) for am in self.analyze_models])
        assert(train_ids.std(axis=0).max() == 0)
        assert(val_ids.std(axis=0).max() == 0)
        ids = val_ids[0]

        for id in ids:
            peds = [am.get_pedestrian(id) for am in self.analyze_models]
            if np.any([p is None or not hasattr(p, 'pred_outputs') for p in peds]):
                continue

            images = peds[0].images
            N = len(peds[0].pred_outputs)
            for n in xrange(N):
                im = images[n+K]
                if ax_im is None:
                    ax_im = ax.imshow(im)
                    plt.show(block=False)
                else:
                    ax_im.set_array(im)
                    f.canvas.draw()
                plt.pause(0.01)

                output = np.vstack((np.expand_dims(peds[0].inputs[n][-1,:2], 0), peds[0].outputs[n])) # TODO assumes first 2 inputs position
                coords = perform_homography(homography, output)
                plot_memoized('gt', ax, coords[:, 1], coords[:, 0],
                              color='k', label='Ground Truth', marker='^', markersize=10, linewidth=3)

                for i, (input, pred_output) in enumerate(zip([p.inputs[n] for p in peds],
                                                             [p.pred_outputs[n] for p in peds])):
                    color = cmaps.magma((i + 1) / float(len(self.analyze_models) + 1))

                    for j, pred_output_j in enumerate(pred_output):
                        pred_output_j = np.vstack((np.expand_dims(input[-1,:2], 0), pred_output_j))
                        coords = perform_homography(homography, pred_output_j)
                        plot_memoized('sample_{0}_{1}'.format(i, j), ax, coords[:, 1], coords[:, 0],
                                      color=color, linestyle='--', linewidth=2)

                    pred_output_mean = pred_output.mean(axis=0)
                    pred_output_mean = np.vstack((np.expand_dims(input[-1,:2], 0), pred_output_mean))
                    coords = perform_homography(homography, pred_output_mean)
                    plot_memoized('mean_{0}'.format(i), ax, coords[:, 1], coords[:, 0],
                                  color=color, label=labels[i], marker='^', markersize=10, linewidth=3)


                f.canvas.draw()
                read = raw_input('Pedestrian {0}: {1}'.format(id, n))
                if read == 'b':
                    break

    def plot_accuracy_curve(self, title, displacement_metric, average_samples, save_folder=None):
        markers = itertools.cycle(['^', 's', 'h', '*', 'd'])

        fig = plt.figure(figsize=(30, 10))
        gs = gridspec.GridSpec(2, len(self.analyze_models), height_ratios=[3, 1])
        ax_mean = fig.add_subplot(gs[0, :])
        ax_stds = [fig.add_subplot(gs[1, i]) for i in xrange(len(self.analyze_models))]
        # ax_mean = plt.subplot2grid((2, len(self.analyze_models)), (0, 0), colspan=len(self.analyze_models))
        # ax_stds = [plt.subplot2grid((2, len(self.analyze_models)), (1, i)) for i in xrange(len(self.analyze_models))]

        ### get prediction distance error
        dists_means = []
        dists_stds = []
        labels = []
        for am in self.analyze_models:
            # inputs, outputs, pred_outputs = am.train_inputs, am.train_outputs, am.train_pred_outputs
            inputs, outputs, pred_outputs = am.val_inputs, am.val_outputs, am.val_pred_outputs
            dists_mean, dists_std = displacement_metric(outputs, pred_outputs, average_samples)
            dists_means.append(dists_mean)
            dists_stds.append(dists_std)
            labels.append(am.name)

        for i, (label, dists_mean, dists_std) in enumerate(zip(labels, dists_means, dists_stds)):
            sort_idxs = np.argsort(dists_mean)

            xs = dists_mean[sort_idxs]
            ys = np.linspace(0, 100, len(xs))
            xerrs = dists_std[sort_idxs]
            color = cmaps.plasma(i / float(len(labels)))
            marker = next(markers)

            ax_mean.plot(xs, ys, label=label, color=color, linewidth=3,
                         marker=marker, markevery=int(0.1*len(xs)), markersize=10., markeredgewidth=2.,
                         markeredgecolor=color, markerfacecolor='w')
            # ax_mean.fill_betweenx(ys, xs - xerrs, xs + xerrs, color=color, alpha=0.5)

            ax_stds[i].plot(xs, xerrs, 'o', color=color, markeredgecolor=color, markersize=2)

            rho = np.corrcoef(xs, xerrs)[0, 1]
            ax_stds[i].text(0.7, 0.85, r"$\rho = {0:.3f}$".format(rho), ha='center', va='center', transform=ax_stds[i].transAxes)

        ### axes
        max_xlim = max([ax.get_xlim()[1] for ax in ax_stds])
        # max_ylim = max([ax.get_ylim()[1] for ax in ax_stds])
        for ax in ax_stds:
            ax.set_xlim((0, max_xlim))
            # ax.set_ylim((0, max_ylim))
            ax.set_ylim((0, ax.get_ylim()[1]))

        ### formatting
        # if average_samples:
        #     title += ' (averaging samples)'
        # else:
        #     title += ' (NOT averaging samples)'
        if title is None:
            title = 'Average displacement'
        fig.suptitle(title)

        for ax in [ax_mean] + ax_stds:
            ax.set_xlabel('Threshold (meters)')
            ax.legend()
        for ax in ax_stds:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
        ax_mean.yaxis.set_major_locator(MaxNLocator(prune='lower'))


        ax_mean.set_ylabel('% correctly\npredicted trajectories')
        ax_stds[0].set_ylabel('std')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)

        if save_folder is not None:
            fig.savefig(os.path.join(save_folder, title.lower().replace(' ', '_') + '_avg_disp.png'), dpi=200)

        plt.close(fig)

    def plot_cost_histogram(self, title, displacement_metric, average_samples, num_bins=21):
        f, axes = plt.subplots(2, 1, sharex=True)
        ax_hist = axes[0]
        ax_hist_std = axes[1]

        ### get prediction distance error
        dists_means = []
        dists_stds = []
        labels = []
        for am in self.analyze_models:
            inputs, outputs, pred_outputs = am.train_inputs, am.train_outputs, am.train_pred_outputs
            dists_mean, dists_std = displacement_metric(outputs, pred_outputs, average_samples)
            dists_means.append(dists_mean)
            dists_stds.append(dists_std)
            labels.append(am.name)

        ### create histogram
        bins = np.linspace(np.concatenate(dists_means).min(), np.concatenate(dists_means).max(), num_bins)
        hists = []
        hists_std = []
        for dists_mean, dists_std in zip(dists_means, dists_stds):
            N = len(dists_mean)
            hist, bin_edges = np.histogram(dists_mean, bins, weights=(1. / N) * np.ones(N))

            hist_idxs = np.digitize(dists_mean, bin_edges) - 1
            hist_std = [dists_std[hist_idxs == i].mean() for i in xrange(len(hist))]

            hists.append(hist)
            hists_std.append(hist_std)


        ### plot histogram
        bar_width = 0.8 * (bins[1] - bins[0]) / len(hists)
        for i, (pcts, hist, hist_std, label) in enumerate(zip(hists, hists, hists_std, labels)):
            color = cm.jet(i / float(len(hists)))
            full_label = '{0}, mean: {1:.3f}'.format(label, np.mean(hist))

            ax_hist.bar(bins[:-1] + i * bar_width, pcts, width=bar_width, color=color, label=full_label)
            ax_hist_std.plot(bins[:-1], hist_std, color=color, label=full_label)

        ### formatting
        if average_samples:
            title += ' (averaging samples)'
        else:
            title += ' (NOT averaging samples)'
        f.suptitle(title)

        ax_hist.set_ylabel('Fraction')

        ax_hist_std.set_ylabel('Std (meters)')

        for ax in axes:
            ax.legend()
            ax.set_xticks(bins)
            ax.set_xlabel('L2 distance (meters)')


        plt.show(block=False)
        plt.pause(0.05)

    def run(self, title=None, save_folder=None):
        # self.plot_cost_histogram('Average displacement', AnalyzeModels.average_displacement_metric, True)
        # self.plot_cost_histogram('Final displacement', AnalyzeModels.final_displacement_metric, True)
        # self.plot_cost_histogram('Average displacement', AnalyzeModels.average_displacement_metric, False)
        # self.plot_cost_histogram('Final displacement', AnalyzeModels.final_displacement_metric, False)

        self.plot_accuracy_curve(title, AnalyzeModels.average_displacement_metric, True, save_folder=save_folder)
        # self.plot_accuracy_curve('Average displacement', AnalyzeModels.average_displacement_metric, False)

        # self.plot_predictions_on_images()

        # raw_input('Press enter to exit')



