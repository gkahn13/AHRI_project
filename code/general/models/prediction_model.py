import os, abc, shutil, pickle

import numpy as np
import matplotlib.pyplot as plt

from general.utils.logger import get_logger

class MLPlotter:
    """
    Plot/save machine learning data
    """
    def __init__(self, title, subplot_dicts, shape=None, figsize=None):
        """
        :param title: title of plot
        :param subplot_dicts: dictionary with dictionaries of form
                              name: {subplot, title, color, ylabel}
        """
        ### setup plot figure
        num_subplots = max(d['subplot'] for d in subplot_dicts.values()) + 1
        if shape is None:
            shape = (1, num_subplots)
        if figsize is None:
            figsize = (30, 7)
        self.f, self.axes = plt.subplots(shape[0], shape[1], figsize=figsize)
        mng = plt.get_current_fig_manager()
        # mng.window.showMinimized()
        plt.suptitle(title)
        plt.show(block=False)
        plt.pause(0.5)

        self.train_lines = {}
        self.val_lines = {}

        axes = self.axes.ravel().tolist()
        for name, d in subplot_dicts.items():
            ax = axes[d['subplot']]
            ax.set_xlabel('Training samples')
            if 'title' in d: ax.set_title(d['title'])
            if 'ylabel' in d: ax.set_ylabel(d['ylabel'])

            self.train_lines[name] = ax.plot([], [], color=d['color'], linestyle='-', label=name)[0]
            self.val_lines[name] = ax.plot([], [], color=d['color'], linestyle='--')[0]

        self.f.canvas.draw()
        plt.pause(0.5)

    def _update_line(self, line, new_x, new_y):
        xdata, ydata = line.get_xdata(), line.get_ydata()

        xdata = np.concatenate((xdata, [new_x]))
        ydata = np.concatenate((ydata, [new_y]))

        line.set_xdata(xdata)
        line.set_ydata(ydata)

        ax = line.get_axes()
        ax.relim()
        ax.autoscale_view()

    def add_train(self, name, training_samples, value):
        self._update_line(self.train_lines[name], training_samples, value)

    def add_val(self, name, value):
        xdata = self.train_lines[name].get_xdata()
        self._update_line(self.val_lines[name], xdata[-1] if len(xdata) > 0 else 0, value)

    def plot(self):
        self.f.canvas.draw()
        plt.pause(0.01)

    def save(self, save_dir, name='training.png'):
        self.f.savefig(os.path.join(save_dir, name))
        with open(os.path.join(save_dir, 'plotter.pkl'), 'w') as f:
            pickle.dump(dict([(k, (v.get_xdata(), v.get_ydata())) for k, v in self.train_lines.items()] +
                             [(k, (v.get_xdata(), v.get_ydata())) for k, v in self.val_lines.items()]),
                        f)



    def close(self):
        plt.close(self.f)

class PredictionModel(object):

    def __init__(self, exp_folder, data_folder, params):
        self.exp_folder = exp_folder
        if not os.path.exists(self.exp_folder):
            os.makedirs(self.exp_folder)
        self.data_folder = data_folder
        assert(os.path.exists(self.data_folder))
        self.params = params

        self.logger = get_logger(self.__class__.__name__, 'debug', os.path.join(self.exp_folder, 'debug.txt'))

        ### copy file model file over
        code_file_exists = os.path.exists(self._code_file)
        if code_file_exists:
            self.logger.info('Creating OLD graph')
        else:
            self.logger.info('Creating NEW graph')
            shutil.copyfile(self._this_file, self._code_file)

    #############
    ### Files ###
    #############

    @abc.abstractproperty
    @property
    def _this_file(self):
        raise NotImplementedError('Implement in subclass')

    @property
    def _code_file(self):
        return os.path.join(os.path.abspath(self.exp_folder),
                            'prediction_model_{0}.py'.format(os.path.basename(self.exp_folder)))

    ################
    ### Training ###
    ################

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError('Implement in subclass')

    ##################
    ### Evaluating ###
    ##################

    @abc.abstractmethod
    def eval(self, input):
        """
        :return np.array [batch_size, num_sample, H, 2]
        """
        raise NotImplementedError('Implement in subclass')

    #############################
    ### Load/save/reset/close ###
    #############################

    @abc.abstractmethod
    def load(self):
        raise NotImplementedError('Implement in subclass')

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError('Implement in subclass')

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError('Implement in subclass')
