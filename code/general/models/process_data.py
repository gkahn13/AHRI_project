import abc
import os
import pickle

import numpy as np
import imageio

class Pedestrian(object):
    def __init__(self, id, frames, positions, velocities, groups, image_reader):
        self.id = id
        self.frames = np.array(frames)
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.groups = np.array(groups)
        self.image_reader = image_reader

        assert(len(self.frames) == len(self.positions))
        assert(len(self.positions) == len(self.velocities))

    @property
    def images(self):
        return [np.array(self.image_reader.get_data(frame)) for frame in self.frames]

    def __len__(self):
        return len(self.positions)

class ProcessData(object):
    """
    Process data for ETHZ BIWI Walking Pedestrians Dataset
    """

    def __init__(self, folder, params):
        self.folder = folder
        self.params = params

        assert(os.path.exists(self.annotation_file))
        assert(os.path.exists(self.video_file))
        assert(os.path.exists(self.groups_file))

    #############
    ### Files ###
    #############

    @property
    def annotation_file(self):
        return os.path.join(self.folder, 'obsmat.txt')

    @property
    def video_file(self):
        for fname in os.listdir(self.folder):
            if 'avi' in fname:
                break
        else:
            raise Exception('Could not find video file')

        return os.path.join(self.folder, fname)

    @property
    def groups_file(self):
        return os.path.join(self.folder, 'groups.txt')

    @property
    def data_folder(self):
        data_folder = os.path.join(self.folder, '{0}_tfrecords'.format(self.params['feature_type']))
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        return data_folder

    def train_data_file(self, i):
        return os.path.join(self.data_folder, 'train_{0:04d}.tfrecords'.format(i))

    def val_data_file(self, i):
        return os.path.join(self.data_folder, 'val_{0:04d}.tfrecords'.format(i))

    @property
    def all_train_data_files(self):
        i = 0
        data_files = []

        while os.path.exists(self.train_data_file(i)):
            data_files.append(self.train_data_file(i))
            i += 1

        return data_files

    @property
    def all_val_data_files(self):
        i = 0
        data_files = []

        while os.path.exists(self.val_data_file(i)):
            data_files.append(self.val_data_file(i))
            i += 1

        return data_files

    @property
    def all_data_files(self):
        return self.all_train_data_files + self.all_val_data_files

    @property
    def reshape_data_file(self):
        return os.path.join(self.data_folder, 'reshape.pkl')

    @property
    def input_shape(self):
        with open(self.reshape_data_file, 'r') as f:
            d = pickle.load(f)
        return d['input']

    @property
    def output_shape(self):
        with open(self.reshape_data_file, 'r') as f:
            d = pickle.load(f)
        return d['output']

    ###############
    ### Parsing ###
    ###############

    def parse(self):
        ### read annotation file
        annotation = np.loadtxt(self.annotation_file)
        frames = annotation[:, 0].astype(int)
        ids = annotation[:, 1].astype(int)
        positions = annotation[:, (2, 4)]
        velocities = annotation[:, (5, 7)]

        ### video file reader
        image_reader = imageio.get_reader(self.video_file)

        ### read groups
        groups = [np.array(map(int, line.split())) for line in open(self.groups_file)]

        pedestrians = []
        for id in np.unique(ids):
            mask = (ids == id)

            group = [g for g in groups if id in g]
            assert(len(group) <= 1)
            if len(group) == 1:
                group = [i for i in group[0] if i != id]

            pedestrians.append(Pedestrian(id,
                                          frames[mask],
                                          positions[mask],
                                          velocities[mask],
                                          group,
                                          image_reader))

        return pedestrians

    #################
    ### Save data ###
    #################

    @abc.abstractmethod
    def save_data(self, pedestrians):
        raise NotImplementedError('Implement in subclass')

    ###############
    ### Process ###
    ###############

    def process(self):
        pedestrians = self.parse()
        self.save_data(pedestrians)
