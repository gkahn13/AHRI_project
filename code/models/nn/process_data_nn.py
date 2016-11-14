import os
import random
import pickle

import numpy as np
import tensorflow as tf

from general.models.process_data import ProcessData

from config import params

class ProcessDataNN(ProcessData):

    def __init__(self, folder):
        ProcessData.__init__(self, folder)

    #################
    ### Save data ###
    #################

    def save_data(self, pedestrians):
        ### only keep long enough trajectories
        pedestrians = [ped for ped in pedestrians if len(ped) >= params['K'] + params['H']]

        ### split into train/val by trajectory
        random.shuffle(pedestrians)
        num_val = int(params['val_pct'] * len(pedestrians))
        val_pedestrians = pedestrians[:num_val]
        train_pedestrians = pedestrians[num_val:]

        ### select feature method
        if params['feature_type'] == 'position':
            features_method = self.position_features
            reshape_data = self.position_reshape_data
        else:
            raise Exception('Feature type {0} not valid'.format(params['feature_type']))

        ### delete previous features
        for data_file in self.all_data_files:
            os.remove(data_file)

        ### save features
        for fname, features in features_method(self.train_data_file, train_pedestrians):
            self.save_tfrecord(fname, features)
        for fname, features in features_method(self.val_data_file, val_pedestrians):
            self.save_tfrecord(fname, features)

        ### save reshape data
        with open(self.reshape_data_file, 'w') as f:
            pickle.dump(reshape_data, f)

    def save_tfrecord(self, fname, features):
        writer = tf.python_io.TFRecordWriter(fname)

        for feature in features:
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()

    ################
    ### Features ###
    ################

    @staticmethod
    def _floatlist_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _int64list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def position_features(self, data_file_func, pedestrians):
        feature_num = 0
        features = []
        for ped in pedestrians:
            for start in xrange(params['K'], len(ped) - (params['K'] + params['H'])):
                fname = data_file_func(feature_num)
                feature_fname = os.path.splitext(os.path.basename(fname))[0]
                input = ped.positions[start - params['K'] + 1:start + 1]
                output = ped.positions[start + 1:start + params['H'] + 1]

                features.append({
                    'fname': ProcessDataNN._bytes_feature(feature_fname),
                    'input': ProcessDataNN._floatlist_feature(input.ravel().tolist()),
                    'output': ProcessDataNN._floatlist_feature(output.ravel().tolist()),
                })

                if len(features) >= params['features_per_file']:
                    yield fname, features
                    features = []
                    feature_num += 1

        if len(features) >= 0:
            yield fname, features

    @property
    def position_reshape_data(self):
        return {
            'input': [params['K'], 2],
            'output': [params['H'], 2]
        }



