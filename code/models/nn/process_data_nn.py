import os

import tensorflow as tf

from general.models.process_data import ProcessData

from config import params

class ProcessDataNN(ProcessData):

    def __init__(self, folder):
        ProcessData.__init__(self, folder)

        self.K = params['K']
        self.H = params['H']
        self.feature_type = params['feature_type']

        for k, v in params['process_data'].items():
            setattr(self, k, v)

    #############
    ### Files ###
    #############

    def data_file(self, i):
        data_folder = os.path.join(self.folder, '{0}_tfrecords'.format(self.feature_type))
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        return os.path.join(data_folder, '{0:04d}.tfrecords'.format(i))

    #################
    ### Save data ###
    #################

    def save_data(self, pedestrians):
        ### only keep long enough trajectories
        pedestrians = [ped for ped in pedestrians if len(ped) >= self.K + self.H]

        ### select feature method
        if self.feature_type == 'position':
            features_method = self.position_features
        else:
            raise Exception('Feature type {0} not valid'.format(self.feature_type))

        ### delete previous features
        i = 0
        while os.path.exists(self.data_file(i)):
            os.remove(self.data_file(i))
            i += 1

        ### save features
        for fname, features in features_method(pedestrians):
            self.save_tfrecord(fname, features)

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

    def position_features(self, pedestrians):
        feature_num = 0
        features = []
        for ped in pedestrians:
            for start in xrange(self.K, len(ped) - (self.K + self.H)):
                fname = self.data_file(feature_num)
                feature_fname = os.path.splitext(os.path.basename(fname))[0]
                input = ped.positions[start - self.K + 1:start + 1]
                output = ped.positions[start + 1:start + self.H + 1]

                features.append({
                    'fname': ProcessDataNN._bytes_feature(feature_fname),
                    'input': ProcessDataNN._floatlist_feature(input.ravel().tolist()),
                    'output': ProcessDataNN._floatlist_feature(output.ravel().tolist()),
                    'K': ProcessDataNN._int64list_feature([self.K]),
                    'H': ProcessDataNN._int64list_feature([self.H]),
                    'dinput': ProcessDataNN._int64list_feature([input.shape[1]]),
                    'doutput': ProcessDataNN._int64list_feature([output.shape[1]])
                })

                if len(features) >= self.batch_size:
                    yield fname, features
                    features = []
                    feature_num += 1

        if len(features) >= 0:
            yield fname, features





