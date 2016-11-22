import os
import itertools

import numpy as np
import tensorflow as tf

from general.models.process_data import ProcessData

class ProcessDataNN(ProcessData):

    def __init__(self, folder, params):
        ProcessData.__init__(self, folder, params)

    #############
    ### Files ###
    #############

    @property
    def data_folder(self):
        data_folder = os.path.join(self.folder, 'nn_' + self.params['exp'])
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        return data_folder

    ##############################
    ### Compute inputs/outputs ###
    ##############################


    #################
    ### Save data ###
    #################

    def save_pedestrians(self, pedestrians):
        train_pedestrians, val_pedestrians = ProcessData.save_pedestrians(self, pedestrians)

        def chunks(l, n):
            for i in xrange(0, len(l), n):
                yield l[i:i + n]

        for pedestrians, data_file_func in [(train_pedestrians, self.train_data_file),
                                            (val_pedestrians, self.val_data_file)]:
            inputs = list(itertools.chain(*[p.inputs for p in pedestrians]))
            outputs = list(itertools.chain(*[p.outputs for p in pedestrians]))

            for i, (input_chunk, output_chunk) in enumerate(zip(chunks(inputs, self.params['features_per_file']),
                                                                chunks(outputs, self.params['features_per_file']))):
                fname = data_file_func(i)
                features = [{
                    'fname': ProcessDataNN._bytes_feature(
                        '{0}_{1}'.format(os.path.splitext(os.path.basename(fname))[0], j)
                    ),
                    'input': ProcessDataNN._floatlist_feature(np.ravel(input).tolist()),
                    'output': ProcessDataNN._floatlist_feature(np.ravel(output).tolist())
                } for j, (input, output) in enumerate(zip(input_chunk, output_chunk))]

                self.save_tfrecord(fname, features)

        return train_pedestrians, val_pedestrians

    @staticmethod
    def _floatlist_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _int64list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def save_tfrecord(self, fname, features):
        writer = tf.python_io.TFRecordWriter(fname)

        for feature in features:
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()
