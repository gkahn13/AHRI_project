import os
import itertools
import pickle

import numpy as np
import tensorflow as tf

from general.models.process_data import ProcessData

class ProcessDataNN(ProcessData):

    def __init__(self, folder, params):
        ProcessData.__init__(self, folder, params)

    ##############################
    ### Compute inputs/outputs ###
    ##############################

    def compute_inputs_outputs(self, pedestrians):
        ### select feature method
        if self.params['feature_type'] == 'position':
            add_inputs_outputs = self.add_inputs_outputs_position
            input_output_shape = self.input_output_shape_position
        else:
            raise Exception('Feature type {0} not valid'.format(self.params['feature_type']))

        ### delete previous features
        for data_file in self.all_data_files:
            os.remove(data_file)

        ### compute inputs/outputs for each pedestrian
        add_inputs_outputs(pedestrians)

        ### save input output shape
        with open(self.input_output_shape_file, 'w') as f:
            pickle.dump(input_output_shape, f)

    @property
    def input_output_shape_position(self):
        return {
            'input': [self.params['K'], 2],
            'output': [self.params['H'], 2]
        }
    def add_inputs_outputs_position(self, pedestrians):
        for ped in pedestrians:
            for start in xrange(self.params['K'], len(ped) - (self.params['K'] + self.params['H'])):
                input = ped.positions[start - self.params['K'] + 1:start + 1]
                output = ped.positions[start + 1:start + self.params['H'] + 1]

                ped.add_input_output(input, output)

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
