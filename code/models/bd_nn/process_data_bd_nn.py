import os, itertools, random

import numpy as np

from models.nn.process_data_nn import ProcessDataNN

def sample_with_replacement(*lists):
    length = len(lists[0])
    assert(np.all([len(l) == length for l in lists]))
    idxs = range(length)
    idxs = [random.choice(idxs) for _ in xrange(length)]
    return [[l[i] for i in idxs] for l in lists]

class ProcessDataBDNN(ProcessDataNN):

    def __init__(self, folder, params):
        ProcessDataNN.__init__(self, folder, params)

    ##############################
    ### Compute inputs/outputs ###
    ##############################

    ### see ProcessDataNN

    #################
    ### Save data ###
    #################

    def save_pedestrians(self, pedestrians):
        ### split it up
        train_pedestrians, val_pedestrians = super(ProcessDataNN, self).save_pedestrians(pedestrians)

        for pedestrians, data_file_func in [(train_pedestrians, self.train_data_file),
                                            (val_pedestrians, self.val_data_file)]:
            inputs = list(itertools.chain(*[p.inputs for p in pedestrians]))
            outputs = list(itertools.chain(*[p.outputs for p in pedestrians]))
            assert(len(inputs) == len(outputs))
            N = len(inputs)

            inputs_b, outputs_b = zip(*[sample_with_replacement(inputs, outputs)
                                        for _ in xrange(self.params['bootstrap'])])

            i = 0
            while i < N:
                features = []

                fname = data_file_func(i)
                for j in xrange(min(self.params['features_per_file'], N - i)):
                    feature = {
                        'fname': ProcessDataNN._bytes_feature(
                            '{0}_{1}'.format(os.path.splitext(os.path.basename(fname))[0], j))
                    }
                    for b in xrange(self.params['bootstrap']):
                        feature['input_b{0}'.format(b)] = ProcessDataNN._floatlist_feature(np.ravel(inputs_b[b][i]).tolist())
                        feature['output_b{0}'.format(b)] = ProcessDataNN._floatlist_feature(np.ravel(outputs_b[b][i]).tolist())
                    features.append(feature)

                i += self.params['features_per_file']

                self.save_tfrecord(fname, features)
