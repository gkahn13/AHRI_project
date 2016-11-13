############################
# Runs specific experiment #
############################

import argparse
import os
import numpy as np, random

from models.nn.process_data_nn import ProcessDataNN

from config import load_params, params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_process = subparsers.add_parser('process')
    parser_process.set_defaults(run='process')
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(run='train')
    parser_analyze = subparsers.add_parser('analyze')
    parser_analyze.set_defaults(run='analyze')

    ### all arguments
    for subparser in (parser_process, parser_train):
        subparser.add_argument('model', type=str, choices=('nn',))

    args = parser.parse_args()
    run = args.run
    model = args.model
    data_folder = '/home/gkahn/code/AHRI_project/data/seq_hotel'

    # load yaml so all files can access
    yaml_path = os.path.join(os.path.dirname(__file__), 'models/{0}/params_{0}.yaml'.format(model))
    load_params(yaml_path)
    params['yaml_path'] = yaml_path

    np.random.seed(params['random_seed'])
    random.seed(params['random_seed'])

    if run == 'process':
        if model == 'nn':
            process_data = ProcessDataNN(data_folder)

        process_data.process()

    elif run == 'train':
        pass # TODO
    else:
        raise Exception('Run {0} is not valid'.format(run))

