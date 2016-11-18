############################
# Runs specific experiment #
############################

import argparse, yaml
import os, shutil
import numpy as np, random

from general.models.analyze_model import AnalyzeModels

from models.nn.prediction_model_nn import PredictionModelNN
from models.bd_nn.prediction_model_bd_nn import PredictionModelBDNN

DATA_FOLDER = '/home/gkahn/code/AHRI_project/data/seq_hotel'
EXP_PARENT_FOLDER = '/home/gkahn/code/AHRI_project/exps/'

def load_yaml_from_model(model):
    yaml_path = os.path.join(os.path.dirname(__file__), 'models/{0}/params_{0}.yaml'.format(model))
    with open(yaml_path, "r") as f:
        params = yaml.load(f)
    params['yaml_path'] = yaml_path
    return params

def load_yaml_from_exp(model, exp):
    yaml_path = os.path.join(EXP_PARENT_FOLDER, model, exp, 'params.yaml')
    with open(yaml_path, "r") as f:
        params = yaml.load(f)
    return params

def get_exp_folder(model, params):
    return os.path.join(EXP_PARENT_FOLDER, model, params['exp'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(run='train')
    parser_analyze = subparsers.add_parser('analyze')
    parser_analyze.set_defaults(run='analyze')

    ### parser train
    parser_train.add_argument('model', type=str, choices=('nn', 'bd_nn'))

    ### parser analyze
    parser_analyze.add_argument('exps', nargs='+')

    args = parser.parse_args()
    run = args.run

    if run == 'train':
        model = args.model

        ### load yaml so all files can access
        params = load_yaml_from_model(model)

        np.random.seed(params['random_seed'])
        random.seed(params['random_seed'])

        ### create exp folder
        exp_folder = os.path.join(EXP_PARENT_FOLDER, model, params['exp'])
        assert(not os.path.exists(exp_folder))
        os.makedirs(exp_folder)
        shutil.copy(params['yaml_path'], os.path.join(exp_folder, 'params.yaml'))

        if model == 'nn':
            prediction_model = PredictionModelNN(exp_folder, DATA_FOLDER, params)
        elif model == 'bd_nn':
            prediction_model = PredictionModelBDNN(exp_folder, DATA_FOLDER, params)

        prediction_model.train()
    elif run == 'analyze':
        exps = args.exps

        prediction_models = []
        for exp in exps:
            model, exp = exp.split('/')
            params = load_yaml_from_exp(model, exp)
            exp_folder = get_exp_folder(model, params)

            assert(os.path.exists(exp_folder))

            if model == 'nn':
                prediction_model = PredictionModelNN(exp_folder, DATA_FOLDER, params)
            elif model == 'bd_nn':
                prediction_model = PredictionModelBDNN(exp_folder, DATA_FOLDER, params)
            else:
                raise Exception('Model {0} not valid'.format(model))

            prediction_models.append(prediction_model)

        analyze_models = AnalyzeModels(prediction_models)
        analyze_models.run()

    else:
        raise Exception('Run {0} is not valid'.format(run))

