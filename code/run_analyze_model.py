import os

from main import load_yaml_from_exp, get_exp_folder, DATA_FOLDER, RESULTS_FOLDER

from general.models.analyze_model import AnalyzeModels

from models.nn.prediction_model_nn import PredictionModelNN
from models.bd_nn.prediction_model_bd_nn import PredictionModelBDNN
from models.gp.prediction_model_gp import PredictionModelGP
from models.gp_nn.prediction_model_gp_nn import PredictionModelGPNN
from models.lr.prediction_model_lr import PredictionModelLR

analyze0_dict = {
    'title': 'Linear regression',
    'exps': ['lr/exp0', 'lr/exp1', 'lr/exp2', 'lr/exp3'],
    'lr/exp0': {'name': '$H = 3$'},
    'lr/exp1': {'name': '$H = 6$'},
    'lr/exp2': {'name': '$H = 9$'},
    'lr/exp3': {'name': '$H = 12$'}
}

analyze1_dict = {
    'title': 'Gaussian process',
    'exps': ['gp/exp0', 'gp/exp1', 'gp/exp2', 'gp/exp3'],
    'gp/exp0': {'name': '$H = 3$'},
    'gp/exp1': {'name': '$H = 6$'},
    'gp/exp2': {'name': '$H = 9$'},
    'gp/exp3': {'name': '$H = 12$'}
}

analyze2_dict = {
    'title': 'Neural network',
    'exps': ['nn/exp0', 'nn/exp1', 'nn/exp2', 'nn/exp3'],
    'nn/exp0': {'name': '$H = 3$'},
    'nn/exp1': {'name': '$H = 6$'},
    'nn/exp2': {'name': '$H = 9$'},
    'nn/exp3': {'name': '$H = 12$'}
}

analyze3_dict = {
    'title': 'Bootstrap + dropout neural network',
    'exps': ['bd_nn/exp0', 'bd_nn/exp1', 'bd_nn/exp2', 'bd_nn/exp3'],
    'bd_nn/exp0': {'name': '$H = 3$'},
    'bd_nn/exp1': {'name': '$H = 6$'},
    'bd_nn/exp2': {'name': '$H = 9$'},
    'bd_nn/exp3': {'name': '$H = 12$'}
}

analyze4_dict = {
    'title': 'Neural network + gaussian process',
    'exps': ['gp_nn/exp0', 'gp_nn/exp1', 'gp_nn/exp2', 'gp_nn/exp3'],
    'gp_nn/exp0': {'name': '$H = 3$'},
    'gp_nn/exp1': {'name': '$H = 6$'},
    'gp_nn/exp2': {'name': '$H = 9$'},
    'gp_nn/exp3': {'name': '$H = 12$'}
}

analyze5_dict = {
    'title': 'H = 3',
    'exps': ['lr/exp0', 'nn/exp0', 'gp/exp0', 'bd_nn/exp0', 'gp_nn/exp0'],
    'lr/exp0': {'name': 'lr'},
    'nn/exp0': {'name': 'nn'},
    'gp/exp0': {'name': 'gp'},
    'bd_nn/exp0': {'name': 'bd nn'},
    'gp_nn/exp0': {'name': 'gp nn'},
}

analyze6_dict = {
    'title': 'H = 6',
    'exps': ['lr/exp1', 'nn/exp1', 'gp/exp1', 'bd_nn/exp1', 'gp_nn/exp1'],
    'lr/exp1': {'name': 'lr'},
    'nn/exp1': {'name': 'nn'},
    'gp/exp1': {'name': 'gp'},
    'bd_nn/exp1': {'name': 'bd nn'},
    'gp_nn/exp1': {'name': 'gp nn'},
}

analyze7_dict = {
    'title': 'H = 9',
    'exps': ['lr/exp2', 'nn/exp2', 'gp/exp2', 'bd_nn/exp2', 'gp_nn/exp2'],
    'lr/exp2': {'name': 'lr'},
    'nn/exp2': {'name': 'nn'},
    'gp/exp2': {'name': 'gp'},
    'bd_nn/exp2': {'name': 'bd nn'},
    'gp_nn/exp2': {'name': 'gp nn'},
}

analyze8_dict = {
    'title': 'H = 12',
    'exps': ['lr/exp3', 'nn/exp3', 'gp/exp3', 'bd_nn/exp3', 'gp_nn/exp3'],
    'lr/exp3': {'name': 'lr'},
    'nn/exp3': {'name': 'nn'},
    'gp/exp3': {'name': 'gp'},
    'bd_nn/exp3': {'name': 'bd nn'},
    'gp_nn/exp3': {'name': 'gp nn'},
}

analyze9_dict = {
    'title': 'TEMP',
    'exps': ['lr/exp2'],
    'lr/exp2': {'name': 'Model'},
}

# analyze_dicts = [analyze0_dict, analyze1_dict, analyze2_dict, analyze3_dict, analyze4_dict]
# analyze_dicts += [analyze5_dict, analyze6_dict, analyze7_dict, analyze8_dict]
analyze_dicts = [analyze9_dict]

for analyze_dict in analyze_dicts:
    prediction_models = []
    for exp_full in analyze_dict['exps']:
        model, exp = exp_full.split('/')
        params = load_yaml_from_exp(model, exp)
        params.update(analyze_dict[exp_full]) # add name etc to it
        exp_folder = get_exp_folder(model, params)

        assert(os.path.exists(exp_folder))

        if model == 'nn':
            prediction_model = PredictionModelNN(exp_folder, DATA_FOLDER, params)
        elif model == 'bd_nn':
            prediction_model = PredictionModelBDNN(exp_folder, DATA_FOLDER, params)
        elif model == 'gp':
            prediction_model = PredictionModelGP(exp_folder, DATA_FOLDER, params)
        elif model == 'gp_nn':
            prediction_model = PredictionModelGPNN(exp_folder, DATA_FOLDER, params)
        elif model == 'lr':
            prediction_model = PredictionModelLR(exp_folder, DATA_FOLDER, params)
        else:
            raise Exception('Model {0} not valid'.format(model))

        prediction_models.append(prediction_model)

    analyze_models = AnalyzeModels(prediction_models)
    analyze_models.run(title=analyze_dict['title'], save_folder=RESULTS_FOLDER)
