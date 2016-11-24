import os

from general.models.process_data import ProcessData

class ProcessDataGPNN(ProcessData):

    def __init__(self, folder, params):
        ProcessData.__init__(self, folder, params)

    #############
    ### Files ###
    #############

    @property
    def data_folder(self):
        data_folder = os.path.join(self.folder, 'gp_nn_' + self.params['exp'])
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
        return train_pedestrians, val_pedestrians