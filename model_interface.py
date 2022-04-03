import os
import sidekit
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from multiprocessing import cpu_count
from utils import parse_yaml
from sklearn.metrics import confusion_matrix
import pandas as pd
import openpyxl
import seaborn as sns

class SidekitModel():
    """This class is just an interface for my models to inherit"""

    def __init__(self, conf_filepath):
        self.conf = parse_yaml(conf_filepath)
        # use 0 to disable multi-processing
        self.NUM_THREADS = cpu_count()
        # The parent directory of the project
        self.BASE_DIR = self.conf['outpath']
    

    def createFeatureServer(self, group=None):
        """
        This methos is used to crate FeatureServer object which is an object for
        features management. It loads datasets from a HDF5 files ( produced by
        a FeaturesExtractor object)
        Args:
            group (string): the group of features to manage. If None, then it 
            will create a generic FeatureServer over the feat directory.
        Returns:
            server: which is the FeatureServer object
        """
        if group:
            feat_dir = os.path.join(self.BASE_DIR, "feat", group)
        else:
            feat_dir = os.path.join(self.BASE_DIR, "feat")
        # feature_filename_structure: structure of the filename to use to load HDF5 files
        # dataset_list: string of the form ["cep", "fb", vad", energy", "bnf"]
        # feat_norm: type of normalization to apply as post-processing
        # delta: if True, append the first order derivative
        # double_delta: if True, append the second order derivative
        # rasta: if True, perform RASTA filtering
        # keep_all_features: boolean, if True, keep all features; if False,
        #       keep frames according to the vad labels
        server = sidekit.FeaturesServer(
                feature_filename_structure=os.path.join(feat_dir, "{}.h5"),
                dataset_list=self.conf['features'],
                feat_norm="cmvn", #cepstral mean-variance normalization
                delta=True,
                double_delta=True,
                rasta=True,
                keep_all_features=True)
        logging.info("Feature-Server is created")
        logging.debug(server)
        return server
    
    def train(self):
        pass
    
    def evaluate(self):
        pass
    
    def plotDETcurve(self):
        pass
    
    def getAccuracy(self, pathology, test_files, scores, threshold=0):
        """
        This method is used to get the accuracy of a model over a bunch of data
        files. The accuracy is returned in percentage. 
        NOTE: The file is considered to be correct if the correct speaker has the
        max score (that's in case the speaker is in the training set). And also
        if the max score is below threshold (that's in case the speaker is not
        in the training set).
        Args
            pathology: list of pathologies
            test_files: list of filenames that used to evaluate model
            scores: the score numpy matrix obtained by the model
            threshold: the value above which we will consider the verification
                is done correctly. In other words, if the score>threshold, then
                the answer is considered; otherwise, the answer is not considered
        Returns
            accuracy (float): accuracy of the model in percentage
        """
        assert scores.shape == (len(pathology), len(test_files)),\
            "The dimensions of the input don't match"
        accuracy = 0.
        pathology = [sp.decode() for sp in pathology]
        max_indices = np.argmax(scores, axis=0)
        max_scores = np.max(scores, axis=0)
        actual_pathology_list=[]
        predicted_pathology_list=[]
        for idx, test_filename in enumerate(test_files):
            test_filename = test_filename.decode() #convert from byte to string
            actual_pathology = test_filename.split("/")[-1].split(".")[0]
            predicted_pathology = pathology[max_indices[idx]]
            #evaluate the test data file
            if max_scores[idx] < threshold:
                if actual_pathology not in pathology:
                    accuracy += 1
            else:
                if predicted_pathology == actual_pathology:
                    accuracy += 1.
            actual_pathology_list.append(actual_pathology)
            predicted_pathology_list.append(predicted_pathology)
            
        df_confusion=confusion_matrix(actual_pathology_list, predicted_pathology_list)   
        print(df_confusion)
        
        #df = pd.DataFrame([actual_pathology_list],[predicted_pathology_list],
                #   columns=['actual', 'predicted'])

        #print(df)

        #df.to_excel('pandas_to_excel.xlsx', sheet_name='new_sheet_name')
            
        return accuracy*100./len(test_files)
