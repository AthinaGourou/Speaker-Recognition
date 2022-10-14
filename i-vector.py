import os
import sidekit
import numpy as np
from glob import glob
from multiprocessing import cpu_count
from glob import glob
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)

from model_interface import SidekitModel
from ubm import UBM


class IVector(SidekitModel):
    """Identity Vectors"""
    
    def __init__(self, conf_path):
        super().__init__(conf_path)
        # Set parameters of your system
        self.conf_path = conf_path
        self.NUM_GAUSSIANS = self.conf['num_gaussians']
        self.BATCH_SIZE = self.conf['batch_size']
        self.TV_RANK = self.conf['tv_rank']
        self.TV_ITERATIONS = self.conf['tv_iterations']
        self.ENABLE_PLDA = self.conf['enable_plda']


    def __create_stats(self):
        """
        This private method is used to create Statistic Servers.
        TODO: post some more info
        """
        # Read tv_idmap
        tv_idmap = sidekit.IdMap.read(os.path.join(self.BASE_DIR, "task", "tv_idmap.h5"))
        back_idmap = tv_idmap
        # If PLDA is enabled
        if self.ENABLE_PLDA:
            # Read plda_idmap
            plda_idmap = sidekit.IdMap.read(os.path.join(self.BASE_DIR, "task", "plda_idmap.h5"))
            # Create a joint StatServer for TV and PLDA training data
            back_idmap = plda_idmap.merge(tv_idmap)
            if not back_idmap.validate():
                raise RuntimeError("Error merging tv_idmap & plda_idmap")
        
        # Check UBM model
        ubm_name = "ubm_{}.h5".format(self.NUM_GAUSSIANS)
        ubm_path = os.path.join(self.BASE_DIR, "ubm", ubm_name)
        if not os.path.exists(ubm_path):
            #if UBM model does not exist, train one
            logging.info("Training UBM-{} model".format(self.NUM_GAUSSIANS))
            ubm = UBM(self.conf_path)
            ubm.train()
        #load trained UBM model
        logging.info("Loading trained UBM-{} model".format(self.NUM_GAUSSIANS))
        ubm = sidekit.Mixture()
        ubm.read(ubm_path)
        back_stat = sidekit.StatServer( statserver_file_name=back_idmap, 
                                        ubm=ubm
                                      )
        # Create Feature Server
        fs = self.createFeatureServer()
        
        # Jointly compute the sufficient statistics of TV and (if enabled) PLDA data
        back_filename = 'back_stat_{}.h5'.format(self.NUM_GAUSSIANS)
        if not os.path.isfile(os.path.join(self.BASE_DIR, "stat", back_filename)):
            #BUG: don't use self.NUM_THREADS when assgining num_thread
            # as it's prune to race-conditioning
            back_stat.accumulate_stat(
                ubm=ubm,
                feature_server=fs,
                seg_indices=range(back_stat.segset.shape[0])
                )
            back_stat.write(os.path.join(self.BASE_DIR, "stat", back_filename))
        
        # Load the sufficient statistics from TV training data
        tv_filename = 'tv_stat_{}.h5'.format(self.NUM_GAUSSIANS)
        if not os.path.isfile(os.path.join(self.BASE_DIR, "stat", tv_filename)):
            tv_stat = sidekit.StatServer.read_subset(
                os.path.join(self.BASE_DIR, "stat", back_filename),
                tv_idmap
                )
            tv_stat.write(os.path.join(self.BASE_DIR, "stat", tv_filename))
        
        # Load sufficient statistics and extract i-vectors from PLDA training data
        if self.ENABLE_PLDA:
            plda_filename = 'plda_stat_{}.h5'.format(self.NUM_GAUSSIANS)
            if not os.path.isfile(os.path.join(self.BASE_DIR, "stat", plda_filename)):
                plda_stat = sidekit.StatServer.read_subset(
                    os.path.join(self.BASE_DIR, "stat", back_filename),
                    plda_idmap
                    )
                plda_stat.write(os.path.join(self.BASE_DIR, "stat", plda_filename))
        
        # Load sufficient statistics from test data
        filename = 'test_stat_{}.h5'.format(self.NUM_GAUSSIANS)
        if not os.path.isfile(os.path.join(self.BASE_DIR, "stat", filename)):
            test_idmap = sidekit.IdMap.read(os.path.join(self.BASE_DIR, "task", "test_idmap.h5"))
            test_stat = sidekit.StatServer( statserver_file_name=test_idmap, 
                                            ubm=ubm
                                          )
            # Create Feature Server
            fs = self.createFeatureServer()
            # Jointly compute the sufficient statistics of TV and PLDA data
            #BUG: don't use self.NUM_THREADS when assgining num_thread as it's prune to race-conditioning
            test_stat.accumulate_stat(ubm=ubm,
                                    feature_server=fs,
                                    seg_indices=range(test_stat.segset.shape[0])
                                    )
            test_stat.write(os.path.join(self.BASE_DIR, "stat", filename))

        #enroll
        enroll_filename = 'enroll_stat_{}.h5'.format(self.NUM_GAUSSIANS)        
        if not os.path.isfile(os.path.join(self.BASE_DIR, "stat", enroll_filename)):
            enroll_idmap = sidekit.IdMap.read(os.path.join(self.BASE_DIR, "task", "enroll_idmap.h5"))
            enroll_stat = sidekit.StatServer( statserver_file_name=enroll_idmap, 
                                            ubm=ubm
                                          )
            # Create Feature Server
            fs = self.createFeatureServer()
            # Jointly compute the sufficient statistics of TV and PLDA data
            #BUG: don't use self.NUM_THREADS when assgining num_thread as it's prune to race-conditioning
            enroll_stat.accumulate_stat(ubm=ubm,
                                    feature_server=fs,
                                    seg_indices=range(enroll_stat.segset.shape[0])
                                    )
            enroll_stat.write(os.path.join(self.BASE_DIR, "stat", filename))    



    def train_tv(self):
        """
        This method is used to train the Total Variability (TV) matrix
        and save it into 'ivector' directory !! 
        """
        # Create status servers
        self.__create_stats()

        # Load UBM model
        model_name = "ubm_{}.h5".format(self.NUM_GAUSSIANS)
        ubm = sidekit.Mixture()
        ubm.read(os.path.join(self.BASE_DIR, "ubm", model_name))

        # Train TV matrix using FactorAnalyser
        filename = "tv_matrix_{}".format(self.NUM_GAUSSIANS)
        outputPath = os.path.join(self.BASE_DIR, "ivector", filename)
        tv_filename = 'tv_stat_{}.h5'.format(self.NUM_GAUSSIANS)
        fa = sidekit.FactorAnalyser()
        fa.total_variability_single(os.path.join(self.BASE_DIR, "stat", tv_filename),
                                    ubm,
                                    tv_rank=self.TV_RANK,
                                    nb_iter=self.TV_ITERATIONS,
                                    min_div=True,
                                    tv_init=None,
                                    batch_size=self.BATCH_SIZE,
                                    save_init=False,
                                    output_file_name=outputPath
                                   )
        # tv = fa.F # TV matrix
        # tv_mean = fa.mean # Mean vector
        # tv_sigma = fa.Sigma # Residual covariance matrix

        # Clear files produced at each iteration
        filename_regex = "tv_matrix_{}_it-*.h5".format(self.NUM_GAUSSIANS)
        lst = glob(os.path.join(self.BASE_DIR, "ivector", filename_regex))
        for f in lst:
            os.remove(f)
    

    def evaluate(self, explain=True):
        """
        This method is used to score our trained model. 
        """
        # Load UBM model
        model_name = "ubm_{}.h5".format(self.NUM_GAUSSIANS)
        ubm = sidekit.Mixture()
        ubm.read(os.path.join(self.BASE_DIR, "ubm", model_name))

        # Load TV matrix
        filename = "tv_matrix_{}".format(self.NUM_GAUSSIANS)
        outputPath = os.path.join(self.BASE_DIR, "ivector", filename)
        fa = sidekit.FactorAnalyser(outputPath+".h5")

        # Extract i-vectors from enrollment data
        logging.info("Extracting i-vectors from enrollment data")
        filename = 'enroll_stat_{}.h5'.format(self.NUM_GAUSSIANS)
        enroll_stat = sidekit.StatServer.read(os.path.join(self.BASE_DIR, 'stat', filename))
        enroll_iv = fa.extract_ivectors_single( ubm=ubm,
                                                stat_server=enroll_stat,
                                                uncertainty=False
                                              )
    
        # Extract i-vectors from test data
        logging.info("Extracting i-vectors from test data")
        filename = 'test_stat_{}.h5'.format(self.NUM_GAUSSIANS)
        test_stat = sidekit.StatServer.read(os.path.join(self.BASE_DIR, 'stat', filename))
        test_iv = fa.extract_ivectors_single(ubm=ubm,
                                             stat_server=test_stat,
                                             uncertainty=False
                                            )
        logging.info("Extracting i-vectors from target data")
        filename = 'plda_stat_{}.h5'.format(self.NUM_GAUSSIANS)
        plda = os.path.join(self.BASE_DIR, "stat", filename)
        #  Load sufficient statistics and extract i-vectors from PLDA training data
        plda_iv = fa.extract_ivectors(ubm=ubm,
                                       stat_server_filename = plda,
                                       batch_size=self.BATCH_SIZE,
                                       num_thread=self.NUM_THREADS
                                      )





        # Do cosine distance scoring and write results
        logging.info("Calculating cosine score")
        test_ndx = sidekit.Ndx.read(os.path.join(self.BASE_DIR, "task", "test_ndx.h5"))
       
        wccn = plda_iv.get_wccn_choleski_stat1()
        scores_cos_wccn = sidekit.iv_scoring.cosine_scoring(enroll_iv, test_iv, test_ndx, wccn=wccn)
        
        # Write scores
        filename = "ivector_scores_cos_{}.h5".format(self.NUM_GAUSSIANS)
        scores_cos_wccn.write(os.path.join(self.BASE_DIR, "result", filename))
        
        # Explain the Analysis by writing more readible text file
        if explain:
            modelset = list(scores_cos_wccn.modelset)
            segset = list(scores_cos_wccn.segset)
            scores = np.array(scores_cos_wccn.scoremat)
            filename = "ivector_scores_explained_{}.txt".format(iv.NUM_GAUSSIANS)
            fout = open(os.path.join(iv.BASE_DIR, "result", filename), "a")
            fout.truncate(0) #clear content
            for seg_idx, seg in enumerate(segset):
                fout.write("Wav: {}\n".format(seg))
                for pathology_idx, pathology in enumerate(modelset):
                    fout.write("\Pathology {}:\t{}\n".format(pathology, scores[pathology_idx, seg_idx]))
                fout.write("\n")
            fout.close()
        
        # Do mahalanobis distance
        #logging.info("Calculating cosine score")
        #test_ndx = sidekit.Ndx.read(os.path.join(self.BASE_DIR, "task", "test_ndx.h5"))
       
        #meanEFR, CovEFR = plda_iv.estimate_spectral_norm_stat1(3)

        #plda_iv_efr1 = copy.deepcopy(plda_iv)
        #enroll_iv_efr1 = copy.deepcopy(enroll_iv)
        #test_iv_efr1 = copy.deepcopy(test_iv)

        #plda_iv_efr1.spectral_norm_stat1(meanEFR[:1], CovEFR[:1])
        #enroll_iv_efr1.spectral_norm_stat1(meanEFR[:1], CovEFR[:1])
        #test_iv_efr1.spectral_norm_stat1(meanEFR[:1], CovEFR[:1])
        #M1 = plda_iv_efr1.get_mahalanobis_matrix_stat1()
        #scores_mah_efr1 = sidekit.iv_scoring.mahalanobis_scoring(enroll_iv_efr1, test_iv_efr1, test_ndx, M1)

        # Write scores
        # filename = "ivector_scores_cos_{}.h5".format(self.NUM_GAUSSIANS)
        # scores_mah_efr1.write(os.path.join(self.BASE_DIR, "result", filename))

        # Explain the Analysis by writing more readible text file
        #if explain:
        #    modelset = list(scores_mah_efr1.modelset)
        #    segset = list(scores_mah_efr1.segset)
        #    scores = np.array(scores_mah_efr1.scoremat)
        #    filename = "ivector_scores_explained_{}.txt".format(iv.NUM_GAUSSIANS)
        #    fout = open(os.path.join(iv.BASE_DIR, "result", filename), "a")
        #    fout.truncate(0) #clear content
        #    for seg_idx, seg in enumerate(segset):
        #        fout.write("Wav: {}\n".format(seg))
        #        for speaker_idx, speaker in enumerate(modelset):
        #            fout.write("\tSpeaker {}:\t{}\n".format(speaker, scores[speaker_idx, seg_idx]))
        #        fout.write("\n")
        #    fout.close()
        
        #WCCN-LDA
        #LDA = plda_iv.get_lda_matrix_stat1(150)

        #plda_iv_lda = copy.deepcopy(plda_iv)
        #enroll_iv_lda = copy.deepcopy(enroll_iv)
        #test_iv_lda = copy.deepcopy(test_iv)

        #plda_iv_lda.rotate_stat1(LDA)
        #enroll_iv_lda.rotate_stat1(LDA)
        #test_iv_lda.rotate_stat1(LDA)

        #scores_cos_lda = sidekit.iv_scoring.cosine_scoring(enroll_iv_lda, test_iv_lda, test_ndx, wccn=None)
        #wccn = plda_iv_lda.get_wccn_choleski_stat1()
        #scores_cos_lda_wcnn = sidekit.iv_scoring.cosine_scoring(enroll_iv_lda, test_iv_lda, test_ndx, wccn=wccn) 
        
        
        
        # Write scores
        #filename = "ivector_scores_cos_{}.h5".format(self.NUM_GAUSSIANS)
        #scores_cos_lda_wcnn.write(os.path.join(self.BASE_DIR, "result", filename))
        
        # Explain the Analysis by writing more readible text file
        #if explain:
        #    modelset = list(scores_cos_lda_wcnn.modelset)
        #    segset = list(scores_cos_lda_wcnn.segset)
        #    scores = np.array(scores_cos_lda_wcnn.scoremat)
        #    filename = "ivector_scores_explained_{}.txt".format(iv.NUM_GAUSSIANS)
        #    fout = open(os.path.join(iv.BASE_DIR, "result", filename), "a")
        #    fout.truncate(0) #clear content
        #    for seg_idx, seg in enumerate(segset):
        #        fout.write("Wav: {}\n".format(seg))
        #        for speaker_idx, speaker in enumerate(modelset):
        #            fout.write("\tSpeaker {}:\t{}\n".format(speaker, scores[speaker_idx, seg_idx]))
        #        fout.write("\n")
        #    fout.close()      
        


    def plotDETcurve(self):
        """
        This method is used to plot the DET (Detection Error Tradeoff) and 
        save it on the disk.
        """
        # Read test scores
        filename = "ivector_scores_cos_{}.h5".format(self.NUM_GAUSSIANS)
        scores_dir = os.path.join(self.BASE_DIR, "result", filename)
        scores_gmm_ubm = sidekit.Scores.read(scores_dir)
        # Read the key
        key = sidekit.Key.read_txt(os.path.join(self.BASE_DIR, "task", "test_trials.txt"))

        # Make DET plot
        logging.info("Drawing DET Curve")
        dp = sidekit.DetPlot(window_style='sre10', plot_title='Scores IVector')
        dp.set_system_from_scores(scores_gmm_ubm, key, sys_name='Cosine WCCN')
        dp.create_figure()
        # DET type
        if self.conf['DET_curve'] == "rocch":
            dp.plot_rocch_det(idx=0)
        elif self.conf['DET_curve'] == "steppy":
            dp.plot_steppy_det(idx=0)
        else:
            raise NameError("Unsupported DET-curve-plotting method!!")
        dp.plot_DR30_both(idx=0) #dotted line for Doddington's Rule
        prior = sidekit.logit_effective_prior(0.001, 1, 1)
        dp.plot_mindcf_point(prior, idx=0) #minimum dcf point
        # Save the graph
        graphname = "det_ivector{}.png".format(self.NUM_GAUSSIANS)
        dp.__figure__.savefig(os.path.join(self.BASE_DIR, "result", graphname))


    def getAccuracy(self):
        import h5py
        # Load scores file
        filename = "ivector_scores_cos_{}.h5".format(self.NUM_GAUSSIANS)
        filepath = os.path.join(self.BASE_DIR, "result", filename)
        h5 = h5py.File(filepath, mode="r")
        modelset = list(h5["modelset"])
        segest = list(h5["segset"])
        scores = np.array(h5["scores"])
        
        # Get Accuracy
        accuracy = super().getAccuracy(modelset, segest, scores, threshold=0)
        return accuracy



if __name__ == "__main__":
    conf_path = "conf.yaml"
    iv = IVector(conf_path)
    iv.train_tv()
    iv.evaluate()
    iv.plotDETcurve()
    print( "Accuracy: {}%".format(iv.getAccuracy()) )
