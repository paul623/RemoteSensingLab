from pathlib import Path


class ConfigForTraining(object):
    def __init__(self, choice, save_dir_name, PATCH_STRIDE=200, batch_size=4, batch_size_test=2,
                 num_workers=8, epochs=500):
        self.choice = choice
        assert self.choice in ['LGC', 'CIA']
        if self.choice == 'LGC':
            self.root_dir = Path("/home/zbl/datasets_paper/LGC/")
            self.save_dir = Path("/home/zbl/datasets/STFusion/RunLog/STFINet/LGC-gan/") / save_dir_name
            self.image_size = [2720, 3200]
            self.patch_size = [340, 400]
            self.PATCH_SIZE = 256
            self.PATCH_STRIDE = 200
        else:
            self.root_dir = Path("/home/zbl/datasets_paper/CIA/")
            self.save_dir = Path("/home/zbl/datasets/STFusion/RunLog/STFINet/CIA-gan/") / save_dir_name
            self.image_size = [2040, 1720]
            self.patch_size = [255, 430]
            self.PATCH_SIZE = 256
            self.PATCH_STRIDE = 200
        self.train_dir = self.root_dir / 'train'
        self.val_dir = self.root_dir / 'val'
        self.save_tif_dir = self.save_dir / 'test'
        self.last_h2sp = self.save_dir / 'xnet.pth'
        self.best_h2sp = self.save_dir / 'best.pth'
        self.csv_history = self.save_dir / 'history.csv'
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.epochs = epochs

class ConfigForTrainingSwin(object):
    def __init__(self, choice, save_dir_name, PATCH_SIZE=256, PATCH_STRIDE=200, batch_size=4, batch_size_test=2,
                 num_workers=8, epochs=500):
        self.choice = choice
        assert self.choice in ['LGC', 'CIA']
        if self.choice == 'LGC':
            self.root_dir = Path("/home/zbl/datasets_paper/LGC-swinSTFM/")
            self.save_dir = Path("/home/zbl/datasets/STFusion/RunLog/FinePainterNet/LGC/") / save_dir_name
            self.image_size = [2720, 3200]
            self.patch_size = [340, 200]
        else:
            self.root_dir = Path("/home/zbl/datasets_paper/CIA-swinSTFM/")
            self.save_dir = Path("/home/zbl/datasets/STFusion/RunLog/STFINet/CIA/") / save_dir_name
            self.image_size = [2040, 1720]
            self.patch_size = [255, 215]
        self.train_dir = self.root_dir / 'train'
        self.val_dir = self.root_dir / 'val'
        self.save_tif_dir = self.save_dir / 'test'
        self.last_h2sp = self.save_dir / 'xnet.pth'
        self.best_h2sp = self.save_dir / 'best.pth'
        self.csv_history = self.save_dir / 'history.csv'
        self.PATCH_SIZE = PATCH_SIZE
        self.PATCH_STRIDE = PATCH_STRIDE
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.epochs = epochs

class ConfigForTrainingGAN(object):
    def __init__(self, choice, save_dir_name, PATCH_SIZE=256, PATCH_STRIDE=200, batch_size=4, batch_size_test=2,
                 num_workers=12, epochs=500):
        self.choice = choice
        assert self.choice in ['LGC', 'CIA']
        if self.choice == 'LGC':
            self.root_dir = Path("/home/zbl/datasets_paper/LGC-swinSTFM/")
            self.save_dir = Path("/home/zbl/datasets_paper/RunLog/FinePainterNet-GAN/LGC/") / save_dir_name
            self.image_size = [2720, 3200]
            self.patch_size = [340, 200]
        else:
            self.root_dir = Path("/home/zbl/datasets_paper/CIA-temp")
            self.save_dir = Path("/home/zbl/datasets_paper/RunLog/FinePainterNet-GAN/CIA/") / save_dir_name
            self.image_size = [2040, 1720]
            self.patch_size = [255, 430]
        self.train_dir = self.root_dir / 'train'
        self.val_dir = self.root_dir / 'val'
        self.save_tif_dir = self.save_dir / 'test'
        self.last_g = self.save_dir / 'generator.pth'
        self.last_d = self.save_dir / 'discriminator.pth'
        self.best = self.save_dir / 'best.pth'
        self.csv_history = self.save_dir / 'history.csv'
        self.PATCH_SIZE = PATCH_SIZE
        self.PATCH_STRIDE = PATCH_STRIDE
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.epochs = epochs


class ConfigForEvaluation(object):
    def __init__(self, choice, save_dir_name):
        self.choice = choice
        assert choice in ['LGC', 'CIA']
        if choice == 'LGC':
            self.ground_truth_dir = "/home/zbl/datasets/STFusion/LGC/LGC_data/refs/"
            self.predict_dir = save_dir_name
            self.predict_img_names = ['PRED_2005_029_0129-2005_013_0113.tif', 'PRED_2005_045_0214-2005_029_0129.tif',
                                      'PRED_2005_045_0214-2005_061_0302.tif', 'PRED_2005_061_0302-2005_045_0214.tif']
            self.ref_img_names = ['20050113_TM.tif', '20050129_TM.tif', '20050302_TM.tif', '20050214_TM.tif']
        else:
            self.ground_truth_dir = "/home/zbl/datasets_paper/CIA/refs/"
            self.predict_dir = save_dir_name
            self.predict_img_names = ['PRED_2001_306_1102-2001_290_1017.tif', 'PRED_2001_290_1017-2001_306_1102.tif',
                                      'PRED_2001_306_1102-2001_313_1109.tif', 'PRED_2001_338_1204-2001_329_1125.tif',
                                      'PRED_2001_329_1125-2001_338_1204.tif']
            self.ref_img_names = ['20011017_TM.tif', '20011102_TM.tif', '20011109_TM.tif', '20011125_TM.tif',
                                  '20011204_TM.tif']


class ConfigForEvaluationForSwin(object):
    def __init__(self, choice, save_dir_name):
        assert choice in ['LGC', 'CIA']
        self.choice = choice
        if choice == 'LGC':
            self.ground_truth_dir = "/home/zbl/datasets_paper/LGC-swinSTFM/refs/"
            self.predict_dir = save_dir_name
            self.predict_img_names = ['PRED_2004_331_1126-2004_347_1212.tif']
            self.ref_img_names = ['20041212_TM.tif']
        else:
            self.ground_truth_dir = "/home/zbl/datasets_paper/CIA-swinSTFM/refs/"
            self.predict_dir = save_dir_name
            # self.predict_img_names = ['PRED_2001_329_1125-2002_012_0112.tif', 'PRED_2002_005_0105-2002_012_0112.tif',
            #                           'PRED_2002_044_0213-2002_012_0112.tif', 'PRED_2002_076_0317-2002_012_0112.tif']
            # self.ref_img_names = ['20020112_TM.tif',  '20020112_TM.tif', '20020112_TM.tif', '20020112_TM.tif']
            self.predict_img_names = ['PRED_2001_329_1125-2002_012_0112.tif']
            self.ref_img_names = ['20020112_TM.tif']

class ConfigForEvaluationForMLFF_GAN(object):
    def __init__(self, choice, save_dir_name):
        self.choice = choice
        assert choice in ['LGC', 'CIA']
        if choice == 'LGC':
            self.ground_truth_dir = "/home/zbl/datasets/STFusion/LGC/LGC_data/refs/"
            self.predict_dir = save_dir_name
            self.predict_img_names = ['PRED_2005_029_0129-2005_013_0113.tif', 'PRED_2005_045_0214-2005_029_0129.tif',
                                      'PRED_2005_045_0214-2005_061_0302.tif', 'PRED_2005_061_0302-2005_045_0214.tif']
            self.ref_img_names = ['20050113_TM.tif', '20050129_TM.tif', '20050302_TM.tif', '20050214_TM.tif']
        else:
            self.ground_truth_dir = "/home/zbl/datasets_paper/CIA-temp/refs/"
            self.predict_dir = save_dir_name
            self.predict_img_names = ['PRED_2002_076_0317-2002_092_0402.tif', 'PRED_2002_092_0402-2002_101_0411.tif',
                                      'PRED_2002_101_0411-2002_108_0418.tif', 'PRED_2002_108_0418-2002_117_0427.tif',
                                      'PRED_2002_117_0427-2002_124_0504.tif']
            self.ref_img_names = ['20020402_TM.tif', '20020411_TM.tif', '20020418_TM.tif', '20020427_TM.tif',
                                  '20020504_TM.tif']
