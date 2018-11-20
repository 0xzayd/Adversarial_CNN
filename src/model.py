import glob
import re
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import os
from .architectures import SegmentationNet, AdversarialNet
from .trainvaltensorboard import TrainValTensorBoard
from .utils import mean_iou
from .utils import make_trainable
import numpy as np

class CNN2DModel:
    
    def __init__(self, num_gpus = 1):
        self.img_height = 256
        self.img_width = 256
        self.seg_model = None
        self.adv_model = None
        self.adv_seg_model = None
        self.callbackList = None
        self.initial_epoch = 0
        self.num_gpus = num_gpus
        self.num_bands = 4
        self.path = './model'
        self.model_type = None
        
    
    def build_SegmentationNet(self, k_size = (3, 3),
                           blocks=[64, 64, 64, 64],
                           k_init='lecun_normal',
                           activation='selu',
                           lr=1e-3,
                           verbose=False):
        
        self.model_type = 'Segmentation'
        
        if K.image_data_format() == 'channels_first':
            self.input_img_shape = (self.num_bands, self.img_height, self.img_width)
        if K.image_data_format() == 'channels_last':
            self.input_img_shape = (self.img_height, self.img_width, self.num_bands)    
        
        img_inp = Input(self.img_input_shape,name='image_input')
        
        self.seg_model = SegmentationNet(img_inp,
                                     self.num_bands,
                                     blocks, k_size, k_init, activation,'segmentation')
        self.seg_model.compile(adam(lr=lr),
                           loss='binary_crossentropy',
                           metrics=[mean_iou])
        
        if verbose:
            print('Adversarial Net Summary:')
            print(self.seg_model.summary())

    
    def build_AdvSegNet(self, k_size = (3, 3),
                        seg_blocks=[64, 64, 64, 64],
                        adv_blocks=[64, 64, 64],
                        num_adv_kernels=64,
                        k_init='lecun_normal',
                        activation='selu',
                        scale = 1e-1,
                        seg_lr=1e-3,
                        adv_lr=1e-3,
                        verbose=True):
                
        if K.image_data_format() == 'channels_first':
            self.input_img_shape = (self.num_bands, self.img_height, self.img_width)
            self.input_label_shape = (1, self.img_height, self.img_width)   
        if K.image_data_format() == 'channels_last':
            self.input_img_shape = (self.img_height, self.img_width, self.num_bands)
            self.input_label_shape = (self.img_height, self.img_width, 1)
		
        img_inp = Input(self.input_img_shape)
        label_inp = Input(self.input_label_shape)
        
        self.model_type = 'AdvSeg'
        
        adv_optimizer = adam(lr=adv_lr)
        seg_optimizer = adam(lr=seg_lr)
        
        #Building only the adversarial model
        
        self.adv_model = AdversarialNet(img_inp, 
                                        label_inp, 
                                        adv_blocks, 
                                        k_size, 
                                        k_init, 
                                        activation, 
                                        num_adv_kernels,
                                        'adv_model')
        
        # freeze the model and create the frozen instance of it
        make_trainable(self.adv_model, False)
        frozen_adv = Model(inputs=self.adv_model.inputs,
                           outputs=self.adv_model.outputs,
                           name='frozen_adv_model')
        
        frozen_adv.compile(adv_optimizer, 
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        
        #Building the segmentation model
        self.seg_model = SegmentationNet(inp=img_inp,
                                      num_bands=self.num_bands,
                                      blocks=seg_blocks,
                                      filter_size=k_size,
                                      filter_initializer=k_init,
                                      activation=activation,
                                      name='seg_model')
        
        self.seg_model.compile(seg_optimizer,
                               loss='binary_crossentropy',
                               metrics=[mean_iou])
        
        if verbose:
            print('summary of {0}:'.format(self.seg_model.name))
            print(self.seg_model.summary())
            
        # get the prediction from the segmentation
        pred = self.seg_model(img_inp)
        
        #get the probability of the predicted label being not fake
        prob = frozen_adv([img_inp,pred])
        
        self.adv_seg_model = Model(inputs=[img_inp, label_inp],
                                   outputs=[pred,prob],
                                   name='adv_seg_model')
        
        self.adv_seg_model.compile(seg_optimizer,
                                   loss=['binary_crossentropy',
                                         'binary_crossentropy'],
                                   loss_weights=[1.,scale],
                                   metrics=['accuracy'])
        
        if verbose:
            print('summary of {0}:'.format(self.adv_seg_model.name))
            print(self.adv_seg_model.summary())
        
        # unfreeze the adversarial model and compile it
        make_trainable(self.adv_model, True)
        self.adv_model.compile(adv_optimizer,
                               loss='binary_crossentropy',
                               metrics=['accuracy'])
        
        if verbose:
            print('summary of {0}:'.format(self.adv_model.name))
            print(self.adv_model.summary())

############## fit model
            
    def fit_model(self, X_train, Y_train, verbose = 1, validation_split=0.2, batch_size=32, use_tfboard=True,
                  adv_epochs=10, adv_steps_per_epoch=10, seg_epochs=10, seg_steps_per_epoch=10,
                  num_rounds=1):
        
        
        print('fitting model {0}'.format(self.model_type))
        
        if self.model_type == 'Segmentation':
            callbackl = self.build_callbackList(use_tfboard=use_tfboard, monitor='val_mean_iou') 
            self.model.fit(x=X_train,y=Y_train, verbose=verbose, validation_split=validation_split,
                          batch_size=batch_size, epochs=self.initial_epoch + seg_epochs, steps_per_epoch=seg_steps_per_epoch, callbacks=callbackl,
                          initial_epoch=self.initial_epoch)
            
        elif self.model_type == 'AdvSeg':
            adv_callbackl = self.build_callbackList(use_tfboard=use_tfboard,
                                 phase='AdversarialNet') 
            seg_callbackl = self.build_callbackList(use_tfboard=use_tfboard,
                                 monitor='val_seg_model_mean_iou', 
                                 phase='SegmentationNet') 
            
            # changing the input data for each model
            # Input data for the segmentor Segmentor
            y1 = np.ones([Y_train.shape[0], 1])
            X_seg_train = [X_train, Y_train]
            Y_seg_train = [Y_train, y1]
            
            # Input for the Adversarial
            pred = self.seg_model.predict(X_train)
            XX = np.concatenate([X_train, X_train], axis=0)
            YY = np.concatenate([Y_train, pred], axis=0)
            y1 = np.ones([Y_train.shape[0], 1])
            y0 = np.zeros([Y_train.shape[0], 1])
            prob = np.concatenate([y1, y0], axis=0)
            X_adv_train = [XX, YY]
            Y_adv_train = prob
            
            
            for i in range(num_rounds):
                print('round {0}: fitting seg_model'.format(i))
                self.adv_seg_model.fit(x=X_seg_train, y=Y_seg_train, validation_split=validation_split,
                                       verbose=verbose, epochs=(i+1)*seg_epochs,steps_per_epoch=seg_steps_per_epoch, callbacks=seg_callbackl,
                                       initial_epoch=i*seg_epochs)
                print('round {0}: fitting the adversarial model'.format(i))
                self.adv_model.fit(x=X_adv_train, y=Y_adv_train, validation_split=validation_split,
                                       verbose=verbose, epochs=(i+1)*adv_epochs,steps_per_epoch=adv_steps_per_epoch, callbacks=adv_callbackl,
                                       initial_epoch=i*adv_epochs)
                
                
                #do i need to change from sigmoid to softmax ??????????
                
    def build_callbackList(self, use_tfboard=True, monitor = None, phase = None, save=True):        
        if self.model_type == None:
            raise ValueError('model is not built yet, please build Segmentation or AdvSeg')
        else:
            path = './{0}'.format(self.model_type)

        # Model Checkpoints
        if monitor is None:
            callbackList = []
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            filepath=path+'/weights-{epoch:02d}-{'+'{0}'.format(monitor)+':.2f}.hdf5'
            checkpoint = ModelCheckpoint(filepath,
                                         monitor=monitor,
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='max')

            # Bring all the callbacks together into a python list
            callbackList = [checkpoint]
                    
        # Tensorboard
        if use_tfboard:
            if phase is None:
                tfpath = './logs/{0}'.format(self.model_type)
            else:
                tfpath = './logs/{0}/{1}'.format(self.model_type, phase)
            tensorboard = TrainValTensorBoard(log_dir=tfpath)
            callbackList.append(tensorboard)
        return callbackList

        
    def load_checkpoint(self):
        if self.model_type == None:
            raise ValueError('model is not built yet, please build Segmentation or AdvSeg!')
        else:
            path = './model/{0}'.format(self.model_type)
        try:
            checkfile = sorted(glob.glob(path+"/weights-*-*.hdf5"))[-1]
            self.model.load_weights(checkfile)
            self.initial_epoch = int(re.search(r"weights-(\d*)-", checkfile).group(1))
            print("{0} weights loaded, resuming from epoch {1}".format(self.model_type, self.initial_epoch))
        except IndexError:
            try:
                self.model.load_weights(path+"/model-weights.hdf5")
                print("{0} weights loaded, starting from epoch {1}".format(self.model_type, self.initial_epoch))
            except OSError:
                pass

    def save_weights(self, suffix='model-1'):
        if self.model_type == None:
            raise ValueError('model is not built yet, please build model')
        else:
            path = './model/{0}'.format(self.model_type)

        filepath = path+'/{0}.hdf5'.format(suffix)
        self.adv_seg_model.save_weights(filepath=filepath)
        return
    
    
    def load_weights(self, filepath):
        if self.model_type == 'Segmentation':
            self.seg_model.load_weights(filepath=filepath)
        elif self.model_type == 'AdvSeg':
            self.adv_seg_model.load_weights(filepath=filepath)
        else:
            raise ValueError('model is not built yet')
        
    def predict(self, X_tst, verbose=1):
        
        return self.seg_model.predict(X_tst, verbose=verbose) 