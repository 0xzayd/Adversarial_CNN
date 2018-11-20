from keras import backend as K
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, AlphaDropout, Flatten, Dense
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import multi_gpu_model

	
def SegmentationNet(inp = None, num_bands = 4, blocks = [64,64,64,64], filter_size = (3,3), filter_initializer = 'lecun_normal', activation = 'selu', name = None):
    
    if K.image_data_format() == 'channels_first':
        ch_axis = 1
    if K.image_data_format() == 'channels_last':
        ch_axis = 3

    encoder = inp
    list_encoders = []
	
    print('building The segmentation model ...')
    
    print(blocks)
    # Encoding
    for block_id , n_block in enumerate(blocks):
        with K.name_scope('Encoder_block_{0}'.format(block_id)):
            encoder = Conv2D(filters = n_block, 
                             kernel_size = filter_size, 
                             activation = activation, 
                             padding = 'same', 
                             kernel_initializer = filter_initializer)(encoder)
            encoder = AlphaDropout(0,1*block_id, )(encoder)
            
            encoder = Conv2D(filters = n_block, 
                    kernel_size = filter_size, 
                    dilation_rate = (2,2), 
                    activation = activation, 
                    padding='same', 
                    kernel_initializer = filter_initializer)(encoder)
            list_encoders.append(encoder)
			# maxpooling 'BETWEEN' every 2 blocks
            if block_id < len(blocks)-1:
                encoder = MaxPooling2D(pool_size = (2,2))(encoder)
    print('decoder')
	# Decoding
    decoder = encoder
    decoder_blocks = blocks[::-1][1:]
    for block_id, n_block in enumerate(decoder_blocks):
        with K.name_scope('Decoder_block_{0}'.format(block_id)):
            block_id_inv = len(blocks) - 1 - block_id
            decoder = concatenate([decoder, list_encoders[block_id_inv]], axis = ch_axis)
            
            decoder = Conv2D(filters=n_block, 
                             kernel_size = filter_size, 
                             activation = activation, 
                             padding = 'same', dilation_rate = (2,2), 
                             kernel_initializer = filter_initializer)(decoder)
            decoder = AlphaDropout(0,1*block_id, )(decoder)
            
            decoder = Conv2D(filters=n_block, 
                             kernel_size = filter_size, 
                             activation = activation, 
                             padding = 'same',
                             kernel_initializer = filter_initializer)(decoder)
            
            decoder = Conv2DTranspose(filters=n_block, 
                                      kernel_size = filter_size, 
                                      kernel_initializer = filter_initializer, 
                                      padding='same', 
                                      strides=(2,2))(decoder)
			
	# Last Layer...
    outp = Conv2DTranspose(filters=1, kernel_size = filter_size, 
                        activation = 'sigmoid', 
                        padding = 'same', 
                        kernel_initializer = 'glorot_normal')(decoder)
    
    return Model(inputs=[inp], outputs=[outp], name=name)


def AdversarialNet(inpX, inpY, adv_blocks, k_size, k_init, activation, num_adv_kernels,name):
    
    print('building Adversarial convolutional net ...')

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
    if K.image_data_format() == 'channels_last':
         ch_axis = 3
    
    with K.name_scope('AdversarialNet'):
        
        with K.name_scope('img_input_conv'):
            X = Conv2D(filters=num_adv_kernels,
                       kernel_size=k_size,
                       activation=activation,
                       padding='same',
                       kernel_initializer=k_init)(inpX)
        with K.name_scope('label_input_conv'):
            Y = Conv2D(filters=num_adv_kernels,
                       kernel_size=k_size,
                       activation=activation,
                       padding='same',
                       kernel_initializer=k_init)(inpY)
        
        encoder = concatenate([X, Y], axis=ch_axis)   # concatenate according to the channel axis
        for l_idx, n_ch in enumerate(adv_blocks):
            with K.name_scope('AdvNet_block_{0}'.format(l_idx)):
                encoder = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 activation=activation,
                                 padding='same',
                                 kernel_initializer=k_init)(encoder)
                # encoder = AlphaDropout(0.1*l_idx, )(encoder)
                # add maxpooling layer except the last layer
                if l_idx < len(adv_blocks) - 1:
                    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
        encoder = Flatten()(encoder)
        outp = Dense(1, activation='sigmoid')(encoder)

    return Model(inputs=[inpX, inpY], outputs=outp,name=name)