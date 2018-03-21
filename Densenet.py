
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np

class densenet:
    def __init__(self, sess, params, name):
    
        # 하이퍼파라미터
        self.num_classes = params['num_classes']
        self.start_n_filters = params['start_n_filters']
        self.start_strides = params['start_strides']
        self.growth_rate = params['growth_rate']
        self.n_blocks = params['n_blocks']
        self.is_bottlenecks = params['is_bottlenecks']
        self.theta = params['theta']
        self.layer_number = range(1, len(self.n_blocks) + 1)
        self.keep_probs = params['keep_probs']
        
        self.momentum = params['momentum']
        self.learning_rate = params['learning_rate']

        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        
        self.height = params['height']
        self.width = params['width']
        self.model_path = params['model_path']
        self.name = name
    #residual learning
    # f(x) + x
    # original resnet : 본래 분기되기전에 활성화 함수 적용 
    # densenet :  더하지 않고 concat시켜서 모두 보전

    # batch_norm + 활성화 함수
    def batch_norm_relu(self, inputs, is_training, reuse, name):
        _BN = tf.layers.batch_normalization(inputs, momentum = self.momentum, training= is_training, reuse = reuse, name= name)
        outputs = tf.nn.relu(_BN)
        return outputs

    #Growth rate
    #Layer 의 channel 수
    #너무 wider 하는거 방지, 파라미터 efficiency 를 위해, k = 12 로 제한.
    #실험 해보니 생각보다 작은 k 로 충분한 결과 얻음
    # 보통 bottle_neck을 많이쓰는 이유는 보통 input이 growth_rate보다 높기때문에 계산효율이 떨어지므로
    # conv 1x1 로 4* growth_rate만큼 뻥튀기해서 계산 효율 증가
    # densnet의 블럭   
    # composite_layer [BN --> ReLU -->conv33]
    # composite_layer with bottleneck[BN --> ReLU --> conv1x1 (4k로 뻥튀기) --> _BN --> ReLU -->conv3x3]
    # output n_channel = inputs n_channel + growth_rate (growth_rate만큼 composite_layer를 지나갈때마다 증가)
    def composite_layer(self, inputs, keep_prob, name, is_bottleneck = True, is_training = True, reuse = False):
        L = inputs
        if is_bottleneck:
            L = self.batch_norm_relu(L, is_training, reuse, name = name + '_BN1')
            L = tf.layers.conv2d(L, 4 * self.growth_rate, 1, 1, padding = 'SAME', name = name + '_conv1', reuse = reuse)
        L = self.batch_norm_relu(L, is_training, reuse, name = name + '_BN2')
        L = tf.layers.conv2d(L, self.growth_rate, 3, 1, padding = 'SAME', name = name + '_conv2', reuse = reuse)   
        L = tf.layers.dropout(L, keep_prob, training = is_training)
        return tf.concat([inputs, L], axis = 3) 


    # [BN --> conv11 --> avg_pool2]
    # transition_layer
    # feature맵 크기 줄여주고 compression factor theta에 따라 압축
    def transition_layer(self, inputs, name, is_training = True, reuse= False):
        shape = inputs.get_shape().as_list()
        #compression -> 기존 채널수 * theta(0~1)
        n_filters = int(shape[3] * self.theta)
        L = self.batch_norm_relu(inputs, is_training, reuse, name = name + '_BN')
        L = tf.layers.conv2d(L, n_filters, 1, 1, padding = 'SAME', name = name + '_conv', reuse = reuse)
        L = tf.layers.average_pooling2d(L, 2, 2, name = 'pool')

        return L

    # dense block 생성
    def dense_block(self, inputs, name, n_block, keep_prob, is_training = True, reuse = False):
        L = inputs
        for i in range(n_block):
            L = self.composite_layer(L,
                                     keep_prob,
                                     name = name +'_bottle_neck'+str(i),
                                     is_bottleneck = True,
                                     is_training = is_training,
                                     reuse = reuse)

        return L

    def get_logits(self, inputs, is_training = True, reuse = False):

        # 첫번째는 먼저 conv 3x3 시행 128x32 -> 64 x 16 
        L = tf.layers.conv2d(inputs = inputs,
                             filters = self.start_n_filters,
                             kernel_size = 3,
                             strides = self.start_strides,
                             padding = 'SAME',
                             name = 'begin_conv',
                             reuse = reuse)
        # 크기가 큰경우는 max_pooling을 거치나 여기는 그렇게 크지않으므로 제외
        
        for i, n_block, is_bottleneck, keep_prob in zip(self.layer_number, self.n_blocks, self.is_bottlenecks, self.keep_probs):
            L = self.dense_block(L,
                                 name = 'block'+str(i),
                                 n_block = n_block,
                                 keep_prob=keep_prob,
                                 is_training = is_training,
                                 reuse = reuse)
            # 마지막이 아니면 transitionlayer
            if i < len(self.layer_number):
                L = self.transition_layer(L,
                                          name = 'transition'+str(i),
                                          is_training = is_training,
                                          reuse = reuse)
        
        #마지막 batch_norm and activation function
        outputs = self.batch_norm_relu(L, is_training, reuse, name = 'last__BN')
        # global average_pooling
        # shape : (batch_size, height, width, n_feature_map)
        shape = outputs.get_shape().as_list()
        # 글로벌 풀링 사이즈 (height, width)
        pool_size = (shape[1], shape[2])
        outputs= tf.layers.average_pooling2d(outputs, pool_size = pool_size, strides = 1, padding = 'VALID')
        # 마지막 dense layer
        outputs = tf.layers.flatten(outputs)
        outputs = tf.layers.dense(outputs, 10, name = 'final_dense', reuse=reuse)
        return outputs

