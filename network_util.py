#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 19:23:54 2020

@author: jd
contact: jamesduv@umich.edu
affiliation: University of Michigan, Department of Aerospace Eng., CASLAB
"""
import tensorflow as tf
import numpy as np

def swish(x):
    return (x*tf.keras.activations.sigmoid(x))

def sin_act(x):
    actval = tf.math.sin(x)
    return actval
def sin_act2(x, omega_0=30):
    actval = tf.math.sin(omega_0 * x)
    return actval

tf.keras.utils.get_custom_objects().update({'swish': swish})
tf.keras.utils.get_custom_objects().update({'sin_act': sin_act})
tf.keras.utils.get_custom_objects().update({'sin_act2': sin_act2})
tf.keras.utils.get_custom_objects().update({'swish_fn': swish})

def set_kernel_regularizer(opt):
    '''Set the kernel regularizer using options defined in opt
    
    Args:
        opt (dict) : settings for constructing and training a model
        
    Returns:
        kernel_regularizer (tf.keras.regularizers) :'''

    regularizers = {'none':tf.keras.regularizers.l2(l=0),
                    'l1':tf.keras.regularizers.l1(l=opt['lambda_l1']),
                    'l2':tf.keras.regularizers.l2(l=opt['lambda_l2']),
                    'l1_l2':tf.keras.regularizers.l1_l2(l1=opt['lambda_l1'],
                                                        l2=opt['lambda_l2'])}
    if opt['kernel_regularizer'] not in regularizers.keys():
            raise Exception('Unsupported kernel regularizer specified: {}'.format(opt['kernel_regularizer']))
    kernel_regularizer = regularizers[opt['kernel_regularizer']]
    return kernel_regularizer

def set_activation(opt):
    '''Set the activation function using options defined in opt'''

    activations = {'elu':tf.keras.activations.elu,
                   'hard_sigmoid':tf.keras.activations.hard_sigmoid,
                   'linear':tf.keras.activations.linear,
                   'relu':tf.keras.activations.relu,
                   'selu':tf.keras.activations.selu,
                   'sigmoid':tf.keras.activations.sigmoid,
                   'softmax':tf.keras.activations.softmax,
                   'tanh':tf.keras.activations.tanh,
                   'swish':swish,
                   'sin_act':sin_act,
                   'sin_act2':sin_act2}
    if opt['activation'] not in activations.keys():
            raise Exception('Unsupported activation function specified: {}'.format(opt['activation']))
    activation = activations[opt['activation']]
    return activation

def set_optimizer(opt):
    '''Set the activation function using options defined in opt'''

    optimizers = {'Adadelta':tf.keras.optimizers.Adadelta,
                  'Adagrad':tf.keras.optimizers.Adagrad,
                  'Adam':tf.keras.optimizers.Adam,
                  'Adamax':tf.keras.optimizers.Adamax,
                  'Ftrl':tf.keras.optimizers.Ftrl,
                  'Nadam':tf.keras.optimizers.Nadam,
                  'RMSprop':tf.keras.optimizers.RMSprop,
                  'SGD':tf.keras.optimizers.SGD}
    if opt['optimizer'] not in optimizers.keys():
        raise Exception('Unsupported optimizer specified: {}'.format(opt['optimizer']))
    optimizer = optimizers[opt['optimizer']](learning_rate = opt['learning_rate'], **opt['optimizer_kwargs'])
    return optimizer

def set_loss(opt):
    '''Set the loss function using options defined in self.opt'''

    losses = {'KLD':tf.keras.losses.KLD,
              'MAE':tf.keras.losses.MAE,
              'MAPE':tf.keras.losses.MAPE,
              'MSE':tf.keras.losses.MSE,
              'MSLE':tf.keras.losses.MSLE,
              'binary_crossentropy':tf.keras.losses.binary_crossentropy,
              'categorical_crossentropy':tf.keras.losses.categorical_crossentropy,
              'categorical_hinge':tf.keras.losses.categorical_hinge}
    if opt['loss'] not in losses.keys():
        raise Exception('Unsupported loss function specified: {}'.format(opt['loss']))
    loss = losses[opt['loss']]
    return loss