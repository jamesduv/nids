#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jd
contact: jamesduv@umich.edu
affiliation: University of Michigan, Department of Aerospace Eng., CASLAB
"""

import pickle, os, random
import tensorflow as tf
import numpy as np

import network_util as nu
import util

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

class nids_v2():
    '''Version 2 of baseline NIDS model.

    Set up to work either with tf.keras.losses or using custom training
    loop as required for certain regularizations.

    Args:
        opt (dict)  : setting to define and train model
        data (dict) : optional, used to train the model. Not required if not training

    Attributes:
        MODELS (tf.keras.Model):
            net1 : mu -> w      (parameter network)
            net2 : x -> h       (spatial network)
    '''

    def __init__(self, opt, data=None):
        self.opt    = opt
        self.data   = data

        #set loss - only used directly when self.train_model is used (not with self.train_model_batch_by_case)
        self.loss           = nu.set_loss(opt)
        self.recon_loss     = nu.set_loss(opt)


        #set kernel_regularizer, acitvation function(s), and optimizer
        self.kernel_regularizer     = nu.set_kernel_regularizer(opt)

        self.activation_net1        = nu.set_activation(opt['net1'])
        self.activation_net2        = nu.set_activation(opt['net2'])
        self.optimizer              = nu.set_optimizer(opt)

        #filenames for saving training loss and weights
        self.fn_train = os.path.join(self.opt['save_dir'], 'training.csv')

        #best/end weights
        self.fn_weights_net1_best   = os.path.join(self.opt['save_dir'],
                                                   'weights.net1.best.h5')
        self.fn_weights_net1_end    = os.path.join(self.opt['save_dir'],
                                                   'weights.net1.end.h5')
        self.fn_weights_net2_best   = os.path.join(self.opt['save_dir'],
                                                   'weights.net2.best.h5')
        self.fn_weights_net2_end    = os.path.join(self.opt['save_dir'],
                                                   'weights.net2.end.h5')

        self.fn_weights_best        = {'net1':self.fn_weights_net1_best,
                                       'net2':self.fn_weights_net2_best}
        self.fn_weights_end         = {'net1':self.fn_weights_net1_end,
                                       'net2':self.fn_weights_net2_end}

        #every X epochs weights
        if self.opt['is_save_all_weights']:
            weight_dir     = os.path.join(self.opt['save_dir'], 'weights')
            if not os.path.exists(weight_dir):
                os.mkdir(weight_dir)
            self.opt['weight_dir'] = weight_dir
            self.fn_weights_net1_epoch = os.path.join(self.opt['weight_dir'],
                                                      'weights.net1.EPOCH.h5')
            self.fn_weights_net2_epoch = os.path.join(self.opt['weight_dir'],
                                                      'weights.net2.EPOCH.h5')

            self.fn_weights_epoch   = {'net1':self.fn_weights_net1_epoch,
                                       'net2':self.fn_weights_net2_epoch}

        self.build_model()

    def siren_weight_init_v2(self, omega_0=30, c=6):
        '''SIREN weight initialization scheme 
        TODO: Change biases to zeros instead of uniform'''

        #parameter network
        if self.opt['net1']['is_siren_init']:
            old_weights = []
            new_weights_tar = []
            for idx, name in enumerate(self.net1_layers):
                #Input layer - no weights
                if idx == 0:
                    continue
    
                #First dense layer with implicit frequency mult.
                elif idx == 1:
                    n   = self.opt['net1']['n_input']
                    high    = 1 / n
                    low     = -1 * high
    
                #All other layers
                else:
                    n   = self.opt['net1']['n_nodes'][idx-2]
                    high    = np.sqrt(c/n) / omega_0
                    low     = -1 * high
    
                layer   = self.net1.get_layer(name = name)
                weights = layer.get_weights()
                old_weights.append(weights)
                wt_ls = []
                for iWeight, wt in enumerate(weights):
                    shape = wt.shape
                    wt_new = np.random.uniform(low = low,
                                               high = high,
                                               size = shape)
                    wt_ls.append(wt_new)
                layer.set_weights(wt_ls)
                new_weights_tar.append(wt_ls)

        #spatial network
        if self.opt['net2']['is_siren_init']:
            old_weights = []
            new_weights_tar = []
            for idx, name in enumerate(self.net2_layers):
                #Input layer - no weights
                if idx == 0:
                    continue
    
                #First dense layer with implicit frequency mult.
                elif idx == 1:
                    n   = self.opt['net2']['n_input']
                    high    = 1 / n
                    low     = -1 * high
    
                #All other layers
                else:
                    n   = self.opt['net2']['n_nodes'][idx-2]
                    high    = np.sqrt(c/n) / omega_0
                    low     = -1 * high
    
                layer   = self.net2.get_layer(name = name)
                weights = layer.get_weights()
                old_weights.append(weights)
                wt_ls = []
                for iWeight, wt in enumerate(weights):
                    shape = wt.shape
                    wt_new = np.random.uniform(low = low,
                                               high = high,
                                               size = shape)
                    wt_ls.append(wt_new)
                layer.set_weights(wt_ls)
                new_weights_tar.append(wt_ls)

    def call_model(self, x, training=False):
        '''Call the full model, return only the final result'''

        net1_output     = self.net1(x[0], training=training)
        weight, bias    = tf.split(net1_output, num_or_size_splits=self.w_dim_ls, axis=1)
        weight          = tf.reshape(weight, (-1, self.opt['net2']['n_output'], self.opt['net2']['n_nodes'][-1]))
        hx              = self.net2(x[1], training=training)
        y_pred          = tf.einsum('ijk,ik->ij', weight, hx) + bias
        return y_pred

    def call_model_verbose(self, x, training=False):
        '''Call the full model, return final and intermediate results'''
        net1_output     = self.net1(x[0], training=training)
        weight, bias    = tf.split(net1_output, num_or_size_splits=self.w_dim_ls, axis=1)
        weight          = tf.reshape(weight, (-1,self.opt['net2']['n_output'],self.opt['net2']['n_nodes'][-1]))
        hx              = self.net2(x[1], training=training)
        y_pred          = tf.einsum('ijk,ik->ij', weight, hx) + bias
        return y_pred, weight, bias, hx

    def call_net1(self, x, training=False):
        '''Call net1 (parameter network) only.'''
        net1_output     = self.net1(x, training=training)
        weight, bias    = tf.split(net1_output, num_or_size_splits=self.w_dim_ls, axis=1)
        weight          = tf.reshape(weight, (-1, self.opt['net2']['n_output'], self.opt['net2']['n_nodes'][-1]))
        return weight, bias

    def call_model_net1sep(self, x, weight, bias, training=False):
        '''Call the full model when the outputs of the parameter network are provided, return final result'''
        hx      = self.net2(x, training=training)
        y_pred  = tf.einsum('ijk,lk->lj', weight, hx) + bias
        return y_pred

    def call_model_net1sep_verbose(self, x, weight, bias, training=False):
        '''Call the full model when the outputs of the parameter network are provided, return final and intermediate result'''
        hx      = self.net2(x, training=training)
        y_pred  = tf.einsum('ijk,lk->lj', weight, hx) + bias
        return y_pred, hx

    def train_model(self):
        '''Custom training loop - train model with validation points using tf.keras.losses loss'''

        self.create_training_log_val()
        # x2_val      = self.data['x2_val']
        # x_val       = self.data['x_val']
        # y_val       = self.data['y_val']

        n_train = self.data['x_train'].shape[0]
        n_val   = self.data['x_val'].shape[0]

        buffer_size   = 5 * self.opt['batch_size']

        #create dataset for validation data, create batches
        val_data  = tf.data.Dataset.from_tensor_slices((self.data['x2_val'],
                                                          self.data['x_val'],
                                                          self.data['y_val']))
        val_data  = val_data.batch(batch_size = self.opt['batch_size'])

        #create the training dataset, shuffle, and create batches
        train_data  = tf.data.Dataset.from_tensor_slices((self.data['x2_train'],
                                                          self.data['x_train'],
                                                          self.data['y_train']))

        train_data  = train_data.shuffle(buffer_size = buffer_size, reshuffle_each_iteration=True)
        train_data  = train_data.batch(batch_size = self.opt['batch_size'])
        n_epochs    = self.opt['epochs']

        training_loss           = []
        val_loss_store          = []
        val_best_store          = []
        for epoch in range(n_epochs):

            #gradient descent over mini-batches
            batch_loss = []
            for batch, (x2, x, y) in enumerate(train_data):
                cur_loss = self.train_on_batch([x2,x],y)
                cur_loss = tf.reduce_sum(cur_loss)
                batch_loss.append(cur_loss)

            mean_loss = np.sum(np.array(batch_loss)) / n_train
            training_loss.append(mean_loss)

            #validation loss
            # y_val_pred      = self.call_model([x2_val, x_val])
            # val_loss_cur    = np.mean(self.loss(y_val_pred, y_val))
            # val_loss_store.append(val_loss_cur)

            #validation loss - use batching to avoid OOM
            batch_loss_val = []
            for batch, (x2, x, y) in enumerate(val_data):
                y_val_pred      = self.call_model([x2, x])
                cur_loss_val    = self.loss(y_val_pred, y)
                cur_loss_val    = tf.reduce_sum(cur_loss_val)
                batch_loss_val.append(cur_loss_val)

            val_loss_cur = np.sum(np.array(batch_loss_val)) / n_val
            val_loss_store.append(val_loss_cur)

            #update training log
            self.update_training_log_val(epoch, training_loss[-1],
                                         val_loss_store[-1])

            #print epoch message to console
            self.print_train_val(epoch, n_epochs, training_loss[-1],
                                 val_loss_store[-1])

            #save weights, update best val
            if epoch == 0:
                val_best_cur = val_loss_cur

            if val_loss_store[-1] <= val_best_cur:
                val_best_cur = val_loss_store[-1]
                message = 'Saving best weights'
                print(message)
                self.save_weights(self.fn_weights_best)

            val_best_store.append(val_best_cur)
            if np.mod(epoch, self.opt['n_epochs_save']) == 0:
                message = 'Saving end weights'
                print(message)
                self.save_weights(self.fn_weights_end)
                self.save_weights_epoch(epoch)

    @tf.function
    def train_on_batch(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.call_model(x, training=True)
            loss_value = self.loss(y_pred, y)

        grads = tape.gradient(loss_value, tape.watched_variables())
        self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        return loss_value

    def shuffle_training_cases(self):
        ''' shuffle training data list, using random.shuffle'''
        random.shuffle(self.data['train_data'])

    def train_model_batch_by_case(self):
        '''Custom training loop, for efficient net evals. The training data is
        stored as a lists of lists of triples (x2, x, y) in net.data. Shuffle this list in 
        place after each epoch to avoid copying the data'''

        self.create_training_log_val()

        n_epochs    = self.opt['epochs']
        n_train     = self.opt['n_data_train']
        n_val       = self.opt['n_data_val']

        training_loss           = []
        val_loss_store          = []
        val_best_store          = []

        for epoch in range(n_epochs):

            #training: gradient descent over mini-batches
            batch_loss = []
            for iBatch, (x2, x, y) in enumerate(self.data['train_data']):
                cur_loss = self.train_on_batch_mb(x2, x, y)
                cur_loss = tf.reduce_sum(cur_loss)
                batch_loss.append(cur_loss)

            mean_loss = np.sum(np.array(batch_loss)) / n_train
            training_loss.append(mean_loss)
            self.shuffle_training_cases()

            #validation loss
            batch_loss_val = []
            for (x2, x, y) in self.data['val_data']:
                weight, bias    = self.call_net1(x2)
                y_pred          = self.call_model_net1sep(x, weight, bias, training=False)
                cur_loss_val    = self.loss(y_pred, y)
                cur_loss_val    = tf.reduce_sum(cur_loss_val)
                batch_loss_val.append(cur_loss_val)

            val_loss_cur = np.sum(np.array(batch_loss_val)) / n_val
            val_loss_store.append(val_loss_cur)

            #update training log
            self.update_training_log_val(epoch, training_loss[-1],
                                          val_loss_store[-1])

            #print epoch message to console
            self.print_train_val(epoch, n_epochs, training_loss[-1],
                                  val_loss_store[-1])

            #save weights, update best val
            if epoch == 0:
                val_best_cur = val_loss_cur

            if val_loss_store[-1] <= val_best_cur:
                val_best_cur = val_loss_store[-1]
                message = 'Saving best weights'
                print(message)
                self.save_weights(self.fn_weights_best)

            val_best_store.append(val_best_cur)
            if np.mod(epoch, self.opt['n_epochs_save']) == 0:
                message = 'Saving end weights'
                print(message)
                self.save_weights(self.fn_weights_end)
                self.save_weights_epoch(epoch)

    @tf.function
    def train_on_batch_mb(self, x2, x, y):
        with tf.GradientTape() as tape:
            weight, bias    = self.call_net1(x2)
            y_pred          = self.call_model_net1sep(x, weight, bias)
            loss_value      = self.recon_loss(y_pred, y)
            mean_loss       = tf.reduce_mean(loss_value)

        grads = tape.gradient(mean_loss, tape.watched_variables())
        self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        return loss_value

    def create_training_log_val(self):
        '''Create training log, with validation data'''
        f           = open(self.fn_train,'a+')
        f.write(','.join(('epoch','loss','val_loss\n')))
        f.close()

    def update_training_log_val(self, epoch, training_loss, val_loss):
        '''Append results for single epoch to training log'''
        f = open(self.fn_train,'a+')
        f.write(','.join((str(epoch), str(training_loss),
                          str(val_loss)+'\n')))
        f.close()

    def print_train_val(self, epoch, n_epochs, loss, loss_val):
        '''Print message to console during training, with validation data'''
        message = 'Epoch [{:1.0f}/{:1.0f}]: Loss: {:1.4e}, Val_loss: {:1.4e}'\
                    .format(epoch, n_epochs, loss, loss_val)
        print(message)

    def tup_data(self, data):
        '''Turn data organized by case in a dictionary like:
            cases = data.keys()
            data[case].keys() =  'x2','x','y'
        Into a dictionary returning tuples (x2,x,y), where the keys are preserved
        '''

        tup_data = {}
        for case, case_data in data.items():
            tup_data[case] = (case_data['x2'], case_data['x'], case_data['y'])
        return tup_data

    def tup_data_counter_list(self, data):
        '''Turn data organized by case in a dictionary like:
            cases = data.keys()
            data[case].keys() =  'x2','x','y'
        Into a list of tuples (x2,x,y) 
        '''

        tup_data = []
        for iCase, case in enumerate(data.keys()):
            case_data = data[case]
            tup_data.append((case_data['x2'], case_data['x'], case_data['y']))
        return tup_data

    def train_model_batch_by_case_databatches(self):
        '''Train the model using batch by case, training is broken into 
        stages where data is slowly introduced in a 'databatch'

        Training data must be organized by dictionaries instead of arrays
        Minibatches are already created, stored as a list of tuples for each
        level of databatching

        '''
        self.create_training_log_val()
        epoch_schedule  = self.opt['epoch_schedule']
        n_epochs_tot    = np.sum(np.array(epoch_schedule))
        n_data_batches  = len(epoch_schedule)

        training_loss_hist      = []
        val_loss_hist           = []
        val_best_hist           = []
        epoch_ct                = 0
        for iData, n_epochs in enumerate(epoch_schedule):

            train_data  = self.data[iData]['minibatches']
            val_data    = self.tup_data(self.data[iData]['val_data'])

            for epoch in range(n_epochs):
                #gradient descent over each minibatch, shuffle each time
                batch_loss      = []
                random.shuffle(train_data)
                for (x2, x, y) in train_data:
                    loss_value = self.train_on_batch_mb_db(x2, x, y)
                    batch_loss.append(loss_value)
                training_loss_hist.append(np.array(batch_loss).mean())
    
                #validation loss
                batch_loss_val          = []
                for idx, (x2, x, y) in val_data.items():
                    weight, bias    = self.call_net1(x2)
                    y_pred          = self.call_model_net1sep(x, weight, bias, training=False)
                    recon_loss      = self.recon_loss(y_pred, y)
                    loss_value      = tf.reduce_mean(recon_loss)
                    batch_loss_val.append(loss_value)

                val_loss_hist.append(np.array(batch_loss_val).mean())
    
                #update training log
                self.update_training_log_val(epoch_ct, training_loss_hist[-1], val_loss_hist[-1])
    
                #print epoch message to console
                self.print_train_val(epoch_ct, n_epochs_tot, training_loss_hist[-1],
                                      val_loss_hist[-1])
    
                #save weights, update best val
                if epoch_ct == 0:
                    val_best_cur = val_loss_hist[-1]
    
                if val_loss_hist[-1] <= val_best_cur:
                    val_best_cur = val_loss_hist[-1]
                    message = 'Saving best weights'
                    print(message)
                    self.save_weights(self.fn_weights_best)
    
                val_best_hist.append(val_best_cur)
                if np.mod(epoch_ct, self.opt['n_epochs_save']) == 0:
                    message = 'Saving end weights'
                    print(message)
                    self.save_weights(self.fn_weights_end)
                    self.save_weights_epoch(epoch_ct)

                epoch_ct += 1

    @tf.function
    def train_on_batch_mb_db(self, x2, x, y):
        with tf.GradientTape() as tape:
            weight, bias    = self.call_net1(x2)
            y_pred          = self.call_model_net1sep(x, weight, bias)
            recon_loss      = self.recon_loss(y_pred, y)
            loss_value       = tf.reduce_mean(recon_loss)

        grads = tape.gradient(loss_value, tape.watched_variables())
        self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        return loss_value

    def train_model_batch_by_case_databatches_tensorarray(self):
        '''Train the model using batch by case, training is broken into 
        stages where data is slowly introduced in a 'databatch'.

        Use dynamically sized tensor arrays instead of lists to save
        the training indo

        Training data must be organized by dictionaries instead of arrays
        Minibatches are already created, stored as a list of tuples for each
        level of databatching

        '''
        self.create_training_log_val()
        epoch_schedule  = self.opt['epoch_schedule']
        n_epochs_tot    = np.sum(np.array(epoch_schedule))
        n_data_batches  = len(epoch_schedule)


        training_loss_hist      = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True, clear_after_read=False)
        val_loss_hist           = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True, clear_after_read=False)
        # val_best_hist           = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True, clear_after_read=False)
        # val_best_cur            = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True, clear_after_read=False)
        epoch_ct                = 0

        for iData, n_epochs in enumerate(epoch_schedule):

            train_data  = self.data[iData]['minibatches']
            # val_data    = self.tup_data(self.data[iData]['val_data'])
            val_data    = self.tup_data_counter_list(self.data[iData]['val_data'])

            for epoch in range(n_epochs):
                #gradient descent over each minibatch, shuffle each time
                batch_loss      = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
                random.shuffle(train_data)
                for iBatch, (x2, x, y) in enumerate(train_data):
                    loss_value = self.train_on_batch_mb_db(x2, x, y)
                    batch_loss = batch_loss.write(iBatch, loss_value)

                #convert the tensor array to a tensor, compute numeric value for batch loss
                batch_loss = batch_loss.stack()
                batch_loss = tf.reduce_mean(batch_loss)
                training_loss_hist = training_loss_hist.write(epoch_ct, batch_loss)

                #validation loss
                batch_loss_val  = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
                for iBatch, (x2, x, y) in enumerate(val_data):
                    weight, bias    = self.call_net1(x2)
                    y_pred          = self.call_model_net1sep(x, weight, bias, training=False)
                    recon_loss      = self.recon_loss(y_pred, y)
                    loss_value      = tf.reduce_mean(recon_loss)
                    batch_loss_val = batch_loss_val.write(iBatch, loss_value)

                #convert the tensor array to a tensor, compute numeric value for batch loss
                batch_loss_val = batch_loss_val.stack()
                batch_loss_val = tf.reduce_mean(batch_loss_val)
                val_loss_hist = val_loss_hist.write(epoch_ct, batch_loss_val)
    
                #update training log
                ###
                self.update_training_log_val(epoch_ct, training_loss_hist.read(epoch_ct).numpy(), val_loss_hist.read(epoch_ct).numpy())

                #print epoch message to console
                self.print_train_val(epoch_ct, n_epochs_tot, training_loss_hist.read(epoch_ct),
                                      val_loss_hist.read(epoch_ct))

                if np.mod(epoch_ct, self.opt['n_epochs_save']) == 0:
                    message = 'Saving end weights'
                    print(message)
                    self.save_weights(self.fn_weights_end)
                    self.save_weights_epoch(epoch_ct)

                epoch_ct += 1

    def save_weights(self, fn_weights):
        self.net1.save_weights(fn_weights['net1'])
        self.net2.save_weights(fn_weights['net2'])

    def save_weights_epoch(self, epoch):
        self.net1.save_weights(self.fn_weights_epoch['net1'].replace('EPOCH','{:1.0f}'.format(epoch)))
        self.net2.save_weights(self.fn_weights_epoch['net2'].replace('EPOCH','{:1.0f}'.format(epoch)))

    def load_weights(self, fn_weights):
        self.net1.load_weights(fn_weights['net1'])
        self.net2.load_weights(fn_weights['net2'])

    def build_model(self):
        self.layer_names = {}

###### Build net 1
        print('Building net1')
        net1_opt        = self.opt['net1']
        net1_layers     = []

        name            = 'input_net1'
        net1_layers.append(name)
        net1_input      = tf.keras.Input(shape=(net1_opt['n_input'],),
                                         name = name)

        for iDense in range(net1_opt['n_layers']-1):
            print("Hidden Layer %d" % (iDense))
            name        = 'dense_' + str(iDense)
            net1_layers.append(name)
            units       = net1_opt['n_nodes'][iDense]
            activation  = self.activation_net1
            
            if iDense == 0:
                output  = tf.keras.layers.Dense(units = units,
                                                activation = activation,
                                                kernel_regularizer = self.kernel_regularizer,
                                                name = name)(net1_input)
            else:
                output  = tf.keras.layers.Dense(units = units,
                                           activation = activation,
                                           kernel_regularizer = self.kernel_regularizer,
                                           name = name)(output)
            if net1_opt['is_batch_norm']:
                name    = 'batch_norm_' + name
                net1_layers.append(name)
                output  = tf.keras.layers.BatchNormalization(name=name)(output)

        if net1_opt['is_linear_output']:
            activation  = tf.keras.activations.linear
        else:
            activation  = self.activation_net1
        name            = 'net1_output'
        net1_layers.append(name)
        units           = net1_opt['n_nodes'][-1]
        net1_output     = tf.keras.layers.Dense(units = units,
                                       activation = activation,
                                       kernel_regularizer = self.kernel_regularizer,
                                       name = name)(output)

        self.net1_layers = net1_layers
        self.net1 = tf.keras.Model(inputs=[net1_input], outputs=[net1_output])
        self.net1.summary()
        self.w_dim_ls = [self.opt['w_mat_dim'], self.opt['net2']['n_output']]

###### Build net 2
        print('Building net 2')
        net2_opt        = self.opt['net2']
        net2_layers     = []

        print('Input Layer')
        name            = 'net2_input'
        net2_input      = tf.keras.Input(shape=(net2_opt['n_input'],), name=name)

        net2_layers.append(name)
        input1 = tf.keras.Input(shape=(net2_opt['n_input'],), name=name)
        # self.layer_call[name] = input1
        
        for iDense in range(net2_opt['n_layers']):
            print("Hidden Layer %d" % (iDense))
            name = 'dense_' + str(iDense)
            net2_layers.append(name)
            units = net2_opt['n_nodes'][iDense]
            activation = self.activation_net2
            
            if iDense == 0:
                output = tf.keras.layers.Dense(units = units,
                                               activation = activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(net2_input)

            else:
                output = tf.keras.layers.Dense(units = units,
                                               activation = activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(output)
            if net2_opt['is_batch_norm']:
                name = 'batch_norm_' + name
                net2_layers.append(name)
                output = tf.keras.layers.BatchNormalization(name=name)(output)
        hx = output
        self.net2_layers = net2_layers
        self.net2 = tf.keras.Model(inputs=[net2_input], outputs=[output])
        self.net2.summary()

class bnids_v1():
    '''Version 1 of B-NIDS model

    DOES NOT ENFORCE DIRICHLET BCS

    Set up to work either with tf.keras.losses or using custom training
    loop as required for certain regularizations.

    Args:
        opt (dict)  : setting to define and train model
        data (dict) : optional, used to train the model. Not required if not training

    Attributes:
        MODELS (tf.keras.Model):
            net1    : mu -> w
            net2    : x -> h
            net2b   : x -> h2
    '''

    def __init__(self, opt, data=None, beta_filter=1e-16):
        self.opt    = opt
        self.data   = data

        #set loss - only used directly when self.train_model is used (not with self.train_model_batch_by_case)
        self.loss           = nu.set_loss(opt)
        self.recon_loss     = nu.set_loss(opt)

        #set kernel_regularizer, acitvation function(s), and optimizer
        self.kernel_regularizer     = nu.set_kernel_regularizer(opt)

        self.activation_net1        = nu.set_activation(opt['net1'])
        self.activation_net2        = nu.set_activation(opt['net2'])
        self.activation_net2b       = nu.set_activation(opt['net2'])
        self.optimizer              = nu.set_optimizer(opt)

        #filenames for saving training loss and weights
        self.fn_train = os.path.join(self.opt['save_dir'], 'training.csv')

        #best/end weights
        self.fn_weights_net1_best   = os.path.join(self.opt['save_dir'],
                                                   'weights.net1.best.h5')
        self.fn_weights_net1_end    = os.path.join(self.opt['save_dir'],
                                                   'weights.net1.end.h5')
        self.fn_weights_net2_best   = os.path.join(self.opt['save_dir'],
                                                   'weights.net2.best.h5')
        self.fn_weights_net2_end    = os.path.join(self.opt['save_dir'],
                                                   'weights.net2.end.h5')
        self.fn_weights_net2b_best   = os.path.join(self.opt['save_dir'],
                                                   'weights.net2b.best.h5')
        self.fn_weights_net2b_end    = os.path.join(self.opt['save_dir'],
                                                   'weights.net2b.end.h5')

        self.fn_weights_best        = {'net1':self.fn_weights_net1_best,
                                       'net2':self.fn_weights_net2_best,
                                       'net2b':self.fn_weights_net2b_best}
        self.fn_weights_end         = {'net1':self.fn_weights_net1_end,
                                       'net2':self.fn_weights_net2_end,
                                       'net2b':self.fn_weights_net2b_end}

        #every X epochs weights
        if self.opt['is_save_all_weights']:
            weight_dir     = os.path.join(self.opt['save_dir'], 'weights')
            ### UNCOMMENT FOLLOWING AFTER DEBUGGING
            if not os.path.exists(weight_dir):
                os.mkdir(weight_dir)
            self.opt['weight_dir'] = weight_dir
            self.fn_weights_net1_epoch = os.path.join(self.opt['weight_dir'],
                                                      'weights.net1.EPOCH.h5')
            self.fn_weights_net2_epoch = os.path.join(self.opt['weight_dir'],
                                                      'weights.net2.EPOCH.h5')
            self.fn_weights_net2b_epoch = os.path.join(self.opt['weight_dir'],
                                                      'weights.net2b.EPOCH.h5')
            self.fn_weights_epoch   = {'net1':self.fn_weights_net1_epoch,
                                       'net2':self.fn_weights_net2_epoch,
                                       'net2b':self.fn_weights_net2b_epoch}

        #set surface network parameters
        self.beta_filter = beta_filter
        try:
            self.idx_sdf    = opt['net2']['inputs'].index('sdf')
        except:
            self.idx_sdf    = opt['net2']['inputs'].index('sdf_overall')

        if 'is_norm_sigma' in self.opt['net2b'].keys():
            self.norm_sigma()
        else:
            self.sigma      = opt['net2b']['sigma']
        self.net2b_BC   = tf.convert_to_tensor(np.array([0]), dtype=tf.float64)

        self.build_model()

    def norm_sigma(self):
        if self.opt['net2b']['is_norm_sigma']:
            x_stats = self.opt['data_stats']['x'][self.idx_sdf]
            if self.opt['net2']['norm_x_by'] == 'range':
                self.sigma = (self.opt['net2b']['sigma'] - x_stats['min']) / (x_stats['max'] - x_stats['min'])
            elif self.opt['net2']['norm_x_by'] == 'mean_std':
                self.sigma = (self.opt['net2b']['sigma'] - x_stats['mean']) / (x_stats['std'])
        else:
            self.sigma = self.opt['net2b']['sigma']

    def siren_weight_init_v2(self, omega_0=30, c=6):

        #parameter network
        if self.opt['net1']['is_siren_init']:
            old_weights = []
            new_weights_tar = []
            for idx, name in enumerate(self.net1_layers):
                #Input layer - no weights
                if idx == 0:
                    continue
    
                #First dense layer with implicit frequency mult.
                elif idx == 1:
                    n   = self.opt['net1']['n_input']
                    high    = 1 / n
                    low     = -1 * high
    
                #All other layers
                else:
                    n   = self.opt['net1']['n_nodes'][idx-2]
                    high    = np.sqrt(c/n) / omega_0
                    low     = -1 * high
    
                layer   = self.net1.get_layer(name = name)
                weights = layer.get_weights()
                old_weights.append(weights)
                wt_ls = []
                for iWeight, wt in enumerate(weights):
                    shape = wt.shape
                    wt_new = np.random.uniform(low = low,
                                               high = high,
                                               size = shape)
                    wt_ls.append(wt_new)
                layer.set_weights(wt_ls)
                new_weights_tar.append(wt_ls)

        #spatial domain network
        if self.opt['net2']['is_siren_init']:
            old_weights = []
            new_weights_tar = []
            for idx, name in enumerate(self.net2_layers):
                #Input layer - no weights
                if idx == 0:
                    continue
    
                #First dense layer with implicit frequency mult.
                elif idx == 1:
                    n   = self.opt['net2']['n_input']
                    high    = 1 / n
                    low     = -1 * high
    
                #All other layers
                else:
                    n   = self.opt['net2']['n_nodes'][idx-2]
                    high    = np.sqrt(c/n) / omega_0
                    low     = -1 * high
    
                layer   = self.net2.get_layer(name = name)
                weights = layer.get_weights()
                old_weights.append(weights)
                wt_ls = []
                for iWeight, wt in enumerate(weights):
                    shape = wt.shape
                    wt_new = np.random.uniform(low = low,
                                               high = high,
                                               size = shape)
                    wt_ls.append(wt_new)
                layer.set_weights(wt_ls)
                new_weights_tar.append(wt_ls)

        #spatial boundary network
        if self.opt['net2b']['is_siren_init']:
            old_weights = []
            new_weights_tar = []
            for idx, name in enumerate(self.net2b_layers):
                #Input layer - no weights
                if idx == 0:
                    continue
    
                #First dense layer with implicit frequency mult.
                elif idx == 1:
                    n   = self.opt['net2b']['n_input']
                    high    = 1 / n
                    low     = -1 * high
    
                #All other layers
                else:
                    n   = self.opt['net2b']['n_nodes'][idx-2]
                    high    = np.sqrt(c/n) / omega_0
                    low     = -1 * high
    
                layer   = self.net2b.get_layer(name = name)
                weights = layer.get_weights()
                old_weights.append(weights)
                wt_ls = []
                for iWeight, wt in enumerate(weights):
                    shape = wt.shape
                    wt_new = np.random.uniform(low = low,
                                               high = high,
                                               size = shape)
                    wt_ls.append(wt_new)
                layer.set_weights(wt_ls)
                new_weights_tar.append(wt_ls)

    def surface_function(self, x):
        beta = np.exp(-0.5 * (x**2) / (self.sigma**2) )
        beta[beta < self.beta_filter] = 0
        beta = np.expand_dims(beta, axis=1)
        beta_min    = 1 - beta
        beta        = tf.convert_to_tensor(beta)
        beta_min    = tf.convert_to_tensor(beta_min)
        return beta, beta_min

    def surface_function_numpy(self, x):
        beta = np.exp(-0.5 * (x**2) / (self.sigma**2) )
        beta[beta < self.beta_filter] = 0
        return beta

    def call_model(self, x, training=False):
        '''Call the full model, return only the final result'''

        #parameter network
        net1_output     = self.net1(x[0], training=training)
        weight, bias    = tf.split(net1_output, num_or_size_splits=self.w_dim_ls, axis=1)
        weight          = tf.reshape(weight, (-1, self.opt['net2']['n_output'], self.opt['net2']['n_nodes'][-1]))

        #domain network
        hx      = self.net2(x[1], training=training)
        yx      = tf.einsum('ijk,ik->ij', weight, hx) + bias

        #boundary network
        hb      = self.net2b(x[1], training=training)
        yb      = tf.einsum('ijk,ik->ij', weight, hb) + bias

        beta, beta_min  = self.surface_function(x[1][:,self.idx_sdf])
        tb          = tf.multiply(beta, yb)
        td          = tf.multiply(beta_min, yx)
        #weighted overall output
        y_pred = tf.add(td, tb)
        return y_pred

    def call_model_betasep(self, x, training=False):
        '''Call the full model, return only the final result'''

        #parameter network
        net1_output     = self.net1(x[0], training=training)
        weight, bias    = tf.split(net1_output, num_or_size_splits=self.w_dim_ls, axis=1)
        weight          = tf.reshape(weight, (-1, self.opt['net2']['n_output'], self.opt['net2']['n_nodes'][-1]))

        #domain network
        hx      = self.net2(x[1], training=training)
        yx      = tf.einsum('ijk,ik->ij', weight, hx) + bias

        #boundary network
        hb      = self.net2b(x[1], training=training)
        yb      = tf.einsum('ijk,ik->ij', weight, hb) + bias

        beta        = x[2]
        beta_min    = x[3]

        # beta, beta_min  = self.surface_function(x[1][:,self.idx_sdf])
        tb          = tf.multiply(beta, yb)
        td          = tf.multiply(beta_min, yx)
        #weighted overall output
        y_pred = tf.add(td, tb)
        return y_pred

    def call_model_verbose(self, x, training=False):
        '''Call the full model, return final and intermediate results'''

        #parameter network
        net1_output     = self.net1(x[0], training=training)
        weight, bias    = tf.split(net1_output, num_or_size_splits=self.w_dim_ls, axis=1)
        weight          = tf.reshape(weight, (-1, self.opt['net2']['n_output'], self.opt['net2']['n_nodes'][-1]))

        #domain network
        hx      = self.net2(x[1], training=training)
        yx      = tf.einsum('ijk,ik->ij', weight, hx) + bias

        #boundary network
        hb      = self.net2b(x[1], training=training)
        yb      = tf.einsum('ijk,ik->ij', weight, hb) + bias

        beta, beta_min    = self.surface_function(x[1][:,self.idx_sdf])

        tb          = tf.multiply(beta, yb)
        td          = tf.multiply(beta_min, yx)
        #weighted overall output
        y_pred = tf.add(td, tb)
        return y_pred, hx, yx, hb, yb, weight, bias, beta, beta_min

    def call_model_no_bd_bias(self, x, training=False):
        '''Call the full model, return only the final result'''

        #parameter network
        net1_output     = self.net1(x[0], training=training)
        weight, bias    = tf.split(net1_output, num_or_size_splits=self.w_dim_ls, axis=1)
        weight          = tf.reshape(weight, (-1, self.opt['net2']['n_output'], self.opt['net2']['n_nodes'][-1]))

        #domain network
        hx      = self.net2(x[1], training=training)
        yx      = tf.einsum('ijk,ik->ij', weight, hx) + bias

        #boundary network
        hb      = self.net2b(x[1], training=training)
        yb      = tf.einsum('ijk,ik->ij', weight, hb)

        beta, beta_min  = self.surface_function(x[1][:,self.idx_sdf])
        tb          = tf.multiply(beta, yb)
        td          = tf.multiply(beta_min, yx)
        #weighted overall output
        y_pred = tf.add(td, tb)
        return y_pred

    def call_model_no_bd_bias_verbose(self, x, training=False):
        '''Call the full model, return final and intermediate results'''

        #parameter network
        net1_output     = self.net1(x[0], training=training)
        weight, bias    = tf.split(net1_output, num_or_size_splits=self.w_dim_ls, axis=1)
        weight          = tf.reshape(weight, (-1, self.opt['net2']['n_output'], self.opt['net2']['n_nodes'][-1]))

        #domain network
        hx      = self.net2(x[1], training=training)
        yx      = tf.einsum('ijk,ik->ij', weight, hx) + bias

        #boundary network
        hb      = self.net2b(x[1], training=training)
        yb      = tf.einsum('ijk,ik->ij', weight, hb)

        beta, beta_min    = self.surface_function(x[1][:,self.idx_sdf])

        tb          = tf.multiply(beta, yb)
        td          = tf.multiply(beta_min, yx)
        #weighted overall output
        y_pred = tf.add(td, tb)
        return y_pred, hx, yx, hb, yb, weight, bias, beta, beta_min

    def call_net1(self, x, training=False):
        '''Call net1 (parameter network) only.'''
        net1_output     = self.net1(x, training=training)
        weight, bias    = tf.split(net1_output, num_or_size_splits=self.w_dim_ls, axis=1)
        weight          = tf.reshape(weight, (-1, self.opt['net2']['n_output'], self.opt['net2']['n_nodes'][-1]))
        return weight, bias

###TODO: precompute beta surface
    def call_model_net1sep(self, x, weight, bias, training=False):
        '''Call the full model when the outputs of the parameter network are provided, return final result'''
        #domain net
        hx      = self.net2(x, training=training)
        yx      = tf.einsum('ijk,lk->lj', weight, hx) + bias

        #boundary net output
        hb      = self.net2b(x, training=training)
        yb      = tf.einsum('ijk,lk->lj', weight, hb) + bias
        beta, beta_min  = self.surface_function(x[:,self.idx_sdf])

        tb      = tf.multiply(beta, yb)
        td      = tf.multiply(beta_min, yx)
        #weighted overall output
        y_pred = tf.add(td, tb)
        return y_pred

    def call_model_net1sep_beta(self, x, beta, beta_min, weight, bias, training=False):
        #domain net
        hx      = self.net2(x, training=training)
        yx      = tf.einsum('ijk,lk->lj', weight, hx) + bias

        #boundary net output
        hb      = self.net2b(x, training=training)
        yb      = tf.einsum('ijk,lk->lj', weight, hb) + bias

        tb      = tf.multiply(beta, yb)
        td      = tf.multiply(beta_min, yx)
        #weighted overall output
        y_pred = tf.add(td, tb)
        return y_pred

    def add_beta_to_data(self):
        beta_train, beta_min_train = self.surface_function(self.data['x_train'][:,self.idx_sdf])
        beta_val, beta_min_val = self.surface_function(self.data['x_val'][:,self.idx_sdf])

        self.data['beta_train'] = beta_train
        self.data['beta_min_train'] = beta_min_train
        self.data['beta_val'] = beta_val
        self.data['beta_min_val'] = beta_min_val

    def train_model_betasep(self):
        self.create_training_log_val()
        self.add_beta_to_data()

        n_train = self.data['x_train'].shape[0]
        n_val   = self.data['x_val'].shape[0]

        buffer_size   = 5 * self.opt['batch_size']

        #create dataset for validation data, create batches
        val_data  = tf.data.Dataset.from_tensor_slices((self.data['x2_val'],
                                                          self.data['x_val'],
                                                          self.data['beta_val'],
                                                          self.data['beta_min_val'],
                                                          self.data['y_val']))
        val_data  = val_data.batch(batch_size = self.opt['batch_size'])

        #create the training dataset, shuffle, and create batches
        train_data  = tf.data.Dataset.from_tensor_slices((self.data['x2_train'],
                                                          self.data['x_train'],
                                                          self.data['beta_train'],
                                                          self.data['beta_min_train'],
                                                          self.data['y_train']))

        train_data  = train_data.shuffle(buffer_size = buffer_size, reshuffle_each_iteration=True)
        train_data  = train_data.batch(batch_size = self.opt['batch_size'])
        n_epochs    = self.opt['epochs']

        training_loss           = []
        val_loss_store          = []
        val_best_store          = []

        for epoch in range(n_epochs):

            #gradient descent over mini-batches
            batch_loss = []
            for batch, (x2, x, beta, beta_min, y) in enumerate(train_data):
                cur_loss = self.train_on_batch_betasep([x2,x,beta,beta_min],y)
                cur_loss = tf.reduce_sum(cur_loss)
                batch_loss.append(cur_loss)

            mean_loss = np.sum(np.array(batch_loss)) / n_train
            training_loss.append(mean_loss)

            #validation loss - use batching to avoid OOM
            batch_loss_val = []
            for batch, (x2, x, beta, beta_min, y) in enumerate(val_data):
                y_val_pred      = self.call_model([x2, x, beta, beta_min])
                cur_loss_val    = self.loss(y_val_pred, y)
                cur_loss_val    = tf.reduce_sum(cur_loss_val)
                batch_loss_val.append(cur_loss_val)

            val_loss_cur = np.sum(np.array(batch_loss_val)) / n_val
            val_loss_store.append(val_loss_cur)

            #update training log
            self.update_training_log_val(epoch, training_loss[-1],
                                         val_loss_store[-1])

            #print epoch message to console
            self.print_train_val(epoch, n_epochs, training_loss[-1],
                                 val_loss_store[-1])

            #save weights, update best val
            if epoch == 0:
                val_best_cur = val_loss_cur

            if val_loss_store[-1] <= val_best_cur:
                val_best_cur = val_loss_store[-1]
                message = 'Saving best weights'
                print(message)
                self.save_weights(self.fn_weights_best)

            val_best_store.append(val_best_cur)
            if np.mod(epoch, self.opt['n_epochs_save']) == 0:
                message = 'Saving end weights'
                print(message)
                self.save_weights(self.fn_weights_end)

    def train_model(self):
        '''Custom training loop - train model with validation points using tf.keras.losses loss'''

        self.create_training_log_val()
        x2_val      = self.data['x2_val']
        x_val       = self.data['x_val']
        y_val       = self.data['y_val']

        #create the training dataset, shuffle, and create batches
        train_data  = tf.data.Dataset.from_tensor_slices((self.data['x2_train'], self.data['x_train'],
                                                         self.data['y_train']))
        n_data      = self.opt['n_data']
        train_data  = train_data.shuffle(buffer_size = n_data, reshuffle_each_iteration=True)
        train_data  = train_data.batch(batch_size = self.opt['batch_size'])
        n_epochs    = self.opt['epochs']

        training_loss           = []
        val_loss_store          = []
        val_best_store          = []
        for epoch in range(n_epochs):

            #gradient descent over mini-batches
            batch_loss = []
            for batch, (x2, x, y) in enumerate(train_data):
                batch_loss.append(np.mean(self.train_on_batch([x2,x],y)))
            training_loss.append(np.array(batch_loss).mean())

            #validation loss
            y_val_pred      = self.call_model([x2_val, x_val])
            val_loss_cur    = np.mean(self.loss(y_val_pred, y_val))
            val_loss_store.append(val_loss_cur)

            #update training log
            self.update_training_log_val(epoch, training_loss[-1],
                                         val_loss_store[-1])

            #print epoch message to console
            self.print_train_val(epoch, n_epochs, training_loss[-1],
                                 val_loss_store[-1])

            #save weights, update best val
            if epoch == 0:
                val_best_cur = val_loss_cur

            if val_loss_store[-1] <= val_best_cur:
                val_best_cur = val_loss_store[-1]
                message = 'Saving best weights'
                print(message)
                self.save_weights(self.fn_weights_best)

            val_best_store.append(val_best_cur)
            if np.mod(epoch, self.opt['n_epochs_save']) == 0:
                message = 'Saving end weights'
                print(message)
                self.save_weights(self.fn_weights_end)

    @tf.function
    def train_on_batch(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.call_model(x, training=True)
            loss_value = self.loss(y_pred, y)

        grads = tape.gradient(loss_value, tape.watched_variables())
        self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        return loss_value

    @tf.function
    def train_on_batch_betasep(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.call_model_betasep(x, training=True)
            loss_value = self.loss(y_pred, y)

        grads = tape.gradient(loss_value, tape.watched_variables())
        self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        return loss_value

    def create_training_log_val(self):
        '''Create training log, with validation data'''
        f           = open(self.fn_train,'a+')
        f.write(','.join(('epoch','loss','val_loss\n')))
        f.close()

    def update_training_log_val(self, epoch, training_loss, val_loss):
        '''Append results for single epoch to training log'''
        f = open(self.fn_train,'a+')
        f.write(','.join((str(epoch), str(training_loss),
                          str(val_loss)+'\n')))
        f.close()

    def print_train_val(self, epoch, n_epochs, loss, loss_val):
        '''Print message to console during training, with validation data'''
        message = 'Epoch [{:1.0f}/{:1.0f}]: Loss: {:1.4e}, Val_loss: {:1.4e}'\
                    .format(epoch, n_epochs, loss, loss_val)
        print(message)

    def tup_data(self, data):
        '''Turn data organized by case in a dictionary like:
            cases = data.keys()
            data[case].keys() =  'x2','x','y'
        Into a dictionary returning tuples (x2,x,y) 
        '''

        tup_data = {}
        for case, case_data in data.items():
            tup_data[case] = (case_data['x2'], case_data['x'], case_data['y'])
        return tup_data

    def train_model_batch_by_case(self, data=None):
        '''Train the model using batch by case 

        Training data must be organized by dictionaries instead of arrays

        Args:
            data (dict) : if is None, use self.data to train. 
    '''
        self.create_training_log_val()
        if data is None:
            train_data  = self.tup_data(self.data['train_data'])
            val_data    = self.tup_data(self.data['val_data'])
        else:
            train_data  = self.tup_data(data['train_data'])
            val_data    = self.tup_data(data['val_data'])

        n_epochs    = self.opt['epochs']

        training_loss_hist      = []
        val_loss_hist           = []
        val_best_hist           = []

        for epoch in range(n_epochs):
            batch_loss      = []
            #gradient descent over each case
            for idx, (x2,x,y) in train_data.items():
                with tf.GradientTape() as tape:
                    weight, bias    = self.call_net1(x2)
                    y_pred          = self.call_model_net1sep(x, weight, bias, training=False)
                    recon_loss      = self.recon_loss(y_pred, y)
                    loss_value      = tf.reduce_mean(recon_loss)

                grads = tape.gradient(loss_value, tape.watched_variables())
                self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))

                batch_loss.append(loss_value)

            training_loss_hist.append(np.array(batch_loss).mean())

            #validation loss
            batch_loss_val          = []

            for idx, (x2,x,y) in val_data.items():
                weight, bias    = self.call_net1(x2)
                y_pred          = self.call_model_net1sep(x, weight, bias, training=False)
                recon_loss      = self.recon_loss(y_pred, y)
                mean_loss       = tf.reduce_mean(recon_loss)
                loss_value      = mean_loss

                batch_loss_val.append(loss_value)

                
            val_loss_hist.append(np.array(batch_loss_val).mean())

            #update training log
            self.update_training_log_val(epoch, training_loss_hist[-1], val_loss_hist[-1])


            #print epoch message to console
            self.print_train_val(epoch, n_epochs, training_loss_hist[-1],
                                  val_loss_hist[-1])

            #save weights, update best val
            if epoch == 0:
                val_best_cur = val_loss_hist[-1]

            if val_loss_hist[-1] <= val_best_cur:
                val_best_cur = val_loss_hist[-1]
                message = 'Saving best weights'
                print(message)
                self.save_weights(self.fn_weights_best)

            val_best_hist.append(val_best_cur)
            if np.mod(epoch, self.opt['n_epochs_save']) == 0:
                message = 'Saving end weights'
                print(message)
                self.save_weights(self.fn_weights_end)
                self.save_weights_epoch(epoch)

    def add_beta_to_minibatches(self, train_data):
        '''Given a list of tuples (x2, x, y) representing minibatches,
        compute beta and (1-beta) and add to tuple as (x2, x, y, beta, beta_min)'''
        new_data = []
        for iBatch, (x2,x,y) in enumerate(train_data):
            beta, beta_min = self.surface_function(x[:,self.idx_sdf])
            new_data.append((x2, x, y, beta, beta_min))
        return new_data

    def add_beta_to_valdata(self, val_data):
        '''Given a dictionary with key=idx, val=(x2,x,y) representing the 
        validation data, compute beta and (1-beta) and add to tuple as 
        (x2, x, y, beta, beta_min)'''
        new_data = {}
        for idx, (x2,x,y) in val_data.items():
            beta, beta_min  = self.surface_function(x[:,self.idx_sdf])
            new_data[idx] = (x2, x, y, beta, beta_min)
        return new_data

    def train_model_batch_by_case_minibatches(self, data=None):
        '''Train the model using batch by case 

        Training data must be organized by dictionaries instead of arrays
        Minibatches are already created, stored as a list of tuples

        Args:
            data (dict) : if is None, use self.data to train. 
    '''
        self.create_training_log_val()
        if data is None:
            # train_data  = self.tup_data(self.data['train_data'])
            train_data  = self.data['minibatches']
            val_data    = self.tup_data(self.data['val_data'])
        else:
            # train_data  = self.tup_data(data['train_data'])
            train_data  = data['minibatches']
            val_data    = self.tup_data(data['val_data'])

        train_data  = self.add_beta_to_minibatches(train_data)
        val_data    = self.add_beta_to_valdata(val_data)

        n_epochs    = self.opt['epochs']

        training_loss_hist      = []
        val_loss_hist           = []
        val_best_hist           = []

        for epoch in range(n_epochs):

            batch_loss      = []

            #gradient descent over each minibatch, shuffle each time
            random.shuffle(train_data)
            for (x2, x, y, beta, beta_min) in train_data:
                loss_value = self.train_on_batch_mb(x2, x, y, beta, beta_min)
                batch_loss.append(loss_value)

            training_loss_hist.append(np.array(batch_loss).mean())

            #validation loss
            batch_loss_val          = []

            for idx, (x2,x,y,beta,beta_min) in val_data.items():
                weight, bias    = self.call_net1(x2)
                y_pred          = self.call_model_net1sep_beta(x, beta, beta_min, weight, bias, training=False)
                recon_loss      = self.recon_loss(y_pred, y)
                mean_loss       = tf.reduce_mean(recon_loss)
                loss_value      = mean_loss

                batch_loss_val.append(loss_value)
  
            val_loss_hist.append(np.array(batch_loss_val).mean())

            #update training log
            self.update_training_log_val(epoch, training_loss_hist[-1], val_loss_hist[-1])

            #print epoch message to console
            self.print_train_val(epoch, n_epochs, training_loss_hist[-1],
                                  val_loss_hist[-1])

            #save weights, update best val
            if epoch == 0:
                val_best_cur = val_loss_hist[-1]

            if val_loss_hist[-1] <= val_best_cur:
                val_best_cur = val_loss_hist[-1]
                message = 'Saving best weights'
                print(message)
                self.save_weights(self.fn_weights_best)

            val_best_hist.append(val_best_cur)
            if np.mod(epoch, self.opt['n_epochs_save']) == 0:
                message = 'Saving end weights'
                print(message)
                self.save_weights(self.fn_weights_end)
                self.save_weights_epoch(epoch)

    @tf.function
    def train_on_batch_mb(self, x2, x, y, beta, beta_min):
        with tf.GradientTape() as tape:
            weight, bias    = self.call_net1(x2)
            y_pred          = self.call_model_net1sep_beta(x, beta, beta_min, weight, bias, training=True)
            recon_loss      = self.recon_loss(y_pred, y)
            loss_value      = tf.reduce_mean(recon_loss)
            grads = tape.gradient(loss_value, tape.watched_variables())
            self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        return loss_value

        grads = tape.gradient(loss_value, tape.watched_variables())
        self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        return loss_value

    def train_model_databatches(self):
        '''Train model WITHOUT batching by case, but do use
        the option for data-batching

        '''
        pass

    def train_model_batch_by_case_databatches(self):
        '''Train the model using batch by case 

        Training data must be organized by dictionaries instead of arrays
        Minibatches are already created, stored as a list of tuples

        Args:
            data (dict) : if is None, use self.data to train. 
    '''
        self.create_training_log_val()
        epoch_schedule  = self.opt['epoch_schedule']
        n_epochs_tot    = np.sum(np.array(epoch_schedule))
        n_data_batches  = len(epoch_schedule)

        training_loss_hist      = []
        val_loss_hist           = []
        val_best_hist           = []
        epoch_ct                = 0
        for iData, n_epochs in enumerate(epoch_schedule):

            train_data  = self.data[iData]['minibatches']
            val_data    = self.tup_data(self.data[iData]['val_data'])
    
            train_data  = self.add_beta_to_minibatches(train_data)
            val_data    = self.add_beta_to_valdata(val_data)

            for epoch in range(n_epochs):
                #gradient descent over each minibatch, shuffle each time
                batch_loss      = []
                random.shuffle(train_data)
                for (x2, x, y, beta, beta_min) in train_data:
                    loss_value = self.train_on_batch_mb(x2, x, y, beta, beta_min)
                    batch_loss.append(loss_value)
    
                training_loss_hist.append(np.array(batch_loss).mean())
    
                #validation loss
                batch_loss_val          = []
                for idx, (x2,x,y,beta,beta_min) in val_data.items():
                    weight, bias    = self.call_net1(x2)
                    y_pred          = self.call_model_net1sep_beta(x, beta, beta_min, weight, bias, training=False)
                    recon_loss      = self.recon_loss(y_pred, y)
                    mean_loss       = tf.reduce_mean(recon_loss)
                    loss_value      = mean_loss
                    batch_loss_val.append(loss_value)
      
                val_loss_hist.append(np.array(batch_loss_val).mean())
    
                #update training log
                self.update_training_log_val(epoch_ct, training_loss_hist[-1], val_loss_hist[-1])
    
                #print epoch message to console
                self.print_train_val(epoch_ct, n_epochs_tot, training_loss_hist[-1],
                                      val_loss_hist[-1])
    
                #save weights, update best val
                if epoch_ct == 0:
                    val_best_cur = val_loss_hist[-1]
    
                if val_loss_hist[-1] <= val_best_cur:
                    val_best_cur = val_loss_hist[-1]
                    message = 'Saving best weights'
                    print(message)
                    self.save_weights(self.fn_weights_best)
    
                val_best_hist.append(val_best_cur)
                if np.mod(epoch_ct, self.opt['n_epochs_save']) == 0:
                    message = 'Saving end weights'
                    print(message)
                    self.save_weights(self.fn_weights_end)
                    self.save_weights_epoch(epoch_ct)

                epoch_ct += 1

    def save_weights(self, fn_weights):
        self.net1.save_weights(fn_weights['net1'])
        self.net2.save_weights(fn_weights['net2'])
        self.net2b.save_weights(fn_weights['net2b'])

    def save_weights_epoch(self, epoch):
        self.net1.save_weights(self.fn_weights_epoch['net1'].replace('EPOCH','{:1.0f}'.format(epoch)))
        self.net2.save_weights(self.fn_weights_epoch['net2'].replace('EPOCH','{:1.0f}'.format(epoch)))
        self.net2b.save_weights(self.fn_weights_epoch['net2b'].replace('EPOCH','{:1.0f}'.format(epoch)))

    def load_weights(self, fn_weights):
        self.net1.load_weights(fn_weights['net1'])
        self.net2.load_weights(fn_weights['net2'])
        self.net2b.load_weights(fn_weights['net2b'])

    def build_model(self):
        self.layer_names = {}

        # Build net 1
        print('Building net1')
        net1_opt        = self.opt['net1']
        net1_layers     = []

        name            = 'input_net1'
        net1_layers.append(name)
        net1_input      = tf.keras.Input(shape=(net1_opt['n_input'],),
                                         name = name)

        for iDense in range(net1_opt['n_layers']-1):
            print("Hidden Layer %d" % (iDense))
            name        = 'dense_' + str(iDense)
            net1_layers.append(name)
            units       = net1_opt['n_nodes'][iDense]
            activation  = self.activation_net1
            
            if iDense == 0:
                output  = tf.keras.layers.Dense(units = units,
                                                activation = activation,
                                                kernel_regularizer = self.kernel_regularizer,
                                                name = name)(net1_input)
            else:
                output  = tf.keras.layers.Dense(units = units,
                                           activation = activation,
                                           kernel_regularizer = self.kernel_regularizer,
                                           name = name)(output)
            if net1_opt['is_batch_norm']:
                name    = 'batch_norm_' + name
                net1_layers.append(name)
                output  = tf.keras.layers.BatchNormalization(name=name)(output)

        if net1_opt['is_linear_output']:
            activation  = tf.keras.activations.linear
        else:
            activation  = self.activation_net1
        name            = 'net1_output'
        net1_layers.append(name)
        units           = net1_opt['n_nodes'][-1]
        net1_output     = tf.keras.layers.Dense(units = units,
                                       activation = activation,
                                       kernel_regularizer = self.kernel_regularizer,
                                       name = name)(output)

        self.net1 = tf.keras.Model(inputs=[net1_input], outputs=[net1_output])
        self.net1.summary()
        self.w_dim_ls = [self.opt['w_mat_dim'], self.opt['net2']['n_output']]
        self.net1_layers = net1_layers

        # Build net 2
        print('Building net 2')
        net2_opt        = self.opt['net2']
        net2_layers     = []

        print('Input Layer')
        name            = 'net2_input'
        net2_input      = tf.keras.Input(shape=(net2_opt['n_input'],), name=name)

        net2_layers.append(name)
        input1 = tf.keras.Input(shape=(net2_opt['n_input'],), name=name)
        # self.layer_call[name] = input1
        
        for iDense in range(net2_opt['n_layers']):
            print("Hidden Layer %d" % (iDense))
            name = 'dense_' + str(iDense)
            net2_layers.append(name)
            units = net2_opt['n_nodes'][iDense]
            activation = self.activation_net2
            
            if iDense == 0:
                output = tf.keras.layers.Dense(units = units,
                                               activation = activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(net2_input)
                
                
            else:
                output = tf.keras.layers.Dense(units = units,
                                               activation = activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(output)
            if net2_opt['is_batch_norm']:
                name = 'batch_norm_' + name
                net2_layers.append(name)
                output = tf.keras.layers.BatchNormalization(name=name)(output)
        hx = output
        self.net2 = tf.keras.Model(inputs=[net2_input], outputs=[output])
        self.net2.summary()
        self.net2_layers = net2_layers

        # Build net 2b (boundary net)
        #use same inputs layer as net2
        print('Building net 2 boundary')
        net2b_opt        = self.opt['net2b']
        net2b_layers     = []
        if 'is_use_bias' in net2b_opt.keys():
            use_bias = net2b_opt['is_use_bias']
        else:
            use_bias = False

        for iDense in range(net2b_opt['n_layers']):
            print("Hidden Layer %d" % (iDense))
            name = 'dense_' + str(iDense)
            net2b_layers.append(name)
            units = net2b_opt['n_nodes'][iDense]
            activation = self.activation_net2b
            
            if iDense == 0:
                output = tf.keras.layers.Dense(units = units,
                                               use_bias = use_bias,
                                               activation = activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(net2_input)
            else:
                output = tf.keras.layers.Dense(units = units,
                                               use_bias = use_bias,
                                               activation = activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(output)
            if net2b_opt['is_batch_norm']:
                name = 'batch_norm_' + name
                net2b_layers.append(name)
                output = tf.keras.layers.BatchNormalization(name=name)(output)
        self.net2b = tf.keras.Model(inputs=[net2_input], outputs=[output])
        self.net2b.summary()
        self.net2b_layers = net2b_layers

class bnidsbc_v1():
    '''Version 1 of B-NIDS-BC model

    Supports enforcement of dirichlet BCs by construction 

    Set up to work either with tf.keras.losses or using custom training
    loop as required for certain regularizations.

    Args:
        opt (dict)  : setting to define and train model
        data (dict) : optional, used to train the model. Not required if not training

    Attributes:
        MODELS (tf.keras.Model):
            net1    : mu -> w
            net2    : x -> h
            net2s   : x -> h2
    '''

    def __init__(self, opt, data=None, beta_filter=1e-16):
        self.opt    = opt
        self.data   = data

        #set loss - only used directly when self.train_model is used (not with self.train_model_batch_by_case)
        self.loss           = nu.set_loss(opt)
        self.recon_loss     = nu.set_loss(opt)

        #set kernel_regularizer, acitvation function(s), and optimizer
        self.kernel_regularizer     = nu.set_kernel_regularizer(opt)

        self.activation_net1        = nu.set_activation(opt['net1'])
        self.activation_net2        = nu.set_activation(opt['net2'])
        self.activation_net2b       = nu.set_activation(opt['net2'])
        self.optimizer              = nu.set_optimizer(opt)

        #filenames for saving training loss and weights
        self.fn_train = os.path.join(self.opt['save_dir'], 'training.csv')

        #best/end weights
        self.fn_weights_net1_best   = os.path.join(self.opt['save_dir'],
                                                   'weights.net1.best.h5')
        self.fn_weights_net1_end    = os.path.join(self.opt['save_dir'],
                                                   'weights.net1.end.h5')
        self.fn_weights_net2_best   = os.path.join(self.opt['save_dir'],
                                                   'weights.net2.best.h5')
        self.fn_weights_net2_end    = os.path.join(self.opt['save_dir'],
                                                   'weights.net2.end.h5')
        self.fn_weights_net2b_best   = os.path.join(self.opt['save_dir'],
                                                   'weights.net2b.best.h5')
        self.fn_weights_net2b_end    = os.path.join(self.opt['save_dir'],
                                                   'weights.net2b.end.h5')


        self.fn_weights_best        = {'net1':self.fn_weights_net1_best,
                                       'net2':self.fn_weights_net2_best,
                                       'net2b':self.fn_weights_net2b_best}
        self.fn_weights_end         = {'net1':self.fn_weights_net1_end,
                                       'net2':self.fn_weights_net2_end,
                                       'net2b':self.fn_weights_net2b_end}

        #every X epochs weights
        if self.opt['is_save_all_weights']:
            weight_dir     = os.path.join(self.opt['save_dir'], 'weights')

            if not os.path.exists(weight_dir):
                os.mkdir(weight_dir)
            self.opt['weight_dir'] = weight_dir
            self.fn_weights_net1_epoch = os.path.join(self.opt['weight_dir'],
                                                      'weights.net1.EPOCH.h5')
            self.fn_weights_net2_epoch = os.path.join(self.opt['weight_dir'],
                                                      'weights.net2.EPOCH.h5')
            self.fn_weights_net2b_epoch = os.path.join(self.opt['weight_dir'],
                                                      'weights.net2b.EPOCH.h5')
            self.fn_weights_net3_epoch = os.path.join(self.opt['weight_dir'],
                                                      'weights.net3.EPOCH.h5')
            self.fn_weights_epoch   = {'net1':self.fn_weights_net1_epoch,
                                       'net2':self.fn_weights_net2_epoch,
                                       'net2b':self.fn_weights_net2b_epoch,
                                       'net3':self.fn_weights_net3_epoch}

        #set surface network parameters
        self.sigma      = opt['net2b']['sigma']
        try:
            self.idx_sdf    = opt['net2']['inputs'].index('sdf')
        except:
            self.idx_sdf    = opt['net2']['inputs'].index('sdf_overall')

        self.net2b_BC_dim           = np.expand_dims(np.array(opt['net2']['outputs_bc']), axis=0)
        self.net2_BC = None
        # self.net2b_BC_dim = tf.convert_to_tensor(netbc, dtype=tf.float64)
        self.beta_filter = beta_filter
        self.build_model()

    def surface_function(self, x):
        beta = np.exp(-0.5 * (x**2) / (self.sigma**2) )
        beta[beta < self.beta_filter] = 0
        beta = np.expand_dims(beta, axis=1)
        beta_min    = 1 - beta
        beta        = tf.convert_to_tensor(beta)
        beta_min    = tf.convert_to_tensor(beta_min)
        return beta, beta_min

    def surface_function_numpy(self, x):
        beta = np.exp(-0.5 * (x**2) / (self.sigma**2) )
        beta[beta < self.beta_filter] = 0
        return beta

    def call_model(self, x, training=False):
        '''Call the full model, return only the final result'''

        #parameter network
        net1_output     = self.net1(x[0], training=training)
        weight, bias    = tf.split(net1_output, num_or_size_splits=self.w_dim_ls, axis=1)
        weight          = tf.reshape(weight, (-1, self.opt['net2']['n_output'], self.opt['net2']['n_nodes'][-1]))

        #domain network
        hx      = self.net2(x[1], training=training)
        yx      = tf.einsum('ijk,ik->ij', weight, hx) + bias

        #boundary network
        hb      = self.net2b(x[2][:,self.idx_sdf], training=training)
        yb      = tf.einsum('ijk,ik->ij', weight, hb) + self.net2b_BC

        beta, beta_min    = self.surface_function(x[1][:,self.idx_sdf])
        tb          = tf.multiply(beta, yb)
        td          = tf.multiply(beta_min, yx)
        #weighted overall output
        y_pred = tf.add(td, tb)
        return y_pred

    def call_model_verbose(self, x, training=False):
        '''Call the full model, return final and intermediate results'''

        #parameter network
        net1_output     = self.net1(x[0], training=training)
        weight, bias    = tf.split(net1_output, num_or_size_splits=self.w_dim_ls, axis=1)
        weight          = tf.reshape(weight, (-1, self.opt['net2']['n_output'], self.opt['net2']['n_nodes'][-1]))

        #domain network
        hx      = self.net2(x[1], training=training)
        yx      = tf.einsum('ijk,ik->ij', weight, hx) + bias

        #boundary network
        hb      = self.net2b(x[1][:,[self.idx_sdf]], training=training)
        yb      = tf.einsum('ijk,ik->ij', weight, hb) + self.net2b_BC

        beta, beta_min    = self.surface_function(x[1][:,self.idx_sdf])
        tb          = tf.multiply(beta, yb)
        td          = tf.multiply(beta_min, yx)
        #weighted overall output
        y_pred = tf.add(td, tb)
        return y_pred, hx, yx, hb, yb, weight, bias, beta, beta_min

    def call_net1(self, x, training=False):
        '''Call net1 (parameter network) only.'''
        net1_output     = self.net1(x, training=training)
        weight, bias    = tf.split(net1_output, num_or_size_splits=self.w_dim_ls, axis=1)
        weight          = tf.reshape(weight, (-1, self.opt['net2']['n_output'], self.opt['net2']['n_nodes'][-1]))
        return weight, bias

###TODO: precompute beta surface
    def call_model_net1sep(self, x, sdf, weight, bias, training=False):
        '''Call the full model when the outputs of the parameter network are provided, return final result'''
        #domain net
        hx      = self.net2(x, training=training)
        yx      = tf.einsum('ijk,lk->lj', weight, hx) + bias

        #boundary net output
        hb      = self.net2b(sdf, training=training)
        yb      = tf.einsum('ijk,lk->lj', weight, hb) + self.net2b_BC
        # yb      = tf.nn.bias_add(ybint, self.net2b_BC)

        beta, beta_min    = self.surface_function(x[1][:,self.idx_sdf])
        tb          = tf.multiply(beta, yb)
        td          = tf.multiply(beta_min, yx)
        #weighted overall output
        y_pred = tf.add(td, tb)
        return y_pred

    def call_model_net1sep_beta(self, x, sdf, beta, beta_min, weight, bias, training=False):
        #domain net
        hx      = self.net2(x, training=training)
        yx      = tf.einsum('ijk,lk->lj', weight, hx) + bias

        #boundary net output
        hb      = self.net2b(sdf, training=training)
        yb      = tf.einsum('ijk,lk->lj', weight, hb) + self.net2b_BC

        tb      = tf.multiply(beta, yb)
        td      = tf.multiply(beta_min, yx)
        #weighted overall output
        y_pred = tf.add(td, tb)
        return y_pred

    def create_training_log_val(self):
        '''Create training log, with validation data'''
        f           = open(self.fn_train,'a+')
        f.write(','.join(('epoch','loss','val_loss\n')))
        f.close()

    def update_training_log_val(self, epoch, training_loss, val_loss):
        '''Append results for single epoch to training log'''
        f = open(self.fn_train,'a+')
        f.write(','.join((str(epoch), str(training_loss),
                          str(val_loss)+'\n')))
        f.close()

    def print_train_val(self, epoch, n_epochs, loss, loss_val):
        '''Print message to console during training, with validation data'''
        message = 'Epoch [{:1.0f}/{:1.0f}]: Loss: {:1.4e}, Val_loss: {:1.4e}'\
                    .format(epoch, n_epochs, loss, loss_val)
        print(message)

    def tup_data(self, data):
        '''Turn data organized by case in a dictionary like:
            cases = data.keys()
            data[case].keys() =  'x2','x','y'
        Into a dictionary returning tuples (x2,x,y) 
        '''

        tup_data = {}
        for case, case_data in data.items():
            tup_data[case] = (case_data['x2'], case_data['x'], case_data['y'])
        return tup_data

    def add_beta_to_minibatches(self, train_data):
        '''Given a list of tuples (x2, x, y) representing minibatches,
        compute beta and (1-beta) and add to tuple as (x2, x, y, beta, beta_min)'''
        new_data = []
        for iBatch, (x2,x,y) in enumerate(train_data):
            beta, beta_min = self.surface_function(x[:,self.idx_sdf])
            new_data.append((x2, x, y, beta, beta_min))
        return new_data

    def add_sdf_to_minibatches(self, train_data):
        '''Given a list of tuples (x2, x, y, beta, beta_min), separate
        the sdf from x and save tuple as (x2, x, sdf, y, beta, beta_min)'''
        new_data = []
        for iBatch, (x2,x,y,beta,beta_min) in enumerate(train_data):
            sdf     = x[:,[self.idx_sdf]]
            new_data.append((x2, x, sdf, y, beta, beta_min))
        return new_data

    def add_beta_to_valdata(self, val_data):
        '''Given a dictionary with key=idx, val=(x2,x,y) representing the 
        validation data, compute beta and (1-beta) and add to tuple as 
        (x2, x, y, beta, beta_min)'''
        new_data = {}
        for idx, (x2,x,y) in val_data.items():
            beta, beta_min  = self.surface_function(x[:,self.idx_sdf])
            new_data[idx] = (x2, x, y, beta, beta_min)
        return new_data

    def add_sdf_to_valdata(self, val_data):
        '''Given a dictionary with key=idx, val=(x2,x,y,beta,beta_min) representing the 
        validation data, extract sdf from x and save as
        (x2, x, sdf, y, beta, beta_min)'''
        new_data = {}
        for idx, (x2,x,y,beta,beta_min) in val_data.items():
            sdf     = x[:,[self.idx_sdf]]
            new_data[idx] = (x2, x, sdf, y, beta, beta_min)
        return new_data

    def set_BC_norm(self, data_stats):
        bc_norm = np.zeros(self.opt['net2']['n_output'])
        for iSignal, signal in enumerate(self.opt['net2']['outputs']):
            bc_norm[iSignal] = util.norm_single_by_stats_flat(x = self.net2b_BC_dim[:,[iSignal]],
                                          x_stats = data_stats['y_stats'][iSignal],
                                          method = self.opt['net2']['norm_y_by'])
        bc_norm = np.expand_dims(bc_norm, axis=0)
        self.net2b_BC = tf.convert_to_tensor(bc_norm, dtype=tf.float64)


    def train_model_batch_by_case_databatches(self):
        '''Train the model using batch by case 

        Training data must be organized by dictionaries instead of arrays
        Minibatches are already created, stored as a list of tuples

        Args:
            data (dict) : if is None, use self.data to train. 
    '''
        self.create_training_log_val()
        epoch_schedule  = self.opt['epoch_schedule']
        n_epochs_tot    = np.sum(np.array(epoch_schedule))
        n_data_batches  = len(epoch_schedule)

        training_loss_hist      = []
        val_loss_hist           = []
        val_best_hist           = []
        epoch_ct                = 0
        for iData, n_epochs in enumerate(epoch_schedule):

            train_data  = self.data[iData]['minibatches']
            val_data    = self.tup_data(self.data[iData]['val_data'])
    
            train_data  = self.add_beta_to_minibatches(train_data)
            val_data    = self.add_beta_to_valdata(val_data)

            train_data  = self.add_sdf_to_minibatches(train_data)
            val_data    = self.add_sdf_to_valdata(val_data)

            #set the BC - depends on the data stats
            try:
                self.set_BC_norm(self.opt['data_stats'][iData])
            except:
                self.set_BC_norm(self.opt['data_stats'])

            for epoch in range(n_epochs):
                #gradient descent over each minibatch, shuffle each time
                batch_loss      = []
                random.shuffle(train_data)
                for (x2, x, sdf, y, beta, beta_min) in train_data:
                    loss_value = self.train_on_batch_mb(x2, x, sdf, y, beta, beta_min)
                    batch_loss.append(loss_value)
    
                training_loss_hist.append(np.array(batch_loss).mean())
    
                #validation loss
                batch_loss_val          = []
                for idx, (x2, x, sdf, y, beta, beta_min) in val_data.items():
                    weight, bias    = self.call_net1(x2)
                    y_pred          = self.call_model_net1sep_beta(x, sdf, beta, beta_min, weight, bias, training=False)
                    recon_loss      = self.recon_loss(y_pred, y)
                    mean_loss       = tf.reduce_mean(recon_loss)
                    loss_value      = mean_loss
                    batch_loss_val.append(loss_value)
      
                val_loss_hist.append(np.array(batch_loss_val).mean())
    
                #update training log
                self.update_training_log_val(epoch_ct, training_loss_hist[-1], val_loss_hist[-1])
    
                #print epoch message to console
                self.print_train_val(epoch_ct, n_epochs_tot, training_loss_hist[-1],
                                      val_loss_hist[-1])
    
                #save weights, update best val
                if epoch_ct == 0:
                    val_best_cur = val_loss_hist[-1]
    
                if val_loss_hist[-1] <= val_best_cur:
                    val_best_cur = val_loss_hist[-1]
                    message = 'Saving best weights'
                    print(message)
                    self.save_weights(self.fn_weights_best)
    
                val_best_hist.append(val_best_cur)
                if np.mod(epoch_ct, self.opt['n_epochs_save']) == 0:
                    message = 'Saving end weights'
                    print(message)
                    self.save_weights(self.fn_weights_end)
                    self.save_weights_epoch(epoch_ct)

                epoch_ct += 1

    # @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float64)))
    @tf.function
    def train_on_batch_mb(self, x2, x, sdf, y, beta, beta_min, experimental_relax_shapes=True):
        with tf.GradientTape() as tape:
            weight, bias    = self.call_net1(x2)
            y_pred          = self.call_model_net1sep_beta(x, sdf, beta, beta_min, weight, bias, training=True)
            recon_loss      = self.recon_loss(y_pred, y)
            loss_value      = tf.reduce_mean(recon_loss)
            grads = tape.gradient(loss_value, tape.watched_variables())
            self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        return loss_value

        grads = tape.gradient(loss_value, tape.watched_variables())
        self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        return loss_value

    def create_training_log_hreg(self):
        '''Create training log, with validation data'''

        f = open(self.fn_train,'a+')
        header = ('epoch', 'loss', 'val_loss', 'recon_loss', 'val_recon_loss',
                  'h_ortho_loss', 'val_h_ortho_loss\n')
        f.write(','.join(header))
        f.close()

    def update_training_log_hreg(self, epoch, training_loss, val_loss, recon_loss,
                                recon_loss_val, h_ortho_loss, h_ortho_loss_val):
        '''Append results for single epoch to training log'''
        f = open(self.fn_train,'a+')
        f.write(','.join((str(epoch), str(training_loss),
                          str(val_loss), str(recon_loss),
                          str(recon_loss_val), str(h_ortho_loss),
                          str(h_ortho_loss_val)+'\n')))
        f.close()

    def save_weights(self, fn_weights):
        self.net1.save_weights(fn_weights['net1'])
        self.net2.save_weights(fn_weights['net2'])
        self.net2b.save_weights(fn_weights['net2b'])

    def save_weights_epoch(self, epoch):
        self.net1.save_weights(self.fn_weights_epoch['net1'].replace('EPOCH','{:1.0f}'.format(epoch)))
        self.net2.save_weights(self.fn_weights_epoch['net2'].replace('EPOCH','{:1.0f}'.format(epoch)))
        self.net2b.save_weights(self.fn_weights_epoch['net2b'].replace('EPOCH','{:1.0f}'.format(epoch)))

    def load_weights(self, fn_weights):
        self.net1.load_weights(fn_weights['net1'])
        self.net2.load_weights(fn_weights['net2'])
        self.net2b.load_weights(fn_weights['net2b'])

    def build_model(self):
        self.layer_names = {}

# Build net 1
        print('Building net1')
        net1_opt        = self.opt['net1']
        net1_layers     = []

        name            = 'input_net1'
        net1_layers.append(name)
        net1_input      = tf.keras.Input(shape=(net1_opt['n_input'],),
                                         name = name)

        for iDense in range(net1_opt['n_layers']-1):
            print("Hidden Layer %d" % (iDense))
            name        = 'dense_' + str(iDense)
            net1_layers.append(name)
            units       = net1_opt['n_nodes'][iDense]
            activation  = self.activation_net1
            
            if iDense == 0:
                output  = tf.keras.layers.Dense(units = units,
                                                activation = activation,
                                                kernel_regularizer = self.kernel_regularizer,
                                                name = name)(net1_input)
            else:
                output  = tf.keras.layers.Dense(units = units,
                                           activation = activation,
                                           kernel_regularizer = self.kernel_regularizer,
                                           name = name)(output)
            if net1_opt['is_batch_norm']:
                name    = 'batch_norm_' + name
                net1_layers.append(name)
                output  = tf.keras.layers.BatchNormalization(name=name)(output)

        if net1_opt['is_linear_output']:
            activation  = tf.keras.activations.linear
        else:
            activation  = self.activation_net1
        name            = 'net1_output'
        net1_layers.append(name)
        units           = net1_opt['n_nodes'][-1]
        net1_output     = tf.keras.layers.Dense(units = units,
                                       activation = activation,
                                       kernel_regularizer = self.kernel_regularizer,
                                       name = name)(output)

        self.net1 = tf.keras.Model(inputs=[net1_input], outputs=[net1_output])
        self.net1.summary()
        self.w_dim_ls = [self.opt['w_mat_dim'], self.opt['net2']['n_output']]

        # Build net 2
        print('Building net 2')
        net2_opt        = self.opt['net2']
        net2_layers     = []

        print('Input Layer')
        name            = 'net2_input'
        net2_input      = tf.keras.Input(shape=(net2_opt['n_input'],), name=name)

        net2_layers.append(name)
        input1 = tf.keras.Input(shape=(net2_opt['n_input'],), name=name)
        # self.layer_call[name] = input1
        
        for iDense in range(net2_opt['n_layers']):
            print("Hidden Layer %d" % (iDense))
            name = 'dense_' + str(iDense)
            net2_layers.append(name)
            units = net2_opt['n_nodes'][iDense]
            activation = self.activation_net2
            
            if iDense == 0:
                output = tf.keras.layers.Dense(units = units,
                                               activation = activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(net2_input)
            else:
                output = tf.keras.layers.Dense(units = units,
                                               activation = activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(output)
            if net2_opt['is_batch_norm']:
                name = 'batch_norm_' + name
                net2_layers.append(name)
                output = tf.keras.layers.BatchNormalization(name=name)(output)
        hx = output
        self.net2 = tf.keras.Model(inputs=[net2_input], outputs=[output])
        self.net2.summary()

# Build net 2b (boundary net)
#use separate input layer
        print('Building net 2 boundary')
        net2b_opt        = self.opt['net2b']
        net2b_layers     = []

        print('Input Layer')
        name            = 'net2b_input'
        net2b_input      = tf.keras.Input(shape=(1,), name=name)

        for iDense in range(net2b_opt['n_layers']):
            print("Hidden Layer %d" % (iDense))
            name = 'dense_' + str(iDense)
            net2b_layers.append(name)
            units = net2b_opt['n_nodes'][iDense]
            activation = self.activation_net2b
            
            if iDense == 0:
                output = tf.keras.layers.Dense(units = units,
                                               use_bias = False,
                                               activation = activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(net2b_input)
            else:
                output = tf.keras.layers.Dense(units = units,
                                               use_bias = False,
                                               activation = activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(output)
            if net2b_opt['is_batch_norm']:
                name = 'batch_norm_' + name
                net2b_layers.append(name)
                output = tf.keras.layers.BatchNormalization(name=name)(output)

        self.net2b = tf.keras.Model(inputs=[net2b_input], outputs=[output])
        self.net2b.summary()