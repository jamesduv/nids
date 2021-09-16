#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jd
contact: jamesduv@umich.edu
affiliation: University of Michigan, Department of Aerospace Eng., CASLAB
"""

import numpy as np

import nids_networks as nids

def train_opt(network_func      = 'nice_v2',
              name              = 'af_test',
              data_opt          = {
                                'dataset'   : 'gm2',
                                'n_cases'   :   10,
                                'idx_cases' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                                'sdf_lim'   : 2,
                                'speed'     : 90},
              is_siren          = True,
              is_mixed_batches  = False,
              batch_size        = 10000,
              is_batch_by_case = True,
              is_truncate_batch_to_min = True,
              is_data_batch     = False,
              n_data_batches    = 3,
              training_fraction = 0.8,
              epochs            = 100000,
              loss              = 'MSE',
              optimizer         = 'Adam',
              n_epochs_save     = 25,
              learning_rate     = 1e-04,
              kernel_regularizer = 'l2',
              is_load_weights   = False,
              fn_weights_train  = None,   #weights to use IF continuing
              is_save_all_weights = True,
              is_ortho_reg_h    = False,
              is_ortho_reg_g    = False,
              lambda_reg_h      = 1e-03,
              lambda_reg_g      = 1e-03,
              lambda_l1         = 1e-05,
              lambda_l2         = 0,
              save_dir_base     = '../NICE_Predictions/',
              epochs_batch       = 4000,
              epochs_final      = 50000,
              is_debugging      = True):
    '''Generate a dictionary of the training options

    Args:
        most are self explanatory, a few exceptions

        is_batch_by_case (bool) : if True, batch_size is ignored and training data
            must be loaded by dictionary instead of by array
        is_save_all_weights (bool) : if True, save the weights every n_epochs_save
            to folder model_save_dir/weights/netNUMBER.weights.EPOCH.h5
    
    Returns:
        opt (dict) : all general training options
    '''

    optimizer_kwargs = {'decay':learning_rate / epochs}

    epoch_schedule = []
    for ii in np.arange(n_data_batches):
        epoch_schedule.append(epochs_batch)
    epoch_schedule.append(epochs_final)

    opt = {'network_func'       : network_func,
              'name'            : name,
              'data_opt'        : data_opt,
              'is_siren'        : is_siren,
              'is_mixed_batches' : is_mixed_batches,
              'is_batch_by_case' : is_batch_by_case,
              'is_truncate_batch_to_min' : is_truncate_batch_to_min,
              'training_fraction': training_fraction,
              'epochs'          : epochs,
              'batch_size'      : batch_size,
              'loss'            : loss,
              'optimizer'       : optimizer,
              'n_epochs_save'   : n_epochs_save,
              'learning_rate'   : learning_rate,
              'kernel_regularizer':kernel_regularizer,
              'is_load_weights' : is_load_weights,
              'fn_weights_train': fn_weights_train,
              'is_save_all_weights':is_save_all_weights,
              'is_ortho_reg_h'  : is_ortho_reg_h,
              'is_ortho_reg_g'  : is_ortho_reg_g,
              'lambda_reg_h'    : lambda_reg_h,
              'lambda_reg_g'    : lambda_reg_g,
              'lambda_l1'       : lambda_l1,
              'lambda_l2'       : lambda_l2,
              'save_dir_base'   : save_dir_base,
              'is_debugging'    : is_debugging,
              'is_data_batch'   : is_data_batch,
              'n_data_batches'  : n_data_batches,
              'epochs_batch'    : epochs_batch,
              'epochs_final'    : epochs_final,
              'epoch_schedule'  : epoch_schedule,
              'optimizer_kwargs': optimizer_kwargs
              }
    return opt

def net1_opt(n_layers           = 4,
             is_expand_linear   = False,
             inputs             = ['xcen', 'ycen', 'radius', 'rotation'],
             outputs            = ['weights'],
             activation         = 'swish',
             is_batch_norm      = False,
             is_linear_output   = True,
             norm_x_by          = 'range',
             norm_y_by          = 'none',
             hidden_dimension   = 10,  #only used if is_expand_linear is false
             is_siren_init      = False
             ):

    '''Define net1 options for non-autodecoder net 1. use with:
        nids.nids_v2
        nice.bnids_v1
        nice.bnidsbc_v1
    '''

    opt = {
        'n_layers'          : n_layers,
        'is_expand_linear'  : is_expand_linear,
        'inputs'            : inputs,
        'outputs'           : outputs,
        'activation'        : activation,
        'is_batch_norm'     : is_batch_norm,
        'is_linear_output'  : is_linear_output,
        'norm_x_by'         : norm_x_by,
        'norm_y_by'         : norm_y_by,
        'hidden_dimension'  : hidden_dimension,
        'is_siren_init'     : is_siren_init
        }
    return opt

def net2b_opt(n_layers          = 5,
             n_nodes            = [25, 25, 25, 25, 50],
             inputs             = ['xc', 'yc', 'sdf_overall'],
             activation         = 'sin_act2',
             is_batch_norm      = False,
             is_linear_output   = True,
             norm_x_by          = 'range',
             norm_y_by          = 'range',
             sigma              = 0.1,
             is_norm_sigma      = True,
             is_use_bias        = True,     #only applies to nice_boundary_v1 networks, bias is not used in boundary_v2
             is_siren_init      = True
             ):
    '''Generate a dictionary of all options for PINN boundary network

    Args:
        self explanatory
    
    Returns:
        opt (dict) : all PINN options'''
    n_input = len(inputs)

    opt = {
        'n_layers'  : n_layers,
        'n_nodes'   : n_nodes,
        'n_input'   : n_input,
        'inputs'    : inputs,
        'activation'    : activation,
        'is_batch_norm' : is_batch_norm,
        'is_linear_output' : is_linear_output,
        'norm_x_by'     : norm_x_by,
        'norm_y_by'     : norm_y_by,
        'sigma'         : sigma,
        'is_norm_sigma' : is_norm_sigma,
        'is_use_bias'   : is_use_bias,
        'is_siren_init' : is_siren_init
        }
    return opt

def net2_opt(n_layers   = 5,
             n_nodes    = [25, 25, 25, 25, 50],
             inputs     = ['xc', 'yc', 'sdf'],
             outputs    = ['p', 'u', 'v'],
             outputs_bc = [100],       #only used in nice_boundary_v2
             activation = 'sin_act2',
             is_batch_norm      = False,
             is_linear_output   = True,
             norm_x_by          = 'range',
             norm_y_by          = 'range',
             is_siren_init      = True
             ):
    '''Generate a dictionary of all options for PINN architecture

    Args:
        self explanatory
    
    Returns:
        opt (dict) : all PINN options'''
    n_input     = len(inputs)
    n_output    = len(outputs)
    #compute the number of weights, nw
    layer_dim   = [n_input]
    for i in range(n_layers):
        layer_dim.append(n_nodes[i])
    layer_dim.append(n_output)

    n_weights = 0
    for i in range(n_layers+1):
        n_weights = n_weights + ((layer_dim[i]+1) * layer_dim[i+1])

    opt = {'n_layers'   : n_layers,
           'n_nodes'    : n_nodes,
           'n_input'    : n_input,
           'n_output'   : n_output,
           'inputs'     : inputs,
           'outputs'    : outputs,
           'outputs_bc' : outputs_bc,
           'n_weights'  : n_weights,
           'activation' : activation,
           'is_batch_norm'      : is_batch_norm,
           'is_linear_output'   : is_linear_output,
           'norm_x_by'  : norm_x_by,
           'norm_y_by'  : norm_y_by,
           'is_siren_init' : is_siren_init}
    return opt



def build_net1_dense_schedule(opt):
    '''Build the schedule for constructing the dense layers'''

    n_param     = opt['n_param']
    n_layers    = opt['net1']['n_layers']
    output_dim  = opt['net2']['n_output']    #number of signals in output of net3
    h_dim       = opt['net2']['n_nodes'][-1]      #dimension of hidden output from net2

    w_dim       = h_dim * output_dim + output_dim     #dimension of w, given a different bias for each output dimension

    n0      = n_param
    n_final = w_dim

    #linearly expanding network
    if opt['net1']['is_expand_linear']:
        step = int(np.floor((n_final - n0) / (n_layers)))
        n_nodes_dense = []
        for i in np.arange(0, n_layers):
            if i==0:
                n_nodes_dense.append(n0+step)
            else:
                n_nodes_dense.append(n_nodes_dense[i-1] + step)
    else:
        n_nodes_dense = []
        for i in np.arange(n_layers):
            n_nodes_dense.append(opt['net1']['hidden_dimension'])

    #ensure final dimension is correct - required if not linearly expanding network
    n_nodes_dense[-1]       = n_final
    opt['net1']['n_nodes']  = n_nodes_dense
    opt['w_dim']            = w_dim
    opt['h_dim']            = h_dim
    opt['w_mat_dim']        = w_dim - output_dim
    return opt

def get_2dshapes_data_opt(size      = 1000,     #int    : number of random instances per shape
                          dr_surf   = 5e-03,    #float  : delta r, surface (mesh spacing)
                          dr_wall   = 5e-02,    #float  : delta r, wall (mesh spacing)
                          Lsrc      = 500,      #float  : source term left, strength
                          Rsrc      = -500,      #float  : source term right, strength
                          bc_wall   = 100,      #float  : wall boundary condition
                          bc_body   = 100,      #float  : inner body boundary condition
                          is_radius_limits = False,     #bool   : if True, ensure smallest and largest radius shapes in training set
                          n_shapes  = 8,        #int    : number of different shapes (1-8)
                          n_cases_shape = 10, #int    : number of cases per shape
                          idx_cases = None):
    n_cases     = n_shapes * n_cases_shape
    data_opt = {
                'size'     : size,
                'dr_surf'  : dr_surf,
                'dr_wall'  : dr_wall,
                'Lsrc'     : Lsrc,
                'Rsrc'     : Rsrc,
                'bc_wall'  : bc_wall,
                'bc_body'  : bc_body,
                'is_radius_limits': is_radius_limits,
                'n_shapes' : n_shapes,
                'n_cases_shape' : n_cases_shape,
                'n_cases'   : n_cases,
                'idx_cases': idx_cases }
    return data_opt

def set_network_func(opt):
    '''Set the class to create the network

    Args:
        opt (dict) : dict containing the options to construct the network'''

    allowed_funcs = {
        'nids_v2':nids.nids_v2,
        'bnids_v1':nids.bnids_v1,
        'bnids':nids.bnidsbc_v1}
    if opt['network_func'] not in allowed_funcs.keys():
        raise Exception('Unsupported network type specified: {}'.format(opt['network_func']))
    return allowed_funcs[opt['network_func']]

def bound_model_test(opt):
    allowed_funcs = {
        'nice_v1':False,
        'nice_v2':False,
        'nice_boundary_v1':True,
        'nice_boundary_v2':True,
        'nice_ad_v1':True
                     }
    if opt['network_func'] not in allowed_funcs.keys():
        raise Exception('Unsupported network type specified: {}'.format(opt['network_func']))
    return allowed_funcs[opt['network_func']]