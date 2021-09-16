#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jd
contact: jamesduv@umich.edu
affiliation: University of Michigan, Department of Aerospace Eng., CASLAB
"""

import os, sys
import numpy as np

import poisson_train, nids_helper

def run_nids_poisson(study, name,
                         network_func   = 'nids_v1',
                         n_shapes       = 8,
                         n_cases_shape  = 1000,
                         is_radius_limits = False,
                         net1_layers    = 5,
                         net1_nodes     = 50,
                         net2_layers    = 5,
                         net2_nodes     = 50,
                         net1_inputs    = ['xcen', 'ycen', 'radius', 'rotation'],
                         net2_inputs    = ['xc', 'yc', 'sdf_overall'],
                         net2_outputs   = ['q'],
                         outputs_bc     = [0],
                         hx_dim         = 10,
                         net1_norm_x_by = 'range',
                         net2_norm_x_by = 'range',
                         net2_norm_y_by = 'range',
                         epochs         = 100000,
                         learning_rate  = 1e-05,
                         is_batch_by_case = False,
                         is_mixed_batches = True,
                         batch_size     = 500000,
                         is_use_bias    = True,
                         is_data_batch  = False,
                         n_data_batches = 4,
                         epochs_batch   = 4000,
                         epochs_final   = 50000,
                         is_set_seed    = True,
                         seed_val       = 0,
                         sigma          = 5e-02,
                         is_norm_sigma  = True,
                         is_siren       = True,
                         net1_act       = 'swish',
                         net2_act       = 'swish',
                         net2b_act      = 'swish',
                         net1_is_siren_init = False,
                         net2_is_siren_init = False,
                         net2b_is_siren_init = False):

    #build directory structure
    save_dir_base = os.path.join(sys.path[0], '..', 'NICE_Predictions')
    if not os.path.exists(save_dir_base):
        os.mkdir(save_dir_base)
    save_dir_base = os.path.join(save_dir_base, study)
    if not os.path.exists(save_dir_base):
        os.mkdir(save_dir_base)

    is_boundary = nids_helper.bound_model_test({'network_func':network_func})

    data_opt = nids_helper.get_2dshapes_data_opt(size      = n_cases_shape,
                          dr_surf   = 5e-03,    #float  : delta r, surface (mesh spacing)
                          dr_wall   = 5e-02,    #float  : delta r, wall (mesh spacing)
                          Lsrc      = 500,      #float  : source term left, strength
                          Rsrc      = -500,      #float  : source term right, strength
                          bc_wall   = 100,      #float  : wall boundary condition
                          bc_body   = 100,      #float  : inner body boundary condition
                          is_radius_limits = is_radius_limits,     #bool   : if True, ensure smallest and largest radius shapes in training set
                          n_shapes  = n_shapes,        #int    : number of different shapes (1-8)
                          n_cases_shape = n_cases_shape, #int    : number of cases per shape
                          idx_cases = None)

    train_opt = nids_helper.train_opt(network_func = network_func,
              name              = name,
              data_opt          = data_opt,
              is_batch_by_case  = is_batch_by_case,
              is_mixed_batches  = is_mixed_batches,
              is_siren          = is_siren,
              training_fraction = 0.8,
              epochs            = epochs,
              batch_size        = batch_size,
              loss              = 'MSE',
              optimizer         = 'Adam',
              n_epochs_save     = 25,
              learning_rate     = learning_rate,
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
              save_dir_base     = save_dir_base,
              is_debugging      = False,
              is_data_batch     = is_data_batch,
              n_data_batches    = n_data_batches,
              epochs_batch      = epochs_batch,
              epochs_final      = epochs_final
              )

    net1_opt = nids_helper.net1_opt(n_layers = net1_layers,
             is_expand_linear   = False,
             inputs             = net1_inputs,
             outputs            = ['weights'],
             activation         = net1_act,
             is_batch_norm      = False,
             is_linear_output   = True,
             norm_x_by          = net1_norm_x_by,
             norm_y_by          = 'none',
             hidden_dimension   = net1_nodes,  #only used if is_expand_linear is false
             is_siren_init      = net1_is_siren_init
             )

    net2_nodes_ls = []
    for ii in np.arange(net2_layers):
        net2_nodes_ls.append(net2_nodes)

    net2_nodes_ls[-1] = hx_dim

    net2_opt = nids_helper.net2_opt(n_layers   = net2_layers,
             n_nodes    = net2_nodes_ls,
             inputs     = net2_inputs,
             outputs    = net2_outputs,
             outputs_bc = outputs_bc,
             activation = net2_act,
             is_batch_norm      = False,
             is_linear_output   = True,
             norm_x_by          = net2_norm_x_by,
             norm_y_by          = net2_norm_y_by,
             is_siren_init      = net2_is_siren_init
             )

    net2b_opt = nids_helper.net2b_opt(n_layers          = net2_layers,
             n_nodes            = net2_nodes_ls,
             inputs             = net2_inputs,
             activation         = net2b_act,
             is_batch_norm      = False,
             is_linear_output   = True,
             norm_x_by          = net2_norm_x_by,
             norm_y_by          = net2_norm_y_by,
             sigma              = sigma,
             is_use_bias        = is_use_bias,
             is_norm_sigma      = is_norm_sigma,
             is_siren_init      = net2b_is_siren_init
             )
    if is_boundary:
        opt = {**train_opt, 'net1':net1_opt, 'net2':net2_opt, 'net2b':net2b_opt, 'net3':None }
    else:

        opt = {**train_opt, 'net1':net1_opt, 'net2':net2_opt, 'net3':None }
    
    poisson_train.train(opt,
                   is_absolute  = False,
                   is_set_seed  = is_set_seed,
                   seed_val     = seed_val)