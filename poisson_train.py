#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jd
contact: jamesduv@umich.edu
affiliation: University of Michigan, Department of Aerospace Eng., CASLAB
"""

import os, sys, pickle
import numpy as np
import tensorflow as tf
from shutil import copy
import meshio

import nids_networks as nids
import util
import norm_data

from nids_helper import build_net1_dense_schedule, bound_model_test, set_network_func

def train(opt,
                   is_absolute  = True,
                   is_set_seed  = True,
                   seed_val     = 0):
    '''Train a model on the 2dshapes Poisson dataset

    Args:
        opt (dict)      : setting for model constructions and training
        is_absolute (bool)  : data loading option for conflux vs. local, to be removed
        is_set_seed (bool)  : if True, set the seed before building training/validation groups
        seed_val (float)    : number to set seed'''

    tf.keras.backend.set_floatx('float64')
    opt     = util.make_model_save_dir(opt, opt['save_dir_base'],
                                         opt['is_debugging'])
    opt['n_cases_total']    = opt['data_opt']['size'] * opt['data_opt']['n_shapes']
    opt['net1']['n_input']  = len(opt['net1']['inputs'])
    opt['n_param']          = len(opt['net1']['inputs'])

    #original method, need databatch version
    #in simplest case, use same indices per shape
    opt = util.split_2dshape_idx(opt, is_set_seed = is_set_seed,
                                 seed_val = seed_val)
    if opt['is_batch_by_case']:
        if opt['is_data_batch']:
            raise Exception('Databatching not implemented for 2d shapes')
        else:
            opt, data, net_data   = build_2dshapes_data_dict_bbc(opt, is_absolute=is_absolute)
    elif opt['is_mixed_batches']:
        if opt['is_data_batch']:
            raise Exception('Databatching not implemented for 2d shapes')
        else:
            opt, data, net_data = build_2dshapes_data_mixed(opt, is_absolute=is_absolute)

    opt = build_net1_dense_schedule(opt)
    f_network = set_network_func(opt)
    net = f_network(opt = opt, data = net_data)
    if opt['is_siren']:
        net.siren_weight_init_v2()

    is_boundary_model = bound_model_test(opt)
    if is_boundary_model:
        opt = util.add_weight_info_nice_bound(opt = opt, model = net)
    else:
        opt = util.add_weight_info_nice2(opt = opt, model = net)
    util.save_opt(opt)

    if opt['is_batch_by_case']:
        epoch_schedule = [opt['epochs']]
        opt['epoch_schedule'] = epoch_schedule
        net.data = [net.data]
        net.train_model_batch_by_case_databatches()
    else:
        if is_boundary_model:
            net.train_model_betasep()
        else:
            net.train_model()

def build_2dshapes_data_dict_bbc(opt, is_absolute=True):
    '''Build the data dictionary for 2dshapes dataset using batch by case.

    Do not force batch sizes to be equal'''

    data_opt    = opt['data_opt']
    dframe = util.load_2dshapes_frame_processed(data_opt, is_absolute=is_absolute)

    data   = util.load_2dshapes_idx_processed(opt, is_absolute=is_absolute)

    #generate minibatch indices for each case
    opt   = build_minibatch_idx_2dshape(opt   = opt,
                                        data  = data)
    case_info = opt['case_info']

    # build flattened training matrices for computing stats
    x, y, x2 = util.build_2dshapes_mat_train(opt = opt, data = data)

    x_stats     = util.compute_stats_flat(x)
    y_stats     = util.compute_stats_flat(y)
    x2_stats    = util.compute_stats_flat(x2)
    del x, y, x2
    opt['data_stats'] = {
        'x_stats'   : x_stats,
        'x2_stats'  : x2_stats,
        'y_stats'   : y_stats
        }

    # #normalize training data
    train_data = {}

    for iCase, idx in enumerate(opt['idx_train']):
        point_data  = data[idx]['point_data']
        gen_fac     = data[idx]['generative_factors']
        n_points_cur  = case_info[iCase]['n_points']

        #construct and normalize input matrix x for current case
        xcur    = np.zeros((n_points_cur, opt['net2']['n_input']))
        xnorm   = np.zeros((n_points_cur, opt['net2']['n_input']))

        for iSignal, cursignal in enumerate(opt['net2']['inputs']):
            xcur[:,iSignal] = point_data[cursignal]

            #ensure sdf is normalized 0-1
            if 'sdf' in cursignal:
                method = 'range'
            else:
                method = opt['net2']['norm_x_by']
            xnorm[:,iSignal] = util.norm_single_by_stats_flat(x = xcur[:,iSignal],
                                                              x_stats = x_stats[iSignal],
                                                              method = method)

        #construct and normalize output matrix y for current case
        ycur    = np.zeros((n_points_cur, opt['net2']['n_output']))
        ynorm   = np.zeros((n_points_cur, opt['net2']['n_output']))
        for iSignal, cursignal in enumerate(opt['net2']['outputs']):
            ycur[:,iSignal] = point_data[cursignal]

            #ensure sdf is normalized 0-1
            if 'sdf' in cursignal:
                method = 'range'
            else:
                method = opt['net2']['norm_y_by']
            ynorm[:,iSignal] = util.norm_single_by_stats_flat(x = ycur[:,iSignal],
                                                              x_stats = y_stats[iSignal],
                                                              method = method)

        #construct and normalize input matrix x2 for current case
        x2cur   = np.zeros((1, opt['net1']['n_input']))
        x2norm  = np.zeros((1, opt['net1']['n_input']))
        for iSignal, cursignal in enumerate(opt['net1']['inputs']):
            x2cur[0, iSignal] = gen_fac[cursignal]

            method = opt['net1']['norm_x_by']
            x2norm[:,iSignal] = util.norm_single_by_stats_flat(x = x2cur[:,iSignal],
                                                              x_stats = x2_stats[iSignal],
                                                              method = method)

        train_data[iCase] = {
            'x'     : xnorm,
            'y'     : ynorm,
            'x2'    : x2norm
            }

    #normalize validation data
    val_data = {}
    for iCase, idx in enumerate(opt['idx_val']):

        point_data  = data[idx]['point_data']
        gen_fac     = data[idx]['generative_factors']
        n_points_cur  = point_data['q'].shape[0]

        #construct and normalize input matrix x for current case
        xcur    = np.zeros((n_points_cur, opt['net2']['n_input']))
        xnorm   = np.zeros((n_points_cur, opt['net2']['n_input']))
        for iSignal, cursignal in enumerate(opt['net2']['inputs']):
            xcur[:,iSignal] = point_data[cursignal]

            #ensure sdf is normalized 0-1
            if cursignal == 'sdf':
                method = 'range'
            else:
                method = opt['net2']['norm_x_by']
            xnorm[:,iSignal] = util.norm_single_by_stats_flat(x = xcur[:,iSignal],
                                                              x_stats = x_stats[iSignal],
                                                              method = method)

        #construct and normalize output matrix y for current case
        ycur    = np.zeros((n_points_cur, opt['net2']['n_output']))
        ynorm   = np.zeros((n_points_cur, opt['net2']['n_output']))
        for iSignal, cursignal in enumerate(opt['net2']['outputs']):
            ycur[:,iSignal] = point_data[cursignal]

            #ensure sdf is normalized 0-1
            if cursignal == 'sdf':
                method = 'range'
            else:
                method = opt['net2']['norm_y_by']
            ynorm[:,iSignal] = util.norm_single_by_stats_flat(x = ycur[:,iSignal],
                                                              x_stats = y_stats[iSignal],
                                                              method = method)
        #construct and normalize input matrix x2 for current case
        x2cur   = np.zeros((1, opt['net1']['n_input']))
        x2norm  = np.zeros((1, opt['net1']['n_input']))
        for iSignal, cursignal in enumerate(opt['net1']['inputs']):
            x2cur[0, iSignal] = gen_fac[cursignal]

            method = opt['net1']['norm_x_by']
            x2norm[:,iSignal] = util.norm_single_by_stats_flat(x = x2cur[:,iSignal],
                                                              x_stats = x2_stats[iSignal],
                                                              method = method)
        val_data[iCase] = {
            'x'     : xnorm,
            'y'     : ynorm,
            'x2'    : x2norm
            }
    net_data = {
        'train_data'    : train_data,
        'val_data'      : val_data
        }
    net_data = build_minibatches_2dshape(opt, net_data)

    return opt, data, net_data


def build_minibatch_idx_2dshape(opt, data):
    '''Compute the indices per minibatch for each case. Compute and store case 
    information in opt['case_info']. 

    Keys for case_info increase sequentially, but cases added are relative
    to idx_train, which has validation indices removed. For example, case_info[0] is for case idx_train[0]'''

    n_data_train    = 0
    case_info   = {}
    idx_train = opt['idx_train']

    for iCase, idx in enumerate(idx_train):

        n_points    = data[idx]['point_data']['xc'].shape[0]
        shape       = data[idx]['generative_factors']['shape']

        n_batches   =  int(np.ceil(n_points / opt['batch_size']))
        final_batch_size = n_points - (opt['batch_size'] * (n_batches-1))

        #generate set of indices for slicing minibatches
        mb_idx = []
        cur_mb = []
        idx1 = 0
        idx2 = idx1 + opt['batch_size']

        for ii in np.arange(n_batches - 1):
            mb_idx.append((idx1, idx2))
            idx1 = idx2
            idx2 = idx1 + opt['batch_size']

        if final_batch_size > 1:
            idx2 = idx1 + final_batch_size
            mb_idx.append((idx1, idx2))
        else:
            idx1 = idx1 - 1
            idx2 = idx1 + 1
            mb_idx.append((idx1, idx2))
        n_data_train += n_points

        case_info[iCase] = {
            'n_points'  : n_points,
            'n_train'   : n_points,
            'n_minibatches' : n_batches,
            'idx_minibatches' : mb_idx,
            'shape_class'   : shape,
            'idx_master'    : idx}

    opt['case_info']    = case_info
    opt['n_data_train'] = n_data_train

    return opt

def build_minibatches_2dshape(opt, net_data):
    '''Build minibatches, store as a list of tuples (x2, x, y) for all training data'''

    mb = []
    for iCase, idx in enumerate(opt['idx_train']):
        cur_info    = opt['case_info'][iCase]
        x2           = net_data['train_data'][iCase]['x2']
        x            = net_data['train_data'][iCase]['x']
        y            = net_data['train_data'][iCase]['y']

        for iBatch, (idx1, idx2) in enumerate(cur_info['idx_minibatches']):
            xcur    = x[idx1:idx2,:]
            ycur    = y[idx1:idx2,:]
            mb.append((x2, xcur, ycur))

    net_data['minibatches'] = mb

    return net_data

def build_2dshapes_data_mixed(opt, is_absolute=True):
    '''Build the data dictionary for 2dshapes dataset using fully mixed batching'''

    data_opt    = opt['data_opt']
    dframe = util.load_2dshapes_frame_processed(data_opt, is_absolute=is_absolute)

    data        = util.load_2dshapes_idx_processed(opt, is_absolute=is_absolute)

    #build matrices, save as dictionary with index as key
    opt, data_mat = util.build_2dshapes_mat_keys(opt, data)

    #build flattened matrices
    x_train, y_train, x2_train, x_val, y_val, x2_val, x2_train_norep = build_2dshapes_matrices_mixed(opt, data_mat)

    #compute data stats
    x_stats     = util.compute_stats_flat(x_train)
    y_stats     = util.compute_stats_flat(y_train)
    x2_stats    = util.compute_stats_flat(x2_train_norep)
    opt['data_stats'] = {
        'x'   : x_stats,
        'x2'  : x2_stats,
        'y'   : y_stats
        }
    #concatenate to normalize
    x = np.concatenate((x_train, x_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    x2 = np.concatenate((x2_train, x2_val), axis=0)

    #normalize all data
    x_norm, x2_norm, y_norm = util.norm_by_stats_2dshapes_mixed(opt, x, y, x2)

    #split back into training and validation matrices
    n_data_train = opt['n_data_train']
    x_train = x_norm[:n_data_train,:]
    y_train = y_norm[:n_data_train,:]
    x2_train = x2_norm[:n_data_train,:]

    x_val   = x_norm[n_data_train:,:]
    y_val   = y_norm[n_data_train:,:]
    x2_val  = x2_norm[n_data_train:,:]

    net_data = {'x_train'   : x_train,
                'y_train'   : y_train,
                'x2_train'  : x2_train,
                'x_val'     : x_val,
                'y_val'     : y_val,
                'x2_val'    : x2_val}
    return opt, data, net_data