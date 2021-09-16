#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jd
contact: jamesduv@umich.edu
affiliation: University of Michigan, Department of Aerospace Eng., CASLAB
"""

import os, pickle, sys

def load_2dshapes_frame_processed(data_opt, is_absolute=True):
    '''Load the data frame which contains case info for all meshes of a given 
    size, where size is the number of random instances per shape.

    Read from preprocessed data storage location'''
    shape_dir, output_dir, mesh_dir, data_dir = get_2dshape_preprocessed_dirs(data_opt, is_absolute=is_absolute)
    fn_read     = os.path.join(output_dir, 'vertices_{:1.0f}_dframe.pickle'.format(data_opt['size']))
    dframe      = pickle.load(open(fn_read, 'rb'))
    return dframe


##TODO: streamline for use by others
def get_2dshape_preprocessed_dirs(data_opt, is_absolute=True):
    cfd_dir     = get_cfd_data_dir(is_absolute=is_absolute)
    shape_dir   = os.path.join(cfd_dir, '2dshapes')
    output_dir  = os.path.join(shape_dir, 'output_{:1.0f}'.format(data_opt['size']))

    mesh_local  = 'meshes_drs{:1.3e}_drw{:1.3e}'.format(data_opt['dr_surf'],
                                                        data_opt['dr_wall'])
    mesh_dir    = os.path.join(output_dir, mesh_local)
    data_dir    = os.path.join(mesh_dir, 'soln_poisson_split_sL{:1.3e}_sR{:1.3e}_bcw{:1.3e}_bcb{:1.3e}'
                                   .format(
                                    data_opt['Lsrc'],
                                    data_opt['Rsrc'],
                                    data_opt['bc_wall'],
                                    data_opt['bc_body'] ))
    return shape_dir, output_dir, mesh_dir, data_dir

def get_cfd_data_dir(is_absolute=True):
    '''Get the absolute path to preprocessed cfd folder'''
    if is_absolute:
        pathls  = ['/home','jd','HDD','Dropbox','Research','GCN_CNN_VAE',
                   'Preprocessed_CFD_Data']
    else:
        pathls = [sys.path[0], '..', 'Preprocessed_CFD_Data']

    d       = ''
    for idx, curpath in enumerate(pathls):
        d = os.path.join(d, curpath)
    return d


def make_model_save_dir(opt, save_dir_base, is_debugging):
    '''Make the directory for saving a model and its weights. 

    If in debugging mode, overwrite previous data. If not in debugging 
    mode, then raise an exception.

    Args:
        opt (dict) : dictionary containing the options for building the model
        save_dir_base (str) : relative path to the directory to place the
            new folder in
        is_debugging (bool) : option specifying operation mode.

    Returns:
        opt (dict) : the same dictionary as before, but with an entry added
            with key 'save_dir', pointing to the newly created directory
    '''

    save_dir = os.path.join(save_dir_base, opt['name'])

    #if debugging, delete save directory
    if is_debugging:
        if os.path.exists(save_dir):
            filelist = os.listdir(save_dir)
            for f in filelist:
                os.remove(os.path.join(save_dir, f))
            os.rmdir(save_dir)
    if os.path.exists(save_dir):
        raise Exception('Network already exists: {}'.format(opt['name']))
    else:
        os.mkdir(save_dir)
    opt['save_dir'] = save_dir
    return opt