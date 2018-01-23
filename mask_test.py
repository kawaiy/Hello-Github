#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer


def mask_sampling(igo, batchsize, lam, igo_mode='update'):
    xp = chainer.cuda.get_array_module(igo.theta)
    m = mask= None
    #print('batchsize',batchsize)
    # IGO sampling
    if igo_mode == 'update':
        if batchsize % lam != 0:
            print('Batch size (%d) cannot divided by lambda (%d)!' % (batchsize, lam))
            exit(1)
        m = igo.sampling(lam)
        reps = np.repeat(batchsize // lam, lam)
        mask = xp.repeat(m, reps, axis=0)

    elif igo_mode == 'sampling':
        mask = igo.sampling(batchsize)

    elif igo_mode == 'mle':
        mm = igo.theta >= 0.5
        mask = xp.repeat(mm[xp.newaxis, :], batchsize, axis=0)

    else:
        print('Undifined igo_mode!')
        exit(1)

    return mask, m

