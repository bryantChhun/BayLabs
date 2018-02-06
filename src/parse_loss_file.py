# -*- coding: utf-8 -*-
"""
Created on 01 February, 2018 @ 12:25 AM

@author: Bryant Chhun
email: bchhun@gmail.com

Project: testing
License: 
"""

import numpy as np

def parse_loss(inputfile):
    npy = np.load(inputfile)
    npy_list = list(npy)
    epoch, loss, acc, val_loss, val_acc = [], [], [], [], []
    for k in range(int(len(npy_list)/5)):
        epoch.append(k)
        loss.append(npy_list[5 * k + 1])
        acc.append(npy_list[5 * k + 2])
        val_loss.append(npy_list[5 * k + 3])
        val_acc.append(npy_list[5 * k + 4])
    return epoch, loss, acc, val_loss, val_acc

def write_loss(inputfile):
    epoch, loss, acc, val_loss, val_acc = parse_loss(inputfile)
    with open('./loss_files/epoch.txt', 'w') as target:
        for e in epoch:
            target.write(str(e)+'\n')
        target.close()
    with open('./loss_files/loss.txt', 'w') as target:
        for l in loss:
            target.write(str(l)+'\n')
        target.close()
    with open('./loss_files/acc.txt', 'w') as target:
        for a in acc:
            target.write(str(a)+'\n')
        target.close()
    with open('./loss_files/val_loss.txt', 'w') as target:
        for vl in val_loss:
            target.write(str(vl)+'\n')
        target.close()
    with open('./loss_files/val_acc.txt', 'w') as target:
        for va in val_acc:
            target.write(str(va)+'\n')
        target.close()

