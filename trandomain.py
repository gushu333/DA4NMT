#( -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import time
import argparse
import subprocess
import torch as tc
from torch import cuda

import wargs
from tools.inputs import Input
from tools.utils import *
from tools.utils import init_dir, wlog, _load_model
from translate import Translator
from inputs_handler import extract_vocab, val_wrap_data, wrap_data
from models.losser import *

def domain_out(model, src_input):
    batch_count = len(src_input)
    point_every, number_every = int(math.ceil(batch_count/100)), int(math.ceil(batch_count/10))
    total_domain = []
    sent_no, words_cnt = 0, 0

    fd_attent_matrixs, trgs = None, None

    trans_start = time.time()

    for bid in range(batch_count):
        src, srcm= src_input[bid][1], src_input[bid][4]
        domain = model(src, None, srcm, None, 'IN')
        total_domain.append(domain)
        if numpy.mod(sent_no + 1, point_every) == 0: wlog('.', False)
        if numpy.mod(sent_no + 1, number_every) == 0: wlog('{}'.format(sent_no + 1), False)

        sent_no += 1

    wlog('Done ...')
    return total_domain


if __name__ == "__main__":

    A = argparse.ArgumentParser(prog='NMT translator ... ')
    A.add_argument('--model-file', dest='model_file', help='model file')

    A.add_argument('--test-file', dest='test_file', default=None,
                   help='the input test file path we will translate')
    args = A.parse_args()
    model_file = args.model_file
    wlog('Using model: {}'.format(model_file))
    from models.rnnsearch import *

    src_vocab = extract_vocab(None, wargs.src_dict)
    trg_vocab = extract_vocab(None, wargs.trg_dict)

    src_vocab_size, trg_vocab_size = src_vocab.size(), trg_vocab.size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(src_vocab_size, trg_vocab_size))

    wlog('Start decoding ... init model ... ', 0)

    nmtModel = NMT(src_vocab_size, trg_vocab_size)
    if wargs.gpu_id:
        cuda.set_device(wargs.gpu_id[0])
        nmtModel.cuda()
        wlog('Push model onto GPU[{}] ... '.format(wargs.gpu_id[0]))
    else:
        nmtModel.cpu()
        wlog('Push model onto CPU ... ')

    assert os.path.exists(model_file), 'Requires pre-trained model'
    _dict = _load_model(model_file)
    # initializing parameters of interactive attention model
    class_dict = None
    if len(_dict) == 4: 
        model_dict, eid, bid, optim = _dict
    elif len(_dict) == 5:
        model_dict, class_dict, eid, bid, optim = _dict
    for name, param in nmtModel.named_parameters():
        if name in model_dict:
            param.requires_grad = not wargs.fix_pre_params
            param.data.copy_(model_dict[name])
            wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
        elif name.endswith('map_vocab.weight'):
            if class_dict is not None:
                param.requires_grad = not wargs.fix_pre_params
                param.data.copy_(class_dict['map_vocab.weight'])
                wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
        elif name.endswith('map_vocab.bias'):
            if class_dict is not None:
                param.requires_grad = not wargs.fix_pre_params
                param.data.copy_(class_dict['map_vocab.bias'])
                wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
        else: init_params(param, name, True)

    wlog('\nFinish to load model.')

    dec_conf()

    nmtModel.eval()

    input_file = '{}{}.{}'.format(wargs.val_tst_dir, args.test_file, wargs.val_src_suffix)
    ref_file = '{}{}.{}'.format(wargs.val_tst_dir, args.test_file, wargs.val_ref_suffix)
    #input_file = args.test_file

    wlog('Translating test file {} ... '.format(input_file))
    test_src_tlst, test_src_lens = val_wrap_data(input_file, src_vocab)
    test_input_data = Input(test_src_tlst, None, 1, volatile=True)

    batch_tst_data = None

    domain_out = domain_out(nmtModel, test_input_data).data


    if wargs.search_mode == 0: p1 = 'greedy'
    elif wargs.search_mode == 1: p1 = 'nbs'
    elif wargs.search_mode == 2: p1 = 'cp'
    p2 = 'GPU' if wargs.gpu_id else 'CPU'
    p3 = 'wb' if wargs.with_batch else 'wob'

    #test_file_name = input_file if '/' not in input_file else input_file.split('/')[-1]
    outdir = 'wexp-{}-{}-{}-{}-{}'.format(args.test_file, p1, p2, p3, model_file.split('/')[0])
    if wargs.ori_search: outdir = '{}-{}'.format(outdir, 'ori')
    init_dir(outdir)
    outprefix = outdir + '/trans_' + args.test_file
    # wTrans/trans
    file_out = "{}_e{}_upd{}_b{}m{}_bch{}".format(
        outprefix, eid, bid, wargs.beam_size, wargs.search_mode, wargs.with_batch)

    fout = open(file_out, 'w')
    fout.writelines(domain_out)
    fout.close()






