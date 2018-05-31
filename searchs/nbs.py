from __future__ import division

import sys
import copy
import numpy as np
import torch as tc
from torch.autograd import Variable

import wargs
from tools.utils import *

import time

class Nbs(object):

    def __init__(self, model, tvcb_i2w, k=10, ptv=None, noise=False, print_att=False):

        self.model = model
        self.decoder_common = model.decoder_common
        self.decoder_in = model.decoder_in
        self.decoder_out = model.decoder_out

        self.tvcb_i2w = tvcb_i2w
        self.k = k
        self.ptv = ptv
        self.noise = noise
        self.xs_mask = None
        self.print_att = print_att

        self.C = [0] * 4

    def beam_search_trans(self, s_list, domain):

        #print '-------------------- one sentence ............'
        self.srcL = len(s_list)
        self.maxL = 2 * self.srcL

        self.beam, self.hyps = [], []
        self.attent_probs = [] if self.print_att is True else None
        self.domain = domain

        s_tensor = tc.Tensor(s_list).long().unsqueeze(-1)

        # get initial state of decoder rnn and encoder context
        # s_tensor: (srcL, batch_size), batch_size==beamsize==1
        self.s0_private, self.enc_src0_private, self.uh0_private = self.model.init(s_tensor, domain, test=True)
        self.s0_common, self.enc_src0_common, self.uh0_common = self.model.init(s_tensor, "common", test=True)
        #if wargs.dec_layer_cnt > 1: self.s0 = [self.s0] * wargs.dec_layer_cnt
        # (1, trg_nhids), (src_len, 1, src_nhids*2)
        init_beam(self.beam, cnt=self.maxL, s0_com=self.s0_common, s0_pri=self.s0_private)

        best_trans, best_loss, attent_matrix = self.batch_search()
        # best_trans w/o <bos> and <eos> !!!

        debug('Src[{}], hyp (w/o EOS)[{}], maxL[{}], loss[{}]'.format(
            self.srcL, len(best_trans), self.maxL, best_loss))

        debug('Average location of bp [{}/{}={:6.4f}]'.format(
            self.C[1], self.C[0], self.C[1] / self.C[0]))
        debug('Step[{}] stepout[{}]'.format(*self.C[2:]))

        return filter_reidx(best_trans, self.tvcb_i2w), best_loss, attent_matrix

    ##################################################################

    # Wen Zhang: beam search, no batch

    ##################################################################
    #@exeTime
    def batch_search(self):

        # s0: (1, trg_nhids), enc_src0: (srcL, 1, src_nhids*2), uh0: (srcL, 1, align_size)
        enc_size = self.enc_src0_common.size(-1)
        L, align_size = self.srcL, wargs.align_size
        hyp_scores = np.zeros(1).astype('float32')
        delete_idx, prevb_id = None, None
        #batch_adj_list = [range(self.srcL) for _ in range(self.k)]

        debug('\nBeam-{} {}'.format(0, '-'*20))
        for b in self.beam[0]:    # do not output state
            debug(b[0:1] + b[-2:])

        for i in range(1, self.maxL + 1):

            debug('\n{} Step-{} {}'.format('#'*20, i, '#'*20))
            prevb = self.beam[i - 1]
            preb_sz = len(prevb)
            cnt_bp = (i >= 2)
            if cnt_bp: self.C[0] += preb_sz
            # batch states of previous beam, (preb_sz, 1, nhids) -> (preb_sz, nhids)
            s_im1_private = tc.stack(tuple([b[-3] for b in prevb]), dim=0).squeeze(1)
            s_im1_common = tc.stack(tuple([b[-4] for b in prevb]), dim=0).squeeze(1)
            y_im1 = [b[-2] for b in prevb]

            # (src_sent_len, 1, src_nhids) -> (src_sent_len, preb_sz, src_nhids)
            self.enc_src_private = self.enc_src0_private.view(L, -1, enc_size).expand(L, preb_sz, enc_size)
            self.enc_src_common = self.enc_src0_common.view(L, -1, enc_size).expand(L, preb_sz, enc_size)

            uh_private = self.uh0_private.view(L, -1, align_size).expand(L, preb_sz, align_size)
            uh_common = self.uh0_common.view(L, -1, align_size).expand(L, preb_sz, align_size)

            a_i_common, s_i_common, y_im1_common, alpha_ij_common = self.decoder_common.step(s_im1_common, self.enc_src_common, uh_common, y_im1)
            if self.domain is "OUT":
                a_i_private, s_i_private, y_im1_private, alpha_ij_private = self.decoder_out.step(s_im1_private, self.enc_src_private, uh_private, y_im1)
                logit_private = self.decoder_out.step_out(s_i_private, y_im1_private, a_i_private)
            elif self.domain is "IN":
                a_i_private, s_i_private, y_im1_private, alpha_ij_private = self.decoder_in.step(s_im1_private, self.enc_src_private, uh_private, y_im1)
                logit_private = self.decoder_in.step_out(s_i_private, y_im1_private, a_i_private)

            if self.attent_probs is not None: 
                self.attent_probs.append(alpha_ij_common)
              #  self.attent_probs.append(alpha_ij_private)
            self.C[2] += 1
            # (preb_sz, out_size)
            logit_common = self.decoder_common.step_out(s_i_common, y_im1_common, a_i_common)

            logit = logit_common + logit_private

            if logit.dim() == 2:    # for decoding
                logit = logit.view(int(logit.size(0)), int(logit.size(1)/2), int(2))
            elif logit.dim() == 3:
                logit = logit.view(logit.size(0), logit.size(1), logit.size(2)/2, 2)

            logit = logit.max(-1)[0] #if self.max_out else self.tanh(logit)
            
            self.C[3] += 1

            debug('For beam[{}], pre-beam ids: {}'.format(i - 1, prevb_id))
            # (preb_sz, vocab_size)
            next_ces = self.model.classifier(logit)
            next_ces = next_ces.cpu().data.numpy()
            cand_scores = hyp_scores[:, None] + next_ces
            cand_scores_flat = cand_scores.flatten()
            ranks_flat = part_sort(cand_scores_flat, self.k - len(self.hyps))
            voc_size = next_ces.shape[1]
            prevb_id = ranks_flat // voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_scores_flat[ranks_flat]

            debug('For beam [{}], pre-beam ids: {}'.format(i, prevb_id))

            tp_bid = tc.from_numpy(prevb_id).cuda() if wargs.gpu_id else tc.from_numpy(prevb_id)
            delete_idx = []
            for _j, b in enumerate(zip(costs, s_i_common[tp_bid], s_i_private[tp_bid], word_indices, prevb_id)):
                if cnt_bp: self.C[1] += (b[-1] + 1)
                if b[-2] == EOS:
                    #print len(self.batch_adj_list), b[-1]
                    delete_idx.append(b[-1])
                    if wargs.len_norm: self.hyps.append(((b[0] / i), b[0]) + b[-2:] + (i, ))
                    else: self.hyps.append((b[0], ) + b[-2:] + (i,))
                    debug('Gen hypo {}'.format(self.hyps[-1]))
                    # because i starts from 1, so the length of the first beam is 1, no <bos>
                    if len(self.hyps) == self.k:
                        # output sentence, early stop, best one in k
                        debug('Early stop! see {} hyps ending with EOS.'.format(self.k))
                        sorted_hyps = sorted(self.hyps, key=lambda tup: tup[0])
                        for hyp in sorted_hyps: debug('{}'.format(hyp))
                        best_hyp = sorted_hyps[0]
                        debug('Best hyp length (w/ EOS)[{}]'.format(best_hyp[-1]))

                        return back_tracking(self.beam, best_hyp, self.attent_probs)
                # should calculate when generate item in current beam
                else: self.beam[i].append(b)

            debug('\n{} Beam-{} {}'.format('-'*20, i, '-'*20))
            for b in self.beam[i]:    # do not output state
                debug(b[0:1] + b[-2:])
            hyp_scores = np.array([b[0] for b in self.beam[i]])

        # no early stop, back tracking
        return back_tracking(self.beam, self.no_early_best(), self.attent_probs)

    def no_early_best(self):

        # no early stop, back tracking
        if len(self.hyps) == 0:
            debug('No early stop, no hyp ending with EOS, select one length {} '.format(self.maxL))
            best_hyp = self.beam[self.maxL][0]
            if wargs.len_norm:
                best_hyp = (best_hyp[0]/self.maxL, best_hyp[0], ) + best_hyp[-2:] + (self.maxL, )
            else:
                best_hyp = (best_hyp[0], ) + best_hyp[-2:] + (self.maxL, )

        else:
            debug('No early stop, no enough {} hyps ending with EOS, select the best '
                  'one from {} hyps.'.format(self.k, len(self.hyps)))
            sorted_hyps = sorted(self.hyps, key=lambda tup: tup[0])
            for hyp in sorted_hyps: debug('{}'.format(hyp))
            best_hyp = sorted_hyps[0]

        debug('Best hyp length (w/ EOS)[{}]'.format(best_hyp[-1]))

        return best_hyp

