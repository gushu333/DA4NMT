import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import wargs
from gru import GRU
from tools.utils import *
from models.losser import *

#DANN
from functions import ReverseLayerF

class NMT(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size):

        super(NMT, self).__init__()

        self.src_lookup_table = nn.Embedding(src_vocab_size, wargs.src_wemb_size, padding_idx=PAD)
        self.trg_lookup_table = nn.Embedding(trg_vocab_size, wargs.trg_wemb_size, padding_idx=PAD)

        self.encoder_common = Encoder(src_vocab_size, wargs.src_wemb_size, wargs.enc_hid_size, self.src_lookup_table)
        #'in' is the in domain private encoder
        self.encoder_in = Encoder(src_vocab_size, wargs.src_wemb_size_pri, wargs.enc_hid_size_pri, self.src_lookup_table)
        #'out' is the out of domain private encoder
        self.encoder_out = Encoder(src_vocab_size, wargs.src_wemb_size_pri, wargs.enc_hid_size_pri, self.src_lookup_table)

        self.s_init = nn.Linear(wargs.enc_hid_size, wargs.dec_hid_size)
        self.s_init_out = nn.Linear(wargs.enc_hid_size_pri, wargs.dec_hid_size_pri)
        self.s_init_in = nn.Linear(wargs.enc_hid_size_pri, wargs.dec_hid_size_pri)

        self.tanh = nn.Tanh()
        self.ha = nn.Linear(wargs.enc_hid_size, wargs.align_size)
        self.ha_in = nn.Linear(wargs.enc_hid_size_pri, wargs.align_size_pri)
        self.ha_out = nn.Linear(wargs.enc_hid_size_pri, wargs.align_size_pri)
        # as above
        self.decoder_common = Decoder(trg_vocab_size, self.trg_lookup_table)
        self.decoder_in = Decoder(trg_vocab_size, self.trg_lookup_table, com=False)
        self.decoder_out = Decoder(trg_vocab_size, self.trg_lookup_table, com=False)

        #DANN
        self.domain_discriminator = Domain_Discriminator()

        self.classifier = Classifier(wargs.out_size, trg_vocab_size, self.trg_lookup_table if wargs.copy_trg_emb is True else None)

    def init_state(self, xs_h, xs_mask=None, domain='IN'):

        assert xs_h.dim() == 3  # slen, batch_size, enc_size
        if xs_mask is not None:
            xs_h = (xs_h * xs_mask[:, :, None]).sum(0) / xs_mask.sum(0)[:, None]
        else:
            xs_h = xs_h.mean(0)


        if domain is 'IN':
            res = self.tanh(self.s_init_in(xs_h))
        elif domain is 'OUT':
            res = self.tanh(self.s_init_out(xs_h))
        else:
            res = self.tanh(self.s_init(xs_h))

        return res

    def init(self, xs, domain, xs_mask=None, test=True):

        if test is True and not isinstance(xs, tc.autograd.variable.Variable):  # for decoding
            if wargs.gpu_id and not xs.is_cuda: xs = xs.cuda()
            xs = Variable(xs, requires_grad=False, volatile=True)

        if domain is "IN":
            xs = self.encoder_in(xs, xs_mask)
            uh = self.ha_in(xs)
        elif domain is "OUT":
            xs = self.encoder_out(xs, xs_mask)
            uh = self.ha_out(xs)
        else:
            xs = self.encoder_common(xs, xs_mask)
            uh = self.ha(xs)

        s0 = self.init_state(xs, xs_mask, domain)
        return s0, xs, uh

    def forward(self, srcs, trgs, srcs_m, trgs_m, domain, isAtt=False, test=False, alpha=0, adv=False):
        #DANN
        
        # (max_slen_batch, batch_size, enc_hid_size)
        s0_private, srcs_private, uh_private = self.init(srcs, domain, srcs_m, test)
        s0_common, srcs_common, uh_common = self.init(srcs, "common", srcs_m, test)
        if adv is True:
            return self.domain_discriminator(s0_common, alpha)

        #!!!!!!!

        logit_com = self.decoder_common(s0_common, srcs_common, trgs, uh_common, srcs_m, trgs_m, isAtt=isAtt)
        if domain is "IN":
            logit_pri  = self.decoder_in(s0_private, srcs_private, trgs, uh_private, srcs_m, trgs_m, isAtt=isAtt)
        else:
            logit_pri = self.decoder_out(s0_private, srcs_private, trgs, uh_private, srcs_m, trgs_m, isAtt=isAtt)
        logit = logit_pri + logit_com

        def decoder_step_out(logit, max_out=True):
            if logit.dim() == 2:
                logit = logit.view(logit.size(0), logit.size(1)/2, 2)
            elif logit.dim() == 3:
                logit = logit.view(logit.size(0), logit.size(1), logit.size(2)/2, 2)
            logit = logit.max(-1)[0] if max_out else self.tanh(logit)
            logit = logit * trgs_m[:, :, None]
            return logit
            
        decoder_output = decoder_step_out(logit)
        domain_output = self.domain_discriminator(s0_common, alpha)

        return decoder_output, domain_output

        

class Encoder(nn.Module):

    '''
        Bi-directional Gated Recurrent Unit network encoder
    '''

    def __init__(self,
                 src_vocab_size,
                 input_size,
                 output_size,
                 src_lookup_table,
                 with_ln=False,
                 prefix='Encoder', **kwargs):

        super(Encoder, self).__init__()

        self.output_size = output_size
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        self.src_lookup_table = src_lookup_table

        self.forw_gru = GRU(input_size, output_size, with_ln=with_ln, prefix=f('Forw'))
        self.back_gru = GRU(output_size, output_size, with_ln=with_ln, prefix=f('Back'))

    def forward(self, xs, xs_mask=None, h0=None):

        max_L, b_size = xs.size(0), xs.size(1)  #40*80
        xs_e = xs if xs.dim() == 3 else self.src_lookup_table(xs)

        right = []
        h = h0 if h0 else Variable(tc.zeros(b_size, self.output_size), requires_grad=False)
        if wargs.gpu_id: h = h.cuda()
        for k in range(max_L):
            # (batch_size, src_wemb_size)
            h = self.forw_gru(xs_e[k], xs_mask[k] if xs_mask is not None else None, h)
            right.append(h)

        left = []
        h = h0 if h0 else Variable(tc.zeros(b_size, self.output_size), requires_grad=False)
        if wargs.gpu_id: h = h.cuda()
        for k in reversed(range(max_L)):
            h = self.back_gru(right[k], xs_mask[k] if xs_mask is not None else None, h)
            left.append(h)

        return tc.stack(left[::-1], dim=0)

class Attention(nn.Module):

    def __init__(self, dec_hid_size, align_size):

        super(Attention, self).__init__()
        self.align_size = align_size
        self.sa = nn.Linear(dec_hid_size, self.align_size)
        self.tanh = nn.Tanh()
        self.maskSoftmax = MaskSoftmax()
        self.a1 = nn.Linear(self.align_size, 1)

    def forward(self, s_tm1, xs_h, uh, xs_mask=None):
        d1, d2, d3 = uh.size()#?????
       # print self.sa(s_tm1)[None, :, :].size(), uh.size()
        _check_a1 = self.a1(self.tanh(self.sa(s_tm1)[None, :, :] + uh)).squeeze(2)
        e_ij = self.maskSoftmax(_check_a1, mask=xs_mask, dim=0)
        # weighted sum of the h_j: (b, enc_hid_size)
        attend = (e_ij[:, :, None] * xs_h).sum(0)

        return e_ij, attend

class Decoder(nn.Module):

    def __init__(self, trg_vocab_size, trg_lookup_table, max_out=True, com=True):

        super(Decoder, self).__init__()

        self.max_out = max_out
        self.attention = Attention(wargs.dec_hid_size if com else wargs.dec_hid_size_pri, wargs.align_size if com else wargs.align_size_pri)
        self.trg_lookup_table = trg_lookup_table
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.gru1 = GRU(wargs.trg_wemb_size if com else wargs.trg_wemb_size_pri, wargs.dec_hid_size if com else wargs.dec_hid_size_pri)
        self.gru2 = GRU(wargs.enc_hid_size if com else wargs.enc_hid_size_pri, wargs.dec_hid_size if com else wargs.dec_hid_size_pri)

        out_size = 2 * wargs.out_size if max_out else wargs.out_size
        self.ls = nn.Linear(wargs.dec_hid_size if com else wargs.dec_hid_size_pri, out_size)
        self.ly = nn.Linear(wargs.trg_wemb_size if com else wargs.trg_wemb_size_pri, out_size)
        self.lc = nn.Linear(wargs.enc_hid_size if com else wargs.enc_hid_size_pri, out_size)

        #self.classifier = Classifier(wargs.out_size, trg_vocab_size,
        #                            self.trg_lookup_table if wargs.copy_trg_emb is True else None)

    def step(self, s_tm1, xs_h, uh, y_tm1, xs_mask=None, y_mask=None):

        if not isinstance(y_tm1, tc.autograd.variable.Variable):
            if isinstance(y_tm1, int): y_tm1 = tc.Tensor([y_tm1]).long()
            elif isinstance(y_tm1, list): y_tm1 = tc.Tensor(y_tm1).long()
            if wargs.gpu_id: y_tm1 = y_tm1.cuda()
            y_tm1 = Variable(y_tm1, requires_grad=False, volatile=True)
            y_tm1 = self.trg_lookup_table(y_tm1)

        if xs_mask is not None and not isinstance(xs_mask, tc.autograd.variable.Variable):
            xs_mask = Variable(xs_mask, requires_grad=False, volatile=True)
            if wargs.gpu_id: xs_mask = xs_mask.cuda()

        s_above = self.gru1(y_tm1, y_mask, s_tm1)
        # (slen, batch_size) (batch_size, enc_hid_size)
        alpha_ij, attend = self.attention(s_above, xs_h, uh, xs_mask)
        s_t = self.gru2(attend, y_mask, s_above)

        return attend, s_t, y_tm1, alpha_ij

    def forward(self, s_tm1, xs_h, ys, uh, xs_mask, ys_mask, isAtt=False, ss_eps=1.):
        #self.decoder(s0, srcs, trgs, uh, srcs_m, trgs_m, isAtt=isAtt)
        tlen_batch_s, tlen_batch_y, tlen_batch_c = [], [], []
        y_Lm1, b_size = ys.size(0), ys.size(1)
        assert (xs_mask is not None) and (ys_mask is not None)

        if isAtt is True: attends = []
        # (max_tlen_batch - 1, batch_size, trg_wemb_size)
        ys_e = ys if ys.dim() == 3 else self.trg_lookup_table(ys)

        for k in range(y_Lm1):

            y_tm1 = ys_e[k]
        #    print k
            #attend is ci; s_tm1 is hiddenstate
            attend, s_tm1, _, alpha_ij = \
                    self.step(s_tm1, xs_h, uh, y_tm1, xs_mask, ys_mask[k])

            tlen_batch_c.append(attend)
            tlen_batch_y.append(y_tm1)
            tlen_batch_s.append(s_tm1)

            if isAtt is True: attends.append(alpha_ij)

        s = tc.stack(tlen_batch_s, dim=0)
        y = tc.stack(tlen_batch_y, dim=0)
        c = tc.stack(tlen_batch_c, dim=0)
        del tlen_batch_s, tlen_batch_y, tlen_batch_c

        logit = self.step_out(s, y, c)
        del s, y, c
        return logit
        '''        
        logit = logit * ys_mask[:, :, None]  # !!!!
        del s, y, c
        results = (logit, tc.stack(attends, 0)) if isAtt is True else logit

        return results
        '''
    def step_out(self, s, y, c):

        # (max_tlen_batch - 1, batch_size, dec_hid_size)
        logit = self.ls(s) + self.ly(y) + self.lc(c)
        # (max_tlen_batch - 1, batch_size, out_size)
        return logit
        '''        
        if logit.dim() == 2:    # for decoding
            logit = logit.view(logit.size(0), logit.size(1)/2, 2)
        elif logit.dim() == 3:
            logit = logit.view(logit.size(0), logit.size(1), logit.size(2)/2, 2)

        return logit.max(-1)[0] if self.max_out else self.tanh(logit)
        '''

class Domain_Discriminator(nn.Module):
    
    def __init__(self):
        super(Domain_Discriminator, self).__init__()
        self.domain_discriminator = nn.Sequential()
        self.domain_discriminator.add_module('d_fc1', nn.Linear(wargs.enc_hid_size, wargs.dis_hid_size))
        self.domain_discriminator.add_module('d_bn1', nn.BatchNorm2d(wargs.dis_hid_size))
        self.domain_discriminator.add_module('d_relu1', nn.ReLU(True))
        self.domain_discriminator.add_module('d_fc2', nn.Linear(wargs.dis_hid_size, wargs.dis_type))
        self.domain_discriminator.add_module('d_softmax', nn.Softmax())

    def forward(self, input, alpha):
        #reverse_input = ReverseLayerF.apply(input, alpha)
        domain_output = self.domain_discriminator(input)

        return domain_output
