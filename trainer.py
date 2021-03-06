from __future__ import division

import os
import sys
import math
import time
import subprocess

import numpy as np
import torch as tc
from torch.autograd import Variable

import wargs
from tools.utils import *
from translate import Translator

class Trainer(object):

    def __init__(self, model, train_data, train_data_domain, vocab_data, optim, valid_data=None, tests_data=None):

        self.model = model
        self.train_data = train_data
        self.train_data_domain = train_data_domain
        self.sv = vocab_data['src'].idx2key
        self.tv = vocab_data['trg'].idx2key
        self.optim = optim
        self.valid_data = valid_data
        self.tests_data = tests_data

        self.model.train()

    def mt_eval(self, eid, bid, domain="IN"):

        state_dict = { 'model': self.model.state_dict(), 'epoch': eid, 'batch': bid, 'optim': self.optim }

        if wargs.save_one_model: model_file = '{}.pt'.format(wargs.model_prefix)
        else: model_file = '{}_e{}_upd{}.pt'.format(wargs.model_prefix, eid, bid)
        tc.save(state_dict, model_file)

        self.model.eval()
        #self.model.classifier.eval()

        tor0 = Translator(self.model, self.sv, self.tv, print_att=wargs.print_att)
        bleu = tor0.trans_eval(self.valid_data, eid, bid, model_file, self.tests_data, domain)

        self.model.train()

        return bleu

    def train(self):

        wlog('Start training ... ')
        assert wargs.sample_size < wargs.batch_size, 'Batch size < sample count'
        # [low, high)
        batch_count = len(self.train_data) if wargs.fine_tune is False else 0
        batch_count_domain = len(self.train_data_domain)
        batch_start_sample = tc.randperm(batch_count+batch_count_domain)[0]
        wlog('Randomly select {} samples in the {}th/{} batch'.format(wargs.sample_size, batch_start_sample, batch_count+batch_count_domain))
        bidx, eval_cnt = 0, [0]
        wlog('Self-normalization alpha -> {}'.format(wargs.self_norm_alpha))

        train_start = time.time()
        wlog('')
        wlog('#' * 120)
        wlog('#' * 30, False)
        wlog(' Start Training ', False)
        wlog('#' * 30)
        wlog('#' * 120)
        #DANN
        loss_domain = tc.nn.NLLLoss()

        for epoch in range(wargs.start_epoch, wargs.max_epochs + 1):

            epoch_start = time.time()

            # train for one epoch on the training data
            wlog('')
            wlog('$' * 30, False)
            wlog(' Epoch [{}/{}] '.format(epoch, wargs.max_epochs), False)
            wlog('$' * 30)

            if wargs.epoch_shuffle and epoch > wargs.epoch_shuffle_minibatch:
                self.train_data.shuffle()
                self.train_data_domain.shuffle()
            # shuffle the original batch
            shuffled_batch_idx = tc.randperm(batch_count+batch_count_domain)

            sample_size = wargs.sample_size
            epoch_loss, epoch_trg_words, epoch_num_correct = 0, 0, 0
            show_loss, show_src_words, show_trg_words, show_correct_num = 0, 0, 0, 0
            domain_loss = 0
            sample_spend, eval_spend, epoch_bidx = 0, 0, 0
            show_start = time.time()

            for name, par in self.model.named_parameters():
                if name.split('.')[0] != "domain_discriminator":
                    par.requires_grad = True
                else:
                    par.requires_grad = False

            for k in range(batch_count + batch_count_domain):

                bidx += 1
                epoch_bidx = k + 1
                batch_idx = shuffled_batch_idx[k] if epoch >= wargs.epoch_shuffle_minibatch else k
                #p = float(k + epoch * (batch_count+batch_count_domain)) / wargs.max_epochs / (batch_count+batch_count_domain)
                #alpha = 2. / (1. + np.exp(-10 * p)) - 1
                if batch_idx < batch_count:

                    # (max_slen_batch, batch_size)
                    _, srcs, trgs, slens, srcs_m, trgs_m = self.train_data[batch_idx]

                    self.model.zero_grad()
                    # (max_tlen_batch - 1, batch_size, out_size)
                    #forward compute DANN out of domain
                    decoder_outputs, domain_outputs = self.model(srcs, trgs[:-1], srcs_m, trgs_m[:-1], 'OUT', alpha=wargs.alpha)
                    if len(decoder_outputs) == 2: (decoder_outputs, _checks) = decoder_outputs
                    this_bnum = decoder_outputs.size(1)
                
                    #batch_loss, grad_output, batch_correct_num = memory_efficient(
                    #    outputs, trgs[1:], trgs_m[1:], self.model.classifier)
                    #backward compute, now we have the grad
                    batch_loss, batch_correct_num, batch_log_norm = self.model.classifier.snip_back_prop(
                        decoder_outputs, trgs[1:], trgs_m[1:], wargs.snip_size)

                    #self.model.zero_grad()
                    #domain_outputs = self.model(srcs, None, srcs_m, None, 'OUT', alpha=alpha)
                    #if wargs.cross_entropy is True:
                     #   domain_label = tc.zeros(len(domain_outputs))
                      #  domain_label = domain_label.long()
                       # domain_label = domain_label.cuda()
                        #domainv_label = Variable(domain_label, volatile=False)
                
                    #    domain_loss_a = loss_domain(tc.log(domain_outputs), domainv_label)
                     #   domain_loss_a.backward(retain_graph = True if wargs.max_entropy is True else False)
                    if wargs.max_entropy is True:
                        #try to max this entropy
                        lam = wargs.alpha if epoch > 1 else 0
                        domain_loss_b = lam * tc.dot(tc.log(domain_outputs), domain_outputs)
                        domain_loss_b.backward()

                    batch_src_words = srcs.data.ne(PAD).sum()
                    assert batch_src_words == slens.data.sum()
                    batch_trg_words = trgs[1:].data.ne(PAD).sum()

                elif batch_idx >= batch_count:

                    #domain_loss_out.backward(retain_graph=True)
                    #DANN in domain compute
                    _, srcs_domain, trgs_domain, slens_domain, srcs_m_domain, trgs_m_domain = self.train_data_domain[batch_idx - batch_count]
                    self.model.zero_grad()
                    decoder_outputs, domain_outputs = self.model(srcs_domain, trgs_domain[:-1], srcs_m_domain, trgs_m_domain[:-1], 'IN', alpha=wargs.alpha)
                    if len(decoder_outputs) == 2: (decoder_outputs, _checks) = decoder_outputs
                    this_bnum = decoder_outputs.size(1)
                    batch_loss, batch_correct_num, batch_log_norm = self.model.classifier.snip_back_prop(
                       decoder_outputs, trgs_domain[1:], trgs_m_domain[1:], wargs.snip_size)
                    #domain_outputs = self.model(srcs_domain, None, srcs_m_domain, None, 'IN', alpha=alpha)
                    #if wargs.cross_entropy is True:
                     #   domain_label = tc.ones(len(domain_outputs))
                      #  domain_label = domain_label.long()
                       # domain_label = domain_label.cuda()
                        #domainv_label = Variable(domain_label, volatile=False)
                
                    #    domain_loss_a = loss_domain(tc.log(domain_outputs), domainv_label)
                     #   domain_loss_a.backward(retain_graph = True if wargs.max_entropy is True else False)
                    if wargs.max_entropy is True:
                        lam = wargs.alpha if epoch > 1 else 0
                        domain_loss_b = lam * tc.dot(tc.log(domain_outputs), domain_outputs)
                        domain_loss_b.backward()

                    batch_src_words = srcs_domain.data.ne(PAD).sum()
                    assert batch_src_words == slens_domain.data.sum()
                    batch_trg_words = trgs_domain[1:].data.ne(PAD).sum()

                _grad_nan = False
                for n, p in self.model.named_parameters():
                    if p.grad is None:
                        debug('grad None | {}'.format(n))
                        continue
                    tmp_grad = p.grad.data.cpu().numpy()
                    if numpy.isnan(tmp_grad).any(): # we check gradient here for vanishing Gradient
                        wlog("grad contains 'nan' | {}".format(n))
                        #wlog("gradient\n{}".format(tmp_grad))
                        _grad_nan = True
                    if n == 'decoder.l_f1_0.weight' or n == 's_init.weight' or n=='decoder.l_f1_1.weight' \
                       or n == 'decoder.l_conv.0.weight' or n == 'decoder.l_f2.weight':
                        debug('grad zeros |{:5} {}'.format(str(not np.any(tmp_grad)), n))

                if _grad_nan is True and wargs.dynamic_cyk_decoding is True:
                    for _i, items in enumerate(_checks):
                        wlog('step {} Variable----------------:'.format(_i))
                        #for item in items: wlog(item.cpu().data.numpy())
                        wlog('wen _check_tanh_sa ---------------')
                        wlog(items[0].cpu().data.numpy())
                        wlog('wen _check_a1_weight ---------------')
                        wlog(items[1].cpu().data.numpy())
                        wlog('wen _check_a1 ---------------')
                        wlog(items[2].cpu().data.numpy())
                        wlog('wen alpha_ij---------------')
                        wlog(items[3].cpu().data.numpy())
                        wlog('wen before_mask---------------')
                        wlog(items[4].cpu().data.numpy())
                        wlog('wen after_mask---------------')
                        wlog(items[5].cpu().data.numpy())

                #outputs.backward(grad_output)
                self.optim.step()
                #del outputs, grad_output


                show_loss += batch_loss
                #domain_loss += domain_loss_a.data.clone()[0]
                #domain_loss += domain_loss_out
                show_correct_num += batch_correct_num
                epoch_loss += batch_loss
                epoch_num_correct += batch_correct_num
                show_src_words += batch_src_words
                show_trg_words += batch_trg_words
                epoch_trg_words += batch_trg_words

                batch_log_norm = tc.mean(tc.abs(batch_log_norm))

                if epoch_bidx % wargs.display_freq == 0:
                    #print show_correct_num, show_loss, show_trg_words, show_loss/show_trg_words
                    ud = time.time() - show_start - sample_spend - eval_spend
                    wlog(
                        'Epo:{:>2}/{:>2} |[{:^5} {:^5} {:^5}k] |acc:{:5.2f}% |ppl:{:4.2f} '
                        '| |logZ|:{:.2f} '
                        '|stok/s:{:>4}/{:>2}={:>2} |ttok/s:{:>2} '
                        '|stok/sec:{:6.2f} |ttok/sec:{:6.2f} |elapsed:{:4.2f}/{:4.2f}m'.format(
                            epoch, wargs.max_epochs, epoch_bidx, batch_idx, bidx/1000,
                            (show_correct_num / show_trg_words) * 100,
                            math.exp(show_loss / show_trg_words), batch_log_norm,
                            batch_src_words, this_bnum, int(batch_src_words / this_bnum),
                            int(batch_trg_words / this_bnum),
                            show_src_words / ud, show_trg_words / ud, ud,
                            (time.time() - train_start) / 60.)
                    )
                    show_loss, show_src_words, show_trg_words, show_correct_num = 0, 0, 0, 0
                    sample_spend, eval_spend = 0, 0
                    show_start = time.time()

                if epoch_bidx % wargs.sampling_freq == 0:

                    sample_start = time.time()
                    self.model.eval()
                    #self.model.classifier.eval()
                    tor = Translator(self.model, self.sv, self.tv)

                    if batch_idx < batch_count:
                        # (max_len_batch, batch_size)
                        sample_src_tensor = srcs.t()[:sample_size]
                        sample_trg_tensor = trgs.t()[:sample_size]
                        tor.trans_samples(sample_src_tensor, sample_trg_tensor, "OUT")

                    elif batch_idx >= batch_count:
                        sample_src_tensor = srcs_domain.t()[:sample_size]
                        sample_trg_tensor = trgs_domain.t()[:sample_size]
                        tor.trans_samples(sample_src_tensor, sample_trg_tensor, "IN")
                    wlog('')
                    sample_spend = time.time() - sample_start
                    self.model.train()

                # Just watch the translation of some source sentences in training data
                if wargs.if_fixed_sampling and bidx == batch_start_sample:
                    # randomly select sample_size sample from current batch
                    rand_rows = np.random.choice(this_bnum, sample_size, replace=False)
                    sample_src_tensor = tc.Tensor(sample_size, srcs.size(0)).long()
                    sample_src_tensor.fill_(PAD)
                    sample_trg_tensor = tc.Tensor(sample_size, trgs.size(0)).long()
                    sample_trg_tensor.fill_(PAD)

                    for id in xrange(sample_size):
                        sample_src_tensor[id, :] = srcs.t()[rand_rows[id], :]
                        sample_trg_tensor[id, :] = trgs.t()[rand_rows[id], :]

                if wargs.epoch_eval is not True and bidx > wargs.eval_valid_from and \
                   bidx % wargs.eval_valid_freq == 0:
                    eval_start = time.time()
                    eval_cnt[0] += 1
                    wlog('\nAmong epoch, batch [{}], [{}] eval save model ...'.format(
                        epoch_bidx, eval_cnt[0]))

                    self.mt_eval(epoch, epoch_bidx, "IN")

                    eval_spend = time.time() - eval_start



            for name, par in self.model.named_parameters():
                if name.split('.')[0] != "domain_discriminator":
                    par.requires_grad = False
                else:
                    par.requires_grad = True

            for k in range(batch_count + batch_count_domain):
                epoch_bidx = k + 1
                batch_idx = shuffled_batch_idx[k] if epoch >= wargs.epoch_shuffle_minibatch else k

                if batch_idx < batch_count:

                    _, srcs, trgs, slens, srcs_m, trgs_m = self.train_data[batch_idx]

                    self.model.zero_grad()
                    # (max_tlen_batch - 1, batch_size, out_size)
                    #forward compute DANN out of domain
                    domain_outputs = self.model(srcs, trgs[:-1], srcs_m, trgs_m[:-1], 'OUT', alpha=wargs.alpha, adv=True)
                    #if len(decoder_outputs) == 2: (decoder_outputs, _checks) = decoder_outputs
                    #this_bnum = decoder_outputs.size(1)
                
                    #batch_loss, grad_output, batch_correct_num = memory_efficient(
                    #    outputs, trgs[1:], trgs_m[1:], self.model.classifier)
                    #backward compute, now we have the grad
                    #batch_loss, batch_correct_num, batch_log_norm = self.model.classifier.snip_back_prop(
                     #   decoder_outputs, trgs[1:], trgs_m[1:], wargs.snip_size)

                    #self.model.zero_grad()
                    #domain_outputs = self.model(srcs, None, srcs_m, None, 'OUT', alpha=alpha)
                    if wargs.cross_entropy is True:
                        domain_label = tc.zeros(len(domain_outputs))
                        domain_label = domain_label.long()
                        domain_label = domain_label.cuda()
                        domainv_label = Variable(domain_label, volatile=False)
                
                        domain_loss_a = loss_domain(tc.log(domain_outputs), domainv_label)
                        domain_loss_a.backward()
                    #if wargs.max_entropy is True:
                        #try to max this entropy
                     #   domain_loss_b = -wargs.alpha*tc.dot(tc.log(domain_outputs), domain_outputs)
                      #  domain_loss_b.backward()

                    batch_src_words = srcs.data.ne(PAD).sum()
                    assert batch_src_words == slens.data.sum()
                    batch_trg_words = trgs[1:].data.ne(PAD).sum()

                elif batch_idx >= batch_count:

                    #domain_loss_out.backward(retain_graph=True)
                    #DANN in domain compute
                    _, srcs_domain, trgs_domain, slens_domain, srcs_m_domain, trgs_m_domain = self.train_data_domain[batch_idx - batch_count]
                    self.model.zero_grad()
                    domain_outputs = self.model(srcs_domain, trgs_domain[:-1], srcs_m_domain, trgs_m_domain[:-1], 'IN', alpha=wargs.alpha, adv=True)
                    #if len(decoder_outputs) == 2: (decoder_outputs, _checks) = decoder_outputs
                    #this_bnum = decoder_outputs.size(1)
                    #batch_loss, batch_correct_num, batch_log_norm = self.model.classifier.snip_back_prop(
                     #  decoder_outputs, trgs_domain[1:], trgs_m_domain[1:], wargs.snip_size)
                    #domain_outputs = self.model(srcs_domain, None, srcs_m_domain, None, 'IN', alpha=alpha)
                    if wargs.cross_entropy is True:
                        domain_label = tc.ones(len(domain_outputs))
                        domain_label = domain_label.long()
                        domain_label = domain_label.cuda()
                        domainv_label = Variable(domain_label, volatile=False)
                
                        domain_loss_a = loss_domain(tc.log(domain_outputs), domainv_label)
                        domain_loss_a.backward(retain_graph = True if wargs.max_entropy is True else False)
                    #if wargs.max_entropy is True:
                     #   domain_loss_b = -wargs.alpha*tc.dot(tc.log(domain_outputs), domain_outputs)
                      #  domain_loss_b.backward()

                    batch_src_words = srcs_domain.data.ne(PAD).sum()
                    assert batch_src_words == slens_domain.data.sum()
                    batch_trg_words = trgs_domain[1:].data.ne(PAD).sum()

                self.optim.step()

                show_loss += batch_loss
                #domain_loss += domain_loss_a.data.clone()[0]
                #domain_loss += domain_loss_out
                show_correct_num += batch_correct_num
                epoch_loss += batch_loss
                epoch_num_correct += batch_correct_num
                show_src_words += batch_src_words
                show_trg_words += batch_trg_words
                epoch_trg_words += batch_trg_words

                #batch_log_norm = tc.mean(tc.abs(batch_log_norm))

                if epoch_bidx % wargs.display_freq == 0:
                    #print show_correct_num, show_loss, show_trg_words, show_loss/show_trg_words
                    ud = time.time() - show_start - sample_spend - eval_spend
                    wlog(
                        'Epo:{:>2}/{:>2} |[{:^5} {:^5} {:^5}k] |acc:{:5.2f}% |ppl:{:4.2f} '
                        '| |logZ|:{:.2f} '
                        '|stok/s:{:>4}/{:>2}={:>2} |ttok/s:{:>2} '
                        '|stok/sec:{:6.2f} |ttok/sec:{:6.2f} |elapsed:{:4.2f}/{:4.2f}m'.format(
                            epoch, wargs.max_epochs, epoch_bidx, batch_idx, bidx/1000,
                            (show_correct_num / show_trg_words) * 100,
                            math.exp(show_loss / show_trg_words), 0,
                            batch_src_words, this_bnum, int(batch_src_words / this_bnum),
                            int(batch_trg_words / this_bnum),
                            show_src_words / ud, show_trg_words / ud, ud,
                            (time.time() - train_start) / 60.)
                    )
                    show_loss, show_src_words, show_trg_words, show_correct_num = 0, 0, 0, 0
                    sample_spend, eval_spend = 0, 0
                    show_start = time.time()

                if epoch_bidx % wargs.sampling_freq == 0:

                    sample_start = time.time()
                    self.model.eval()
                    #self.model.classifier.eval()
                    tor = Translator(self.model, self.sv, self.tv)

                    if batch_idx < batch_count:
                        # (max_len_batch, batch_size)
                        sample_src_tensor = srcs.t()[:sample_size]
                        sample_trg_tensor = trgs.t()[:sample_size]
                        tor.trans_samples(sample_src_tensor, sample_trg_tensor, "OUT")

                    elif batch_idx >= batch_count:
                        sample_src_tensor = srcs_domain.t()[:sample_size]
                        sample_trg_tensor = trgs_domain.t()[:sample_size]
                        tor.trans_samples(sample_src_tensor, sample_trg_tensor, "IN")
                    wlog('')
                    sample_spend = time.time() - sample_start
                    self.model.train()

                # Just watch the translation of some source sentences in training data
                if wargs.if_fixed_sampling and bidx == batch_start_sample:
                    # randomly select sample_size sample from current batch
                    rand_rows = np.random.choice(this_bnum, sample_size, replace=False)
                    sample_src_tensor = tc.Tensor(sample_size, srcs.size(0)).long()
                    sample_src_tensor.fill_(PAD)
                    sample_trg_tensor = tc.Tensor(sample_size, trgs.size(0)).long()
                    sample_trg_tensor.fill_(PAD)

                    for id in xrange(sample_size):
                        sample_src_tensor[id, :] = srcs.t()[rand_rows[id], :]
                        sample_trg_tensor[id, :] = trgs.t()[rand_rows[id], :]



            avg_epoch_loss = epoch_loss / epoch_trg_words
            avg_epoch_acc = epoch_num_correct / epoch_trg_words
            wlog('\nEnd epoch [{}]'.format(epoch))
            wlog('Train accuracy {:4.2f}%'.format(avg_epoch_acc * 100))
            wlog('Average loss {:4.2f}'.format(avg_epoch_loss))
            wlog('Train perplexity: {0:4.2f}'.format(math.exp(avg_epoch_loss)))
            #wlog('Epoch domain loss is {:4.2f}'.format(float(domain_loss)))

            wlog('End epoch, batch [{}], [{}] eval save model ...'.format(epoch_bidx, eval_cnt[0]))
            mteval_bleu = self.mt_eval(epoch, epoch_bidx, "IN")
            self.optim.update_learning_rate(mteval_bleu, epoch)

            epoch_time_consume = time.time() - epoch_start
            wlog('Consuming: {:4.2f}s'.format(epoch_time_consume))

        wlog('Train finished, comsuming {:6.2f} hours'.format((time.time() - train_start) / 3600))
        wlog('Congratulations!')

