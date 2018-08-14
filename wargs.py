
volatile = False
log_norm = False


# Maximal sequence length in training data
max_seq_len = 100

'''
Embedding layer
'''
# Size of word embedding of source word and target word
src_wemb_size = 512
trg_wemb_size = 512
src_wemb_size_pri = 512
trg_wemb_size_pri = 512
'''
Encoder layer
'''
# Size of hidden units in encoder
enc_hid_size = 512
enc_hid_size_pri = 128

'''
Attention layer
'''
# Size of alignment vector
align_size = 512
align_size_pri = 128

'''
Decoder layer
'''
# Size of hidden units in decoder
dec_hid_size = 512
dec_hid_size_pri = 128
# Size of the output vector
out_size = 512

drop_rate = 0.5

# Directory to save model, test output and validation output
dir_model = 'wmodel'
dir_valid = 'wvalid'
dir_tests = 'wtests'

# Validation data
val_shuffle = True
#val_tst_dir = '/home5/wen/2.data/mt/nist_data_stanseg/'
#val_tst_dir = '/home/wen/3.corpus/wmt2017/de-en/'
val_tst_dir = './data/'

#val_prefix = 'nist02'
val_prefix = 'devset'
#val_prefix = 'newstest2014.tc'
#val_src_suffix = 'src'
#val_ref_suffix = 'ref.plain_'
val_src_suffix = 'src'
val_ref_suffix = 'trg'
#val_src_suffix = 'en'
#val_ref_suffix = 'de'
ref_cnt = 16

tests_prefix = None
#tests_prefix = ['nist03', 'nist04', 'nist05', 'nist06', 'nist08']
#tests_prefix = ['devset3.lc', '900']
#tests_prefix = ['test']
#tests_prefix = ['newstest2015.tc', 'newstest2016.tc', 'newstest2017.tc']

# Training data
train_shuffle = True
batch_size = 30
sort_k_batches = 10

# Data path
dir_data = 'data/'
train_src = dir_data + 'train_outdomain.src'
train_trg = dir_data + 'train_outdomain.trg'

#DANN
train_src_domain = dir_data + 'train_indomain.src'
train_trg_domain = dir_data + 'train_indomain.trg'
src_domain_vocab_from = train_src_domain
trg_domain_vocab_from = train_trg_domain
#src_domain_dict_size = 30000
src_domain_dict = dir_data + 'src.dict.domain.tcf'

# Dictionary
src_vocab_from = train_src
trg_vocab_from = train_trg
src_dict_size = 50000
trg_dict_size = 50000
src_dict = dir_data + 'src.dict.tcf'
trg_dict = dir_data + 'trg.dict.tcf'

inputs_data = dir_data + 'inputs.pt'

# Training
max_epochs = 20

epoch_shuffle = True
epoch_shuffle_minibatch = 1

small = True

display_freq = 100 if small else 1000
sampling_freq = 100 if small else 5000
sample_size = 3
if_fixed_sampling = False

epoch_eval = False
final_test = False
eval_valid_from = 210000 if small else 50000
eval_valid_freq = 4000 if small else 20000

save_one_model = False
start_epoch = 1

model_prefix = dir_model + '/model'
best_model = dir_valid + '/best.model.pt' if dir_valid else 'best.model.pt'
# pretrained model
pre_train = None
#pre_train = best_model
fix_pre_params = False

# decoder hype-parameters
search_mode = 1
with_batch = 1
ori_search = 0
beam_size = 10
vocab_norm = 1
len_norm = 1
with_mv = 0
merge_way = 'Y'
avg_att = 0
m_threshold = 100.
ngram = 3
length_norm = 0.
cover_penalty = 0.

# optimizer

'''
Starting learning rate. If adagrad/adadelta/adam is used, then this is the global learning rate.
Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001
'''
opt_mode = 'adadelta'
learning_rate = 1.0

#opt_mode = 'adam'
#learning_rate = 1e-3

#opt_mode = 'sgd'
#learning_rate = 1.

max_grad_norm = 1.0

# Start decaying every epoch after and including this epoch
start_decay_from = None
learning_rate_decay = 0.5
last_valid_bleu = 0.

snip_size = 10
file_tran_dir = 'wexp-gpu-nist03'
laynorm = False
segments = False
seg_val_tst_dir = 'orule_1.7'

with_bpe = True
with_postproc = False
copy_trg_emb = False

print_att = True
self_norm_alpha = None

#dec_gpu_id = [1]
#dec_gpu_id = None
gpu_id = [3]
#gpu_id = None


#args of discriminator DANN
dis_hid_size = 100
dis_type = 2
alpha = 0.05

adv_weight = 0.22

fine_tune = False

none_sense = True

#choose the loss function of the adversarial training
cross_entropy = True
max_entropy = True






