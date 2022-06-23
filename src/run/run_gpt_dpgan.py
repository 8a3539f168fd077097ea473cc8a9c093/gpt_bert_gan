import sys
from subprocess import call
import os

from torch.autograd.grad_mode import F

# Job id and gpu_id
if len(sys.argv) > 2:
    job_id = int(sys.argv[1])
    gpu_id = str(sys.argv[2])
    print('job_id: {}, gpu_id: {}'.format(job_id, gpu_id))
elif len(sys.argv) > 1:
    job_id = int(sys.argv[1])
    gpu_id = 0
    print('job_id: {}, missing gpu_id (use default {})'.format(job_id, gpu_id))
else:
    job_id = 1
    gpu_id = 2
    print('Missing argument: job_id and gpu_id. Use default job_id: {}, gpu_id: {}'.format(job_id, gpu_id))

# Handle extra arguments
show_rewards = int(False)
show_probs = int(False)
if len(sys.argv) > 3:
    if str(sys.argv[3]) == "--show_rewards=True":
        show_rewards = int(True)
    elif str(sys.argv[3]) == "--show_probs=True":
        show_probs = int(True)
if len(sys.argv) > 4:
    if str(sys.argv[4]) == "--show_rewards=True":
        show_rewards = int(True)
    elif str(sys.argv[4]) == "--show_probs=True":
        show_probs = int(True)



# Executables
#executable = '/usr/bin/python3'  # specify your own python interpreter path here
executable = sys.executable
rootdir = '../'
scriptname = 'main.py'

# ===Program===
if_test = int(False)
run_model = 'gpt_dpgan'
sa = int(True)
CUDA = int(True)
oracle_pretrain = int(False)
gen_pretrain = int(False)
dis_pretrain = int(False)
MLE_train_epoch = 4
ADV_train_epoch = 100
tips = 'GPT_DPGAN experiments'

# ===Oracle  or Real===
if_real_data = [int(False), int(True), int(True), int(True), int(True)]
dataset = ['oracle', 'image_coco', 'emnlp_news', "empnlp_news_coco", "emnlp_news_reduced"]
vocab_size = [5000, 0, 0, 0, 0]

# ===Basic Param===
data_shuffle = int(False)
model_type = 'pineapple'
gen_init = 'normal'
dis_init = 'uniform'
samples_num = 1000
batch_size = 32
max_seq_len = 37
gen_lr = 0.00005
dis_lr = 0.01
pre_log_step = 2
adv_log_step = 1

# ===Generator===
ADV_g_step = 1
rollout_num = 16
gen_embed_dim = 32
gen_hidden_dim = 32
gen_num_heads = 4
gen_nlayers = 2

# ===Discriminator===
d_step = 5
d_epoch = 3
ADV_d_step = 4
ADV_d_epoch = 2
dis_embed_dim = 64
dis_hidden_dim = 64
dis_num_heads = 4
dis_nlayers = 2

# ===Metrics===
use_nll_oracle = int(True)
use_nll_gen = int(True)
use_nll_div = int(True)
use_bleu = int(True)
use_self_bleu = int(True)
use_ppl = int(False)

# logs
#show_rewards = show_rewards
#show_probs = show_probs

args = [
    # Program
    '--if_test', if_test,
    '--run_model', run_model,
    '--sa', sa,
    '--cuda', CUDA,
    # '--device', gpu_id,  # comment for auto GPU
    '--ora_pretrain', oracle_pretrain,
    '--gen_pretrain', gen_pretrain,
    '--dis_pretrain', dis_pretrain,
    '--mle_epoch', MLE_train_epoch,
    '--adv_epoch', ADV_train_epoch,
    '--tips', tips,

    # Oracle or Real
    '--if_real_data', if_real_data[job_id],
    '--dataset', dataset[job_id],
    '--vocab_size', vocab_size[job_id],

    # Basic Param
    '--shuffle', data_shuffle,
    '--model_type', model_type,
    '--gen_init', gen_init,
    '--dis_init', dis_init,
    '--samples_num', samples_num,
    '--batch_size', batch_size,
    '--max_seq_len', max_seq_len,
    '--gen_lr', gen_lr,
    '--dis_lr', dis_lr,
    '--pre_log_step', pre_log_step,
    '--adv_log_step', adv_log_step,

    # Generator
    '--adv_g_step', ADV_g_step,
    '--rollout_num', rollout_num,
    '--gen_embed_dim', gen_embed_dim,
    '--gen_hidden_dim', gen_hidden_dim,
    '--gen_num_heads', gen_num_heads,
    '--gen_nlayers', gen_nlayers,

    # Discriminator
    '--d_step', d_step,
    '--d_epoch', d_epoch,
    '--adv_d_step', ADV_d_step,
    '--adv_d_epoch', ADV_d_epoch,
    '--dis_embed_dim', dis_embed_dim,
    '--dis_hidden_dim', dis_hidden_dim,
    '--dis_num_heads', dis_num_heads,
    '--dis_nlayers', dis_nlayers,

    # Metrics
    '--use_nll_oracle', use_nll_oracle,
    '--use_nll_gen', use_nll_gen,
    '--use_nll_div', use_nll_div,
    '--use_bleu', use_bleu,
    '--use_self_bleu', use_self_bleu,
    '--use_ppl', use_ppl,

    # Logs
    '--show_rewards', show_rewards,
    '--show_probs', show_probs,
]

args = list(map(str, args))
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
