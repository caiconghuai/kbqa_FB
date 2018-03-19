from random import randint, uniform, choice
from subprocess import call

epochs = 10
count = 10
for id in range(count):
    learning_rate = 10 ** uniform(-4, -2)
    d_hidden = randint(100, 200)
#    n_layers = randint(2, 3)
    n_layers = 2
    dropout = uniform(0.1, 0.5)
    d_rel_embed = randint(100, 200)
    loss_margin = uniform(0.5, 1.5)
#    loss_margin = uniform(0.05, 0.2)
    clip = 0.6

    channel_size = randint(4, 10)
    conv_kernel_1 = choice([1,3,5])
    conv_kernel_2 = choice([1,3,5])
    pool_kernel_1 = 21
    pool_kernel_2 = 1

    command = "python train.py --gpu 6 " \
            "--train_file data/train.relation_ranking.all-50.pt " \
            "--valid_file data/valid.relation_ranking.all-50.pt " \
            "--dev_every 900 --log_every 300 --save_every 5000 --batch_size 64 " \
            "--save_path saved_checkpoints/model-tuning " \
            "--epochs {} --lr {} --dropout_prob {} --clip_gradient {} --loss_margin {} " \
            "--d_hidden {} --n_layers {} --rnn_type gru --d_rel_embed {} " \
            "--channel_size {} --conv_kernel_1 {} --conv_kernel_2 {} --pool_kernel_1 {} " \
            "--pool_kernel_2 {} >> "\
            "hyperparameter_results_merge.txt".format(epochs, learning_rate, dropout, clip,
                                                    loss_margin, d_hidden, n_layers, d_rel_embed,
                                                    channel_size, conv_kernel_1, conv_kernel_2,
                                                    pool_kernel_1, pool_kernel_2)

    print("Running: " + command)
    call(command, shell=True)
