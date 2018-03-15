from random import randint, uniform
from subprocess import call

epochs = 10
count = 10
for id in range(count):
    learning_rate = 10 ** uniform(-4, -2)
    d_hidden = randint(100, 200)
    n_layers = randint(2, 3)
    dropout = uniform(0.3, 0.5)
    d_rel_embed = randint(100, 200)
#    loss_margin = uniform(0.5, 1.5)
    loss_margin = uniform(0.05, 0.2)
    clip = 0.6

    channel_size = randint(6, 12)
    conv_kernel_1 = randint(1,4)
    conv_kernel_2 = randint(1,4)
    pool_kernel_1 = randint(3,6)
    pool_kernel_2 = randint(3,6)

    '''
    command = "python train.py --gpu 6 " \
            "--dev_every 900 --log_every 300 --save_every 5000 --batch_size 64 " \
            "--save_path saved_checkpoints/model-tuning " \
            "--epochs {} --lr {} --d_hidden {} --n_layers {} --dropout_prob {} --clip_gradient {} " \
            "--rnn_type gru --d_rel_embed {} --loss_margin {} >> " \
            "hyperparameter_results.txt".format(epochs, learning_rate, d_hidden, n_layers, dropout,
                                                clip, d_rel_embed, loss_margin)
    '''
    command = "python train_cnn.py --gpu 6 " \
            "--dev_every 900 --log_every 300 --save_every 5000 --batch_size 64 " \
            "--save_path saved_checkpoints/model-tuning " \
            "--epochs {} --lr {} --dropout_prob {} --clip_gradient {} --loss_margin {} " \
            "--channel_size {} --conv_kernel_1 {} --conv_kernel_2 {} --pool_kernel_1 {} " \
            "--pool_kernel_2 {} >> "\
            "hyperparameter_results_cnn.txt".format(epochs, learning_rate, dropout, clip,
                                                    loss_margin, channel_size, conv_kernel_1,
                                                    conv_kernel_2, pool_kernel_1, pool_kernel_2)

    print("Running: " + command)
    call(command, shell=True)
