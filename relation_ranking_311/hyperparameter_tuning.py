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
    loss_margin = uniform(0.5, 1.5)
    clip = 0.6

    command = "python train.py --gpu 6 " \
            "--dev_every 900 --log_every 300 --save_every 5000 --batch_size 64 " \
            "--save_path saved_checkpoints/model-tuning " \
            "--epochs {} --lr {} --d_hidden {} --n_layers {} --dropout_prob {} --clip_gradient {} " \
            "--rnn_type gru --d_rel_embed {} --loss_margin {} >> " \
            "hyperparameter_results.txt".format(epochs, learning_rate, d_hidden, n_layers, dropout,
                                                clip, d_rel_embed, loss_margin)

    print("Running: " + command)
    call(command, shell=True)
