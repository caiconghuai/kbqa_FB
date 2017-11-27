import torch
import torch.optim as optim
import torch.nn as nn
import time
import os, sys
import glob
import numpy as np

from args import get_args
from model import EntityType
#from evaluation import evaluation
from seqMultiLabelLoader import SeqMultiLabelLoader

# please set the configuration in the file : args.py
args = get_args()
# set the random seed for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but do not use it. You are using CPU for training")


# load data
train_loader = SeqMultiLabelLoader(args.train_file, args.gpu)
print('load train data, batch_num: %d\tbatch_size: %d'
      %(train_loader.batch_num, train_loader.batch_size))
valid_loader = SeqMultiLabelLoader(args.valid_file, args.gpu)
print('load valid data, batch_num: %d\tbatch_size: %d'
      %(valid_loader.batch_num, valid_loader.batch_size))

# load word vocab for questions
word_vocab = torch.load(args.vocab_file)
print('load word vocab, size: %s' % len(word_vocab))
type_vocab = torch.load(args.type_vocab)
print('load type vocab, size: %s' % len(type_vocab))

os.makedirs(args.save_path, exist_ok=True)

# define models
config = args
config.n_cells = config.n_layers

if config.birnn:
    config.n_cells *= 2
print(config)

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = EntityType(word_vocab, config, len(type_vocab))
    if args.word_vectors:
        if os.path.isfile(args.vector_cache):
            pretrained = torch.load(args.vector_cache)
            model.embed.word_lookup_table.weight.data.copy_(pretrained)
        else:
            pretrained = model.embed.load_pretrained_vectors(args.word_vectors, binary=False,
                                            normalize=args.word_normalize)
            torch.save(pretrained, args.vector_cache)
            print('load pretrained word vectors from %s, pretrained size: %s' %(args.word_vectors,
                                                                                pretrained.size()))
    if args.cuda:
        model.cuda()
        print("Shift model to GPU")

# show model parameters
for name, param in model.named_parameters():
    print(name, param.size())

criterion = nn.BCELoss() # binary cross entropy loss function
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# train the model
iterations = 0
start = time.time()
best_dev_acc = 0
best_dev_F = 0
num_iters_in_epoch = train_loader.batch_num
patience = args.patience * num_iters_in_epoch # for early stopping
iters_not_improved = 0 # this parameter is used for stopping early
early_stop = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss       Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:12.4f},{:12.4f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:12.4f},{}'.split(','))
best_snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
print(header)


for epoch in range(1, args.epochs+1):
    if early_stop:
        print("Early stopping. Epoch: {}, Best Dev. Acc: {}".format(epoch, best_dev_acc))
        break

    n_correct, n_total, n_pred_total = 0, 0, 0

    for batch_idx, batch in enumerate(train_loader.next_batch()):
        iterations += 1
        label = batch[1]
        model.train();
        optimizer.zero_grad()

        scores = model(batch)
        hard_pred = torch.ge(scores.data, 0.5).type_as(label.data)

        n_correct += torch.mul(hard_pred, label.data).sum()
        n_total += label.data.sum()
        n_pred_total += hard_pred.sum()
        train_r = 100. * n_correct / n_total
        train_p = 100. * n_correct / n_pred_total
        train_f = 2*train_p*train_r / (train_p+train_r)

        loss = criterion(scores, label)  # BCELoss的label是FloatTensor类型。。。
        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_gradient)
        optimizer.step()

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + \
                        '_iter_{}_acc_{:.4f}_loss_{:.6f}_model.pt'.format(iterations, train_f, loss.data[0])
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:
            model.eval()
            n_dev_correct = 0
            n_dev_total = 0
            n_dev_pred_total = 0

            gold_list = []
            pred_list = []

            for valid_batch_idx, valid_batch in enumerate(valid_loader.next_batch()):
                valid_label = valid_batch[1].data
                answer = model(valid_batch)
                dev_pred = torch.ge(answer.data, 0.5).type_as(valid_label)
                n_dev_correct += torch.mul(dev_pred, valid_label).sum()
                n_dev_total += valid_label.sum()
                n_dev_pred_total += dev_pred.sum()

            dev_r = 100. * n_dev_correct / n_dev_total
            dev_p = 100. * n_dev_correct / n_dev_pred_total
            dev_f = 2*dev_r*dev_p / (dev_r+dev_p)
            print(dev_log_template.format(time.time() - start, epoch, iterations, 
                                          1 + batch_idx, train_loader.batch_num,
                                          100. * (1 + batch_idx) / train_loader.batch_num, 
                                          loss.data[0], train_f, dev_r, dev_p, dev_f))
            # update model
            if dev_f > best_dev_F:
                best_dev_F = dev_f
                iters_not_improved = 0
                snapshot_path = best_snapshot_prefix + \
                                '_iter_{}_devf1_{}_model.pt'.format(iterations, best_dev_F)

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                for f in glob.glob(best_snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            else:
                iters_not_improved += 1
                if iters_not_improved > patience:
                    early_stop = True
                    break

        # print progress message
        elif iterations % args.log_every == 0:
            print(log_template.format(time.time()-start, epoch, iterations, 1+batch_idx, 
                                      train_loader.batch_num, 100. * (1+batch_idx)/train_loader.batch_num, 
                                      loss.data[0], train_f, ' '*12))

