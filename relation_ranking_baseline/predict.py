import os
import sys
import numpy as np
import torch
import pickle

from args import get_args
from model import RelationRanking
from seqRankingLoader import *
sys.path.append('../tools')
import virtuoso

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


if not args.trained_model:
    print("ERROR: You need to provide a option 'trained_model' path to load the model.")
    sys.exit(1)

# load word vocab for questions, relation vocab for relations
word_vocab = torch.load(args.vocab_file)
print('load word vocab, size: %s' % len(word_vocab))
rel_vocab = torch.load(args.rel_vocab_file)
print('load relation vocab, size: %s' %len(rel_vocab))

os.makedirs(args.results_path, exist_ok=True)

# load the model
model = torch.load(args.trained_model, map_location=lambda storage,location: storage.cuda(args.gpu))

def evaluate(dataset = args.test_file, tp = 'test'):

    # load batch data for predict
    data_loader = SeqRankingLoader(dataset, len(rel_vocab), args.gpu)
    print('load %s data, batch_num: %d\tbatch_size: %d'
            %(tp, data_loader.batch_num, data_loader.batch_size))

    model.eval();
    n_correct = 0

    for data_batch_idx, data_batch in enumerate(data_loader.next_batch(shuffle=False)):
        pos_score, neg_score = model(data_batch)
        n_correct += (torch.sum(torch.gt(pos_score, neg_score), 0).data ==
                      neg_score.size(0)).sum()

    total = data_loader.batch_num*data_loader.batch_size
    accuracy = 100. * n_correct / (total)
    print("%s\taccuracy: %8.6f\tcorrect: %d\ttotal: %d" %(tp, accuracy, n_correct, total))
    print("-" * 80)

def rel_pruned(neg_rel, neg_score, data):
    unk_token = rel_vocab.lookup(rel_vocab.unk_token)
    neg_pruned = [i for i in list(zip(neg_rel, neg_score)) # 去掉score<0的rel
                      if i[1] > 0 and i[0] != unk_token]
    if len(neg_pruned) == 0:
        pred_rel = []
        pred_rel_scores = []
    else:
        # 对rel进行剪枝，取score在gap前面的
        neg_pruned.sort(key=lambda x: -x[1])
        pred_rel_index = [neg_pruned[0][0]]
        pred_rel_scores = [neg_pruned[0][1]]
        for i in range(1, len(neg_pruned)-1):
            if neg_pruned[i-1][1] - neg_pruned[i][1] < neg_pruned[i][1] - neg_pruned[i+1][1]:
                pred_rel_index.append(neg_pruned[i][0])
                pred_rel_scores.append(neg_pruned[i][1])
            else:
                break
        if len(neg_pruned) == 2 and neg_pruned[-2][1] - neg_pruned[-1][1] < neg_pruned[-1][1]:
            pred_rel_index.append(neg_pruned[-1][0])
            pred_rel_scores.append(neg_pruned[-1][1])
        pred_rel = rel_vocab.convert_to_word(pred_rel_index)

    # 由剪枝后的rel，找与之相连的sub和obj
    pred_obj = []
    pred_sub = [] # 与pred_rel相连所有sub
    '''
    for rel in pred_rel:
        for sub in data.cand_sub:
            obj = virtuoso.query_object(sub, rel)
            pred_obj.append(obj[:10])
            if obj:
                pred_sub.append(sub)
    print(data.question)
    print(data.subject, data.cand_sub[:5], data.subject in data.cand_sub, len(data.cand_sub))
    print(data.relation, pred_rel, pred_rel_scores)
    print(data.object, pred_obj)
    '''
    return pred_rel, pred_rel_scores, pred_obj, pred_sub


def predict(qa_lable_file, tp='test', write_res=args.write_result, write_score=args.write_score):
    # load batch data for predict
    data_loader = CandidateRankingLoader(qa_lable_file, word_vocab, rel_vocab, args.gpu)
    print('load %s data, batch_num: %d\tbatch_size: %d' %(tp, data_loader.batch_num, 1))
    if write_res:
#        results_file = open(os.path.join(args.results_path, '%s-results.txt' %tp), 'w')
        results_all_file = open(os.path.join(args.results_path, '%s-results-all.txt' %tp), 'w')
    if write_score:
        score_file = open(os.path.join(args.results_path, 'score-rel-%s.txt' %tp), 'w')

    model.eval()
    total = 0
    n_rel_correct = 0
    n_recall = 0
    n_correct = 0
    n_sub_rel_correct = 0
    n_single_sub = 0
    n_single_sub_correct = 0
    for data_batch in data_loader.next_question():
        data = data_batch[3]
        if data.subject not in data.cand_sub: # cand_sub错的就不用管了
            continue
        total += 1

        neg_rel = data_batch[2].squeeze().data.cpu().numpy()
        pos_score, neg_score = model(data_batch[:3])
        neg_score = neg_score.data.squeeze().cpu().numpy()

        if write_score:
            score_file.write('%s\n' %(' '.join(neg_score.astype('str'))))

        pred_rel, pred_rel_scores, pred_obj, pred_sub = rel_pruned(neg_rel, neg_score, data)
        if not pred_rel:continue
        n_rel_correct += (pred_rel[0] == data.relation)

        if len(data.cand_sub) == 1:
            n_single_sub += 1
            if pred_rel[0] == data.relation:
                n_single_sub_correct += 1

        if write_res:
            results_all_file.write('%s\n' %(data.question))
            results_all_file.write('%s\t%s\t%d\n' %(data.subject, data.pred_sub, len(data.cand_sub)))
            results_all_file.write('%s\t%s\t%s\n' %(data.relation, pred_rel, pred_rel_scores))
            results_all_file.write('%s\t%s\n' %(data.object, pred_obj))

        if len(pred_sub) == 1 and pred_sub[0] == data.subject:
            if len(pred_rel) == 1 and pred_rel[0] == data.relation:
                n_sub_rel_correct += 1  # sub和rel都预测正确且唯一的个数
        pred_obj_ = sum(pred_obj, [])  # 神奇的用法，把多维list变成一维
        if data.object in pred_obj_:
            n_recall += 1   # obj包含正确答案的个数
            if len(pred_obj_) == 1:
                n_correct += 1 # obj完全正确的个数

    accuracy = 100. * n_correct / total
    rel_acc = 100. * n_rel_correct / total
    sub_rel_acc = 100. * n_sub_rel_correct / total
    recall = 100. * n_recall / total
    print("%s\taccuracy: %8.6f\tcorrect: %d\ttotal: %d" %(tp, accuracy, n_correct, total))
    print('recall: ', recall, n_recall)
    print('rel_acc: ', rel_acc, n_rel_correct)
    print('sub_rel_acc: ', sub_rel_acc, n_sub_rel_correct)
    single_sub_acc = 100. * n_single_sub_correct / n_single_sub
    multi_sub_acc = 100. * (n_rel_correct-n_single_sub_correct) / (total-n_single_sub)
    print('single sub rel_acc: ', single_sub_acc, n_single_sub_correct, n_single_sub)
    print('multi sub rel_acc: ', multi_sub_acc, 
          (n_rel_correct-n_single_sub_correct), (total-n_single_sub))
    print("-" * 80)

if args.predict:
    predict('../entity_detection/results-2/QAData.label.test.pkl', 'test')
    predict('../entity_detection/results-2/QAData.label.valid.pkl', 'valid')
    predict('../entity_detection/results-2/QAData.label.train.pkl', 'train')
else:
    evaluate(args.test_file, "test")
    evaluate(args.valid_file, "valid")
    evaluate(args.train_file, 'train')
