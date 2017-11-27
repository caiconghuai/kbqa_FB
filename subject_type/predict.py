import os
import sys
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pickle

from args import get_args
from model import EntityType
#from evaluation import evaluation
from seqMultiLabelLoader import SeqMultiLabelLoader
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

# load word vocab for questions
word_vocab = torch.load(args.vocab_file)
print('load word vocab, size: %s' % len(word_vocab))
type_vocab = torch.load(args.type_vocab)
print('load type vocab, size: %s' % len(type_vocab))

os.makedirs(args.results_path, exist_ok=True)

# load the model
model = torch.load(args.trained_model, map_location=lambda storage,location: storage.cuda(args.gpu))

def cosine_similarity(l1, l2, norm=True):
    if norm:
        norm = np.linalg.norm(l1) * np.linalg.norm(l2)
    else:
        norm = 1
    cos_sim = np.sum(l1*l2)/max(norm, 1e-08)
    return cos_sim

def predict(dataset=args.test_file, tp='test',
            write_res=args.write_result, write_score=args.write_score):
    # load QAdata
    qa_data_path = '../merge_sub_rel/data/QAData.cand.%s.pkl'%tp
    qa_data = pickle.load(open(qa_data_path,'rb'))

    # load batch data for predict
    data_loader = SeqMultiLabelLoader(dataset, args.gpu)
    print('load %s data, batch_num: %d\tbatch_size: %d'
            %(tp, data_loader.batch_num, data_loader.batch_size))

    model.eval();

    n_correct = 0
    n_total = 0
    n_pred_total = 0
    correct_pred_sub = 0
    n_single_sub = 0
    n_single_sub_correct = 0
    linenum = 1
    qa_data_idx = 0
    if write_res:
        results_file = open(os.path.join(args.results_path, '%s-results.txt' %tp), 'w')
    if write_score:
        score_file = open(os.path.join(args.results_path, 'score-sub-%s.txt' %tp), 'w')

    for data_batch_idx, data_batch in enumerate(data_loader.next_batch(shuffle=False)):
        if data_batch_idx % 10 == 0:
            print(tp, data_batch_idx)
        label = data_batch[1].data
        model_scores = model(data_batch).data
        pred_type_index = torch.ge(model_scores, 0.5).type_as(label)
        # 计算预测type的precise和recall
        n_correct += torch.mul(pred_type_index, label).sum()
        n_total += label.sum()
        n_pred_total += pred_type_index.sum()

        for i in range(data_loader.batch_size):
            if qa_data_idx >= len(qa_data): # 最后一个batch后面都是<pad>填充的，此时qa_data已经找到头了
                break
            _qa_data = qa_data[qa_data_idx]
            if not hasattr(_qa_data, 'cand_sub'):
                qa_data_idx += 1
                continue
            #cand_sub本来就没有正确答案的就不用预测了
            if _qa_data.subject not in _qa_data.cand_sub:
                qa_data_idx += 1
                continue

            cand_sub_score = {}
            base_sub_score = {}
            pred_score_list = []
            for index, sub in enumerate(_qa_data.cand_sub):
                type = _qa_data.sub_types[index]
                type_index = type_vocab.convert_to_index(type)
                cand_sub_type = np.zeros(label[i].size(0))
                cand_sub_type[type_index] = 1
                score = cosine_similarity(cand_sub_type, pred_type_index[i].cpu().numpy())
                base_score = cosine_similarity(cand_sub_type, label[i].cpu().numpy())
                cand_sub_score[sub] = score
                base_sub_score[sub] = base_score
                pred_score_list.append(str(score))

            cand_sub_score_sorted = sorted(cand_sub_score.items(), key=lambda item:item[1], reverse=True)
            base_sub_score_sorted = sorted(base_sub_score.items(), key=lambda item:item[1], reverse=True)
            # 按照type的打分对cand_sub排序，计算得分第一的准确率
            if cand_sub_score_sorted[0][1] == cand_sub_score[_qa_data.subject]:
                correct_pred_sub += 1
                if len(_qa_data.cand_sub) == 1:
                    n_single_sub_correct += 1
            if len(_qa_data.cand_sub) == 1:
                n_single_sub += 1

#            print(_qa_data.question, _qa_data.subject)
#            print(base_sub_score_sorted)
#            print(cand_sub_score_sorted)

            if write_res:
                results_file.write('%s\t%s\n' %(_qa_data.question, _qa_data.subject))
                results_file.write('%s\n' % base_sub_score_sorted)
                results_file.write('%s\n' % cand_sub_score_sorted)
            if write_score:
                score_file.write('%s\n' % ' '.join(pred_score_list))
            linenum += 1
            qa_data_idx += 1

    accuracy = 100. * correct_pred_sub / (linenum -1)
    print("%s\taccuracy: %8.6f\tcorrect: %d\ttotal: %d" %(tp, accuracy, correct_pred_sub, (linenum-1)))
    single_sub_acc = 100. * n_single_sub_correct / n_single_sub
    print('single sub accuracy: %8.6f\tcorrect: %d\tsingle sub num: %d' 
          %(single_sub_acc, n_single_sub_correct, n_single_sub))
    multi_sub_acc = 100. * (correct_pred_sub - n_single_sub_correct) / (linenum-1-n_single_sub)
    print('multi sub accuracy: %8.6f\tcorrect: %d\tmulti sub num: %d' 
          %(multi_sub_acc, (correct_pred_sub - n_single_sub_correct), (linenum-1-n_single_sub)))

    R = 100. * n_correct / n_total
    P = 100. * n_correct / n_pred_total
    F1 = 2*P*R / (P+R)
    print('P: %s\tR: %s\tF1: %s' %(P, R, F1))
    print("-" * 80)

    if write_res:
        results_file.close()
    if write_score:
        score_file.close()

# run the model on the dev set and write the output to a file
predict(args.valid_file, "valid")

# run the model on the test set and write the output to a file
predict(args.test_file, "test")

# run the model on the train set and write the output to a file
predict(args.train_file, 'train')
