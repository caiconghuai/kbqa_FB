#!/usr/bin/python

import sys
import pickle
sys.path.append('../tools')
import virtuoso

def get_all_ngrams(tokens):
    all_ngrams = set()
    max_n = min(len(tokens), 3)
    for n in range(1, max_n+1):
        ngrams = find_ngrams(tokens, n)
        all_ngrams = all_ngrams | ngrams
    return all_ngrams

def find_ngrams(input_list, n):
    ngrams = zip(*[input_list[i:] for i in range(n)])
    return set(ngrams)


def get_name_ngrams(entity_name):
#    entity_name = processed_text(entity_name) # lowercase the name
    name_tokens = entity_name.split(' ')
    name_ngrams = get_all_ngrams(name_tokens)

    return name_ngrams


def create_inverted_index_entity(namespath, outpath):
    print("creating the index map...")
    index = {}
    size = 0
    entitys = pickle.load(open(namespath, 'rb'))
    cnt = 0
    for entity_mid in entitys:
        cnt += 1
        if cnt % 1000 == 0:
            print("line: {}".format(cnt))

        names = virtuoso.id_query_name(entity_mid)
        names.extend(virtuoso.id_query_alias(entity_mid))
        for entity_name in names:
            name_ngrams = get_name_ngrams(entity_name)
#            print(entity_name, name_ngrams)

            for ngram_tuple in name_ngrams:
                size += 1
                ngram = " ".join(ngram_tuple)
                if ngram in index.keys():
                    index[ngram].add((entity_mid, entity_name))
                else:
                    index[ngram] = set([(entity_mid, entity_name)])


    print("num keys: {}".format(len(index)))
    print("total key-value pairs: {}".format(size))

    print("dumping to pickle...")
    with open(outpath, 'wb') as f:
        pickle.dump(index, f)

    print("DONE")

if __name__ == '__main__':
    create_inverted_index_entity('FB2M.ent.pkl', 'ngram_ent_index.pkl')
    print("Created the entity index.")
