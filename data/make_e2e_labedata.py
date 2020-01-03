import os
import sys

from utils import get_e2e_fields, e2e_key2idx

e2e_train_src = "src_train.txt"
e2e_train_tgt = "train_tgt_lines.txt" # gold generations corresponding to src_train.txt
e2e_val_src = "src_valid.txt"
e2e_val_tgt = "valid_tgt_lines.txt" # gold generations corresponding to src_valid.txt

punctuation = set(['.', '!', ',', ';', ':', '?'])

def get_first_sent_tokes(tokes):
    try:
        first_per = tokes.index('.')
        return tokes[:first_per+1]
    except ValueError:
        return tokes

def  (tokes, fields):
    """找到target和src中完全一样的单词。
    Return:
        labels: list of tuple. 每一个tuple的第三项是属性，第一项和第二项是属性值的索引。
    比如src:__start_name__ The Vaults __end_name__ __start_eatType__ pub __end_eatType__ __start_priceRange__ more than £ 30 __end_priceRange__ __start_customerrating__ 5 out of 5 __end_customerrating__ __start_near__ Café Adriatic __end_near__
    有字段'customerrating'、'eatType'、 'name'、'near'、'priceRange' 
    target中的文本The Vaults pub near Café Adriatic has a 5 star rating . Prices start at £ 30 .
    
    只有'name'-The Vaults、'eatType'-pub和 'near'-Café Adriatic
    
    那么结果[(0, 2, idx('name')), (2, 3, idx('eatType')), (4, 6, idx('near')), (11, 12, idx('unknow')), (17, 18, idx('unknow'))]
    """
    labels = []
    i = 0
    while i < len(tokes):
        matched = False
        for j in xrange(len(tokes), i, -1):
            # first check if it's punctuation
            if all(toke in punctuation for toke in tokes[i:j]):
                labels.append((i, j, len(e2e_key2idx))) # first label after rul labels
                i = j
                matched = True
                break
            # then check if it matches stuff in the table
            for k, v in fields.iteritems():
                # take an uncased match
                if " ".join(tokes[i:j]).lower() == " ".join(v).lower():
                    labels.append((i, j, e2e_key2idx[k]))
                    i = j
                    matched = True
                    break
            if matched:
                break
        if not matched:
            i += 1
    return labels

def print_data(srcfi, tgtfi):
    with open(srcfi) as f1:
        with open(tgtfi) as f2:
            for srcline in f1:
                tgttokes = f2.readline().strip().split()
                senttokes = tgttokes

                fields = get_e2e_fields(srcline.strip().split()) # fieldname -> tokens
                labels = stupid_search(senttokes, fields)
                labels = [(str(tup[0]), str(tup[1]), str(tup[2])) for tup in labels]

                # add eos stuff
                senttokes.append("<eos>")
                labels.append((str(len(senttokes)-1), str(len(senttokes)), '8')) # label doesn't matter

                labelstr = " ".join([','.join(label) for label in labels])
                sentstr = " ".join(senttokes)

                outline = "%s|||%s" % (sentstr, labelstr)
                print outline


if sys.argv[1] == "train":
    print_data(e2e_train_src, e2e_train_tgt)
elif sys.argv[1] == "valid":
    print_data(e2e_val_src, e2e_val_tgt)
else:
    assert False
