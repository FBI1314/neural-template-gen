# -*- coding: utf-8 -*-
# @Author: shangerxin
# @Date:   2019-12-27
# @Last Modified by:   shangerxin
# @Last Modified time: 2019-12-30

"""
this file modified from the word_language_model example
"""
import os
import torch

from collections import Counter, defaultdict

from data.utils import get_wikibio_poswrds, get_e2e_poswrds

import random
random.seed(1111)

#punctuation = set(['.', '!', ',', ';', ':', '?', '--', '-rrb-', '-lrb-'])
punctuation = set() # i don't know why i was so worried about punctuation

class Dictionary(object):
    def __init__(self, unk_word="<unk>"):
        self.unk_word = unk_word
        self.idx2word = [unk_word, "<pad>", "<bos>", "<eos>"] # OpenNMT constants
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def add_word(self, word, train=False):
        """返回输入词在词表中的索引
        returns idx of word
        """
        if train and word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word] if word in self.word2idx else self.word2idx[self.unk_word]

    def bulk_add(self, words):
        """更新idx2word和word2idx
        assumes train=True
        """
        self.idx2word.extend(words)
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def __len__(self):
        return len(self.idx2word)


class SentenceCorpus(object):
    def __init__(self, path, bsz, thresh=0, add_bos=False, add_eos=False,
                 test=False):
        self.dictionary = Dictionary()
        self.bsz = bsz
        self.wiki = "wiki" in path

        train_src = os.path.join(path, "src_train.txt")

        if thresh > 0:
            self.get_vocabs(os.path.join(path, 'train.txt'), train_src, thresh=thresh)
            self.ngen_types = len(self.genset) + 4 # assuming didn't encounter any special tokens
            add_to_dict = False
        else:
            add_to_dict = True
        trsents, trlabels, trfeats, trlocs, inps = self.tokenize(
            os.path.join(path, 'train.txt'), train_src, add_to_dict=add_to_dict,
            add_bos=add_bos, add_eos=add_eos)
        print "using vocabulary of size:", len(self.dictionary)

        # print self.ngen_types, "gen word types"
        self.train, self.train_mb2linenos = self.minibatchify(
            trsents, trlabels, trfeats, trlocs, inps, bsz) # list of minibatches

        if (os.path.isfile(os.path.join(path, 'valid.txt'))
                or os.path.isfile(os.path.join(path, 'test.txt'))):
            if not test:
                val_src = os.path.join(path, "src_valid.txt")
                vsents, vlabels, vfeats, vlocs, vinps = self.tokenize(
                    os.path.join(path, 'valid.txt'), val_src, add_to_dict=False,
                    add_bos=add_bos, add_eos=add_eos)
            else:
                print "using test data and whatnot...."
                test_src = os.path.join(path, "src_test.txt")
                vsents, vlabels, vfeats, vlocs, vinps = self.tokenize(
                    os.path.join(path, 'test.txt'), test_src, add_to_dict=False,
                    add_bos=add_bos, add_eos=add_eos)
            self.valid, self.val_mb2linenos = self.minibatchify(
                vsents, vlabels, vfeats, vlocs, vinps, bsz)


    def get_vocabs(self, path, src_path, thresh=2):
        """unks words occurring <= thresh times"""
        tgt_voc = Counter() # 包含了词、类别、索引，所有的词
        assert os.path.exists(path)

        linewords = []
        with open(src_path, 'r') as f:
            for line in f:
                tokes = line.strip().split()
                if self.wiki:
                    fields = get_wikibio_poswrds(tokes) #key, pos -> wrd
                else:
                    fields = get_e2e_poswrds(tokes) # key, pos -> wrd
                    # with open('/Users/shang/Work/NLG/neural-template-gen/data/e2e_aligned/fields.txt', 'w') as fw:
                    #     for line in fields:
                    #         fw.writelines(str(line)+'\n')
                fieldvals = fields.values()
                tgt_voc.update(fieldvals)
                # 每一行是一句话中的词（无标点符号、去重）
                linewords.append(set(wrd for wrd in fieldvals
                                     if wrd not in punctuation))
                tgt_voc.update([k for k, idx in fields])
                tgt_voc.update([idx for k, idx in fields])

        genwords = Counter() # 每一句话，train中有但是src_train中没有的词
        # Add words to the dictionary
        with open(path, 'r') as f:
            #tokens = 0
            for l, line in enumerate(f):
                words, spanlabels = line.strip().split('|||')
                words = words.split()
                genwords.update([wrd for wrd in words if wrd not in linewords[l]])
                tgt_voc.update(words)

        # prune
        # N.B. it's possible a word appears enough times in total but not in genwords
        # so we need separate unking for generation
        #print "comeon", "aerobatic" in genwords
        # 仅为了打印日志拆成两段
        # for cntr in [tgt_voc, genwords]:
        #     for k in cntr.keys():
        #         if cntr[k] <= thresh:
        #             del cntr[k]

        for cntr in [tgt_voc]:
            for k in cntr.keys():
                if cntr[k] <= thresh:
                    del cntr[k]

        for cntr in [genwords]:
            for k in cntr.keys():
                if cntr[k] <= thresh:
                    # 这个删除操作会造成生成文本中有新词
                    # print 'del {} for {} < {}'.format(k, cntr[k], thresh)
                    del cntr[k]

        self.genset = set(genwords.keys())
        tgtkeys = tgt_voc.keys()
        # make sure gen stuff is first genset中的词在词表的前面
        tgtkeys.sort(key=lambda x: -(x in self.genset))
        self.dictionary.bulk_add(tgtkeys)
        # genset中不能有特殊符号
        # make sure we did everything right (assuming didn't encounter any special tokens)
        assert self.dictionary.idx2word[4 + len(self.genset) - 1] in self.genset
        assert self.dictionary.idx2word[4 + len(self.genset)] not in self.genset
        # 全部词表顺序：特殊符号4个、genset、tgtset、特殊符号5个？？
        self.dictionary.add_word("<ncf1>", train=True)
        self.dictionary.add_word("<ncf2>", train=True)
        self.dictionary.add_word("<ncf3>", train=True)
        self.dictionary.add_word("<go>", train=True)
        self.dictionary.add_word("<stop>", train=True)

        # with open('/Users/shang/Work/NLG/neural-template-gen/data/e2e_aligned/word2idx.dict', 'w') as fw:
        #     for k, v in self.dictionary.word2idx.iteritems():
        #         line = '{}\t{}'.format(k, v)
        #         fw.writelines(line+'\n')
        #     print 'write dictionary done.'

        # with open('/Users/shang/Work/NLG/neural-template-gen/data/e2e_aligned/genset.txt', 'w') as fw:
        #     for w in self.genset:
        #         line = '{}'.format(w)
        #         fw.writelines(line+'\n')
        #     print 'write genset done.'


    def tokenize(self, path, src_path, add_to_dict=False, add_bos=False, add_eos=False):
        """Assumes fmt is sentence|||s1,e1,k1 s2,e2,k2 ....
        Args:
            path:对应的一句话|||这一句话的labels（一组三个）start, end, label
            src_path: 一句话的segment，每一个record（属性、属性值）
        Returns:
            sents:list of list, 内层的list是每一个生成文本中词对应的索引，如果有新词，则用padidx替代
            labels:list of list, 内层的list是每一个生成文本的标签，标签是一组三个
            src_feats:list of list, 内层list是每一个数据文本的特征, 即一个词在全局字典中的类别索引、位置索引，词索引 
            copylocs:list of list, 内层list是每一个生成文本中词对标数据文本的索引，不存在置为-1
            inps:list of list, 内层list是每一个生成文本中词含有go、end这类是否结束的特征
        """
        assert os.path.exists(path)

        src_feats, src_wrd2idxs, src_wrd2fields = [], [], []
        w2i = self.dictionary.word2idx

        # fw_wrd2idxs = open("/Users/shang/Work/NLG/neural-template-gen/data/e2e_aligned/src_wrd2idxs.txt", "w")
        # fw_wrd2fields = open("/Users/shang/Work/NLG/neural-template-gen/data/e2e_aligned/src_wrd2fields.txt", "w")
        # fw_feats = open("/Users/shang/Work/NLG/neural-template-gen/data/e2e_aligned/src_feats.txt", "w")
        with open(src_path, 'r') as f:
            for line in f:
                # 每一条样本计算计算一次
                tokes = line.strip().split()
                #fields = get_e2e_fields(tokes, keys=self.e2e_keys) #keyname -> list of words
                # fields是一个样本的，类别，索引，词的dict
                if self.wiki:
                    fields = get_wikibio_poswrds(tokes) #key, pos -> wrd
                else:
                    fields = get_e2e_poswrds(tokes) # key, pos -> wrd
                
                # wrd2things will be unordered
                # 对每一个词，记录在这个样本中的位置
                # feats 是每一个词在整个词典中的索引-唯一
                # wrd2idxs 是每一个词在这个样本中的索引-多个
                # wrd2fields 是每一个词在这个样本中的特征（在整个词典的索引+是否是类别最后的索引)
                feats, wrd2idxs, wrd2fields = [], defaultdict(list), defaultdict(list)
                # get total number of words per field每一个类别有几个词，比如姓名尚尔昕，那么姓名这个类的长度是3
                fld_cntr = Counter([key for key, _ in fields])
                
                
                for (k, idx), wrd in fields.iteritems():
                    if k in w2i:
                        featrow = [self.dictionary.add_word(k, add_to_dict),
                                   self.dictionary.add_word(idx, add_to_dict),
                                   self.dictionary.add_word(wrd, add_to_dict)]
                        wrd2idxs[wrd].append(len(feats))
                        #nflds = self.dictionary.add_word(fld_cntr[k], add_to_dict)
                        # 如果当前词是昕，是name的最后一个，那么cheatfeat就是stop，否则是go
                        cheatfeat = w2i["<stop>"] if fld_cntr[k] == idx else w2i["<go>"]
                        wrd2fields[wrd].append((featrow[2], featrow[0], featrow[1], cheatfeat))
                        feats.append(featrow)
                src_wrd2idxs.append(wrd2idxs)
                src_wrd2fields.append(wrd2fields)
                src_feats.append(feats)
        #         fw_wrd2idxs.writelines(str(dict(wrd2idxs))+'\n')
        #         fw_wrd2fields.writelines(str(dict(wrd2fields))+'\n')
        #         fw_feats.writelines(str(feats)+'\n')
        # fw_wrd2idxs.close()
        # fw_wrd2fields.close()
        # fw_feats.close()

        sents, labels, copylocs, inps = [], [], [], []

        # Add words to the dictionary
        tgtline = 0
        with open(path, 'r') as f:
            #tokens = 0
            for line in f:
                words, spanlabels = line.strip().split('|||')
                words = words.split()
                # 每一个样本有
                sent, copied, insent = [], [], []
                if add_bos:
                    sent.append(self.dictionary.add_word('<bos>', True))
                for word in words:
                    # sent is just used for targets; we have separate inputs
                    # sents记录了在生成文本中出现但是训练样本中没有的词，类似于“非实时类”描述
                    if word in self.genset:
                        sent.append(w2i[word])
                    else:
                        sent.append(w2i["<unk>"])

                    if word not in punctuation and word in src_wrd2idxs[tgtline]:
                        copied.append(src_wrd2idxs[tgtline][word])
                        # 为什么是一个list of list
                        winps = [[widx, kidx, idxidx, nidx]
                                 for widx, kidx, idxidx, nidx in src_wrd2fields[tgtline][word]]
                        insent.append(winps)
                    else:
                        #assert sent[-1] < self.ngen_types
                        copied.append([-1])
                         # 1 x wrd, tokennum, totalnum
                        #insent.append([[sent[-1], w2i["<ncf1>"], w2i["<ncf2>"]]])
                        insent.append([[sent[-1], w2i["<ncf1>"], w2i["<ncf2>"], w2i["<ncf3>"]]])
                

                #sent.extend([self.dictionary.add_word(word, add_to_dict) for word in words])
                if add_eos:
                    sent.append(self.dictionary.add_word('<eos>', True))
                labetups = [tupstr.split(',') for tupstr in spanlabels.split()]
                labelist = [(int(tup[0]), int(tup[1]), int(tup[2])) for tup in labetups]
                sents.append(sent)
                labels.append(labelist)
                copylocs.append(copied)
                inps.append(insent)
                tgtline += 1
        assert len(sents) == len(labels)
        assert len(src_feats) == len(sents)
        assert len(copylocs) == len(sents)
        return sents, labels, src_feats, copylocs, inps

    def featurize_tbl(self, fields):
        """每个样本的feats数量不一样，但是特征长度都是3
        fields are key, pos -> wrd maps
        returns: nrows x nfeats tensor
        """
        feats = []
        for (k, idx), wrd in fields.iteritems():
            if k in self.dictionary.word2idx:
                featrow = [self.dictionary.add_word(k, False),
                           self.dictionary.add_word(idx, False),
                           self.dictionary.add_word(wrd, False)]
                feats.append(featrow)
        return torch.LongTensor(feats)

    def padded_loc_mb(self, curr_locs):
        """
        curr_locs is a bsz-len list of tgt-len list of locations
        returns:
          a seqlen x bsz x max_locs tensor
        """
        max_locs = max(len(locs) for blocs in curr_locs for locs in blocs)
        for blocs in curr_locs:
            for locs in blocs:
                if len(locs) < max_locs:
                    locs.extend([-1]*(max_locs - len(locs)))
        return torch.LongTensor(curr_locs).transpose(0, 1).contiguous()

    def padded_feat_mb(self, curr_feats):
        """ curr_feats有bsz-len个样本，每个样本的features长度不同。现在将所有样本feat用<pad>补齐统一长度
        curr_feats is a bsz-len list of nrows-len list of features
        returns:
          a bsz x max_nrows x nfeats tensor
        """
        max_rows = max(len(feats) for feats in curr_feats)
        nfeats = len(curr_feats[0][0])
        for feats in curr_feats:
            if len(feats) < max_rows:
                [feats.append([self.dictionary.word2idx["<pad>"] for _ in xrange(nfeats)])
                 for _ in xrange(max_rows - len(feats))]
        return torch.LongTensor(curr_feats)


    def padded_inp_mb(self, curr_inps):
        """
        curr_inps is a bsz-len list of seqlen-len list of nlocs-len list of features
        returns:
          a bsz x seqlen x max_nlocs x nfeats tensor
        """
        max_rows = max(len(feats) for seq in curr_inps for feats in seq)
        nfeats = len(curr_inps[0][0][0])
        for seq in curr_inps:
            for feats in seq:
                if len(feats) < max_rows:
                    # pick random rows
                    randidxs = [random.randint(0, len(feats)-1)
                                for _ in xrange(max_rows - len(feats))]
                    [feats.append(feats[ridx]) for ridx in randidxs]
        return torch.LongTensor(curr_inps)


    def minibatchify(self, sents, labels, feats, locs, inps, bsz):
        """
        this should result in there never being any padding.
        each minibatch is:
          (seqlen x bsz, 
           bsz-length list of lists of (start, end, label) constraints,
           bsz x nfields x nfeats, 
           seqlen x bsz x max_locs, 
           seqlen x bsz x max_locs x nfeats)
        Returns:
            minibatches: list，每一项
                第一个是词，seqlen x bsz
                        ~~~ tensor([[407],[255]])
                第二项是label，bsz x (start, end, label)
                        ~~~ [[(1, 2, 8)]]
                第三项是生成的特征 bsz x nfields x nfeats,
                        ~~~ tensor([[[803, 780, 229],
                                     [850, 781, 808],
                                     [850, 780, 503],
                                     [850, 782, 800],
                                     [855, 780, 858],
                                     [814, 780, 379]]])  这里fileds个数是6，feat个数是3，表示cls_idx, idx_idx, wrd_idx
                第四项是生成句子每一个词对标数据的位置 seqlen x bsz x max_locs
                        ~~~ tensor([[[-1]], [[-1]]])
                第五项是生成句子每一个词含有go、end的特征 seqlen x bsz x max_locs x nfeats
                        ~~~ tensor([[[[407, 867, 868, 869]]], [[[255, 867, 868, 869]]]])
            mb2linenos: list，句子排序前的位置索引
        """
        # sort in ascending order
        # sents变成一个tuple，第一项是排序后的索引
        # sorted_idxs是sent排序前的索引
        sents, sorted_idxs = zip(*sorted(zip(sents, range(len(sents))), key=lambda x: len(x[0])))
        minibatches, mb2linenos = [], []
        curr_batch, curr_labels, curr_feats, curr_locs, curr_linenos = [], [], [], [], []
        curr_inps = []
        curr_len = len(sents[0])
        for i in xrange(len(sents)):
            if len(sents[i]) != curr_len or len(curr_batch) == bsz: # we're done
                # 一个tensor必须是连续的contiguous()才能被查看
                minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                    curr_labels, 
                                    self.padded_feat_mb(curr_feats),
                                    self.padded_loc_mb(curr_locs),
                                    self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous()))
                mb2linenos.append(curr_linenos)
                curr_batch = [sents[i]]
                curr_len = len(sents[i])
                curr_labels = [labels[sorted_idxs[i]]]
                curr_feats = [feats[sorted_idxs[i]]]
                curr_locs = [locs[sorted_idxs[i]]]
                curr_inps = [inps[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
            else:
                curr_batch.append(sents[i])
                curr_labels.append(labels[sorted_idxs[i]])
                curr_feats.append(feats[sorted_idxs[i]])
                curr_locs.append(locs[sorted_idxs[i]])
                curr_inps.append(inps[sorted_idxs[i]])
                curr_linenos.append(sorted_idxs[i])
        # catch last
        if len(curr_batch) > 0:
            minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                curr_labels, 
                                self.padded_feat_mb(curr_feats),
                                self.padded_loc_mb(curr_locs),
                                self.padded_inp_mb(curr_inps).transpose(0, 1).contiguous()))
            mb2linenos.append(curr_linenos)
        return minibatches, mb2linenos

if __name__ == '__main__':
    data_path = '/Users/shang/Work/NLG/neural-template-gen/data/e2e_aligned'
    data = data_path + '/src_uniq_valid.txt'
    with open(data) as f:
        src_lines = f.readlines()[:3]
    for ll, src_line in enumerate(src_lines):
        src_tbl = get_e2e_poswrds(src_line.strip().split())
        fld_cntr = Counter([key for key, _ in src_tbl])
        print dict(fld_cntr), '\t',src_line


