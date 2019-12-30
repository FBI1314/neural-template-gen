# -*- coding: utf-8 -*-
# @Author: shangerxin
# @Date:   2019-12-30
# @Last Modified by:   shangerxin
# @Last Modified time: 2019-12-30

import re
from collections import defaultdict

import torch

import labeled_data

seg_patt = re.compile('([^\|]+)\|(\d+)') # detects segments

def group_by_template(fi, startlineno):
    """
    returns a label-tup -> [(phrase-list, lineno), ...] map
    """
    labes2sents = defaultdict(list)
    lineno = startlineno
    with open(fi) as f:
        for line in f:
            if '|' not in line:
                continue
            seq = seg_patt.findall(line.strip()) # list of 2-tuples
            wordseq, labeseq = zip(*seq) # 2 tuples
            wordseq = [phrs.strip() for phrs in wordseq]
            labeseq = tuple(int(labe) for labe in labeseq)
            labes2sents[labeseq].append((wordseq, lineno))
            lineno += 1
    return labes2sents

def remap_eos_states(top_temps, temps2sents):
    """
    allocates a new state for any state that is also used for an <eos>
    """
    used_states = set()
    [used_states.update(temp) for temp in top_temps]
    final_states = set()
    for temp in top_temps:
        final_state = temp[-1]
        assert any(sent[-1] == "<eos>" for sent, lineno in temps2sents[temp])
        final_states.add(final_state)

    # make new states
    remap = {}
    for i, temp in enumerate(top_temps):
        nutemp = []
        changed = False
        for j, t in enumerate(temp):
            if j < len(temp)-1 and t in final_states:
                changed = True
                if t not in remap:
                    remap[t] = max(used_states) + len(remap) + 1
            nutemp.append(remap[t] if t in remap else t)
        if changed:
            nutuple = tuple(nutemp)
            top_temps[i] = nutuple
            temps2sents[nutuple] = temps2sents[temp]
            del temps2sents[temp]

def just_state2phrases(temps, temps2sents):
    """
    Args:
        temps: list of string
        temps2sents: dict
    Return:
        state2phrases: dict, key是模板编号，value的第一项是list，表示这个模板编号对应所有可能的模板，第二项是一个tensor，所有模板的编码
    """
    state2phrases = defaultdict(lambda: defaultdict(int)) # defaultdict of defaultdict
    for temp in temps:
        for sent, lineno in temps2sents[temp]:
            for i, state in enumerate(temp):
                #state2phrases[state].add(sent[i])
                state2phrases[state][sent[i]] += 1

    nustate2phrases = {}
    for k, v in state2phrases.iteritems():
        phrases = list(v)
        counts = torch.Tensor([state2phrases[k][phrs] for phrs in phrases])
        counts.div_(counts.sum())
        nustate2phrases[k] = (phrases, counts)
    state2phrases = nustate2phrases
    return state2phrases


def extract_from_tagged_data(datadir, bsz, thresh, tagged_fi, ntemplates):
    """
    Args:
        datadir: str
        bsz: int
        thresh: int
        tagged_fi: str
        ntemplates: int
    Return:
        top_temps: list of tuple(str) # [(55, 59, 43, 11, 25, 40, 53, 19)]，每一个数字代表一个state
        temps2sents: dict, key string见top_temps单项, value [(phrase-list, lineno), ...]phrase-list表明这个模板有几个词，len(phrase-list) == len(k)
                比如，k=(55, 59, 43, 11, 25, 40, 53, 19), 
                    v=[(['<unk> Cambridge', 'is', 'high', 'priced', 'English', 'food', ',', 'that is', 'average', 'customer rated', '.', '<eos>'], 5701)]
                    在这里55对应的是<unk> Cambridge，但是它还可以对应 '<unk> <unk> <unk>'，'<unk> <unk>', '<unk> of Cambridge', 'The <unk>', 'the <unk>', '<unk> Cambridge', 'The <unk> <unk>', '<unk>' 
        state2phrases: dict key是state编号，value的第一项是list，表示这个模板编号对应所有可能的模板，第二项是一个tensor，所有模板的编码
                还是以55为例，它对应9个模板，55对应的向量tensor([长度是9]))

    """
    corpus = labeled_data.SentenceCorpus(datadir, bsz, thresh=thresh, add_bos=False,
                                         add_eos=False, test=False)
    nskips = 0
    # 丢掉少于4个字的
    for i in xrange(len(corpus.train)):
        if corpus.train[i][0].size(0) <= 4:
            nskips += corpus.train[i][0].size(1)
    print "assuming we start on line", nskips, "of train"
    temps2sents = group_by_template(tagged_fi, nskips)
    top_temps = sorted(temps2sents.keys(), key=lambda x: -len(temps2sents[x]))[:ntemplates]
    #remap_eos_states(top_temps, temps2sents)
    state2phrases = just_state2phrases(top_temps, temps2sents)


    return top_temps, temps2sents, state2phrases


def topk_phrases(pobj, k):
    phrases, probs = pobj
    thing = sorted(zip(phrases, list(probs)), key=lambda x: -x[1])
    sphrases, sprobs = zip(*thing)
    return sphrases[:k]


def align_cntr(cntr, thresh=0.4):
    tote = float(sum(cntr.values()))
    nug = {k : v/tote for k, v in cntr.iteritems()}
    best, bestp = None, 0
    for k, v in nug.iteritems():
        if v > bestp:
            best, bestp = k, v
    if bestp >= thresh:
        return best
    else:
        return None
