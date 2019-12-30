# -*- coding: utf-8 -*-
# @Author: shangerxin
# @Date:   2019-12-19
# @Last Modified by:   shangerxin
# @Last Modified time: 2019-12-27

from collections import defaultdict

# leaves out familyFriendly, which is a binary thing...
e2e_keys = ["name", "eatType", "food", "priceRange", "customerrating", "area", "near"]
e2e_key2idx = dict((key, i) for i, key in enumerate(e2e_keys))

def get_e2e_fields(tokes, keys=None):
    """
    assumes a key only appears once per line...
    returns keyname -> list of words dict
    """
    if keys is None:
        keys = e2e_keys
    fields = defaultdict(list)
    state = None
    for toke in tokes:
        if "__start" in toke:
            for key in keys:
                if toke == "__start_%s__" % key:
                    assert state is None
                    state = key
        elif "__end" in toke:
            for key in keys:
                if toke == "__end_%s__" % key:
                    assert state == key
                    state = None
        elif state is not None:
            fields[state].append(toke)

    return fields

def get_e2e_poswrds(tokes):
    """将标记好的模板格式化，key是类别，比如姓名，价格变化区间，每一类key对应若干词，用num标记他们的顺序

    assumes a key only appears once per line...
    returns (key, num) -> word
    比如 tokens='''__start_name__ The Vaults __end_name__ __start_eatType__ pub __end_eatType__ __start_priceRange__ more than £ 30 __end_priceRange__ __start_customerrating__ 5 out of 5 __end_customerrating__ __start_near__ Café Adriatic __end_near__'''
    返回：
        {('_customerrating', 1): '5',
         ('_customerrating', 2): 'out',
         ('_customerrating', 3): 'of',
         ('_customerrating', 4): '5',
         ('_eatType', 1): 'pub',
         ('_name', 1): 'The',
         ('_name', 2): 'Vaults',
         ('_near', 1): 'Café',
         ('_near', 2): 'Adriatic',
         ('_priceRange', 1): 'more',
         ('_priceRange', 2): 'than',
         ('_priceRange', 3): '£',
         ('_priceRange', 4): '30'}
    """
    fields = {}
    state, num = None, 1 # 1-idx the numbering
    for toke in tokes:
        if "__start" in toke:
            assert state is None
            state = toke[7:-2]
        elif "__end" in toke:
            state, num = None, 1
        elif state is not None:
            fields[state, num] = toke
            num += 1
    return fields


def get_wikibio_fields(tokes, keep_splits=None):
    """
    key -> list of words
    """
    fields = defaultdict(list)
    for toke in tokes:
        try:
            fullkey, val = toke.split(':')
        except ValueError:
            ugh = toke.split(':') # must be colons in the val
            fullkey = ugh[0]
            val = ''.join(ugh[1:])
        if val == "<none>":
            continue
        #try:
        keypieces = fullkey.split('_')
        if len(keypieces) == 1:
            key = fullkey
        else:
            keynum = keypieces[-1]
            key = '_'.join(keypieces[:-1])
            #key, keynum = fullkey.split('_')
        #except ValueError:
        #    key = fullkey
        if keep_splits is None or key not in keep_splits:
            fields[key].append(val) # assuming keys are ordered...
        else:
            fields[fullkey].append(val)
    return fields


def get_wikibio_poswrds(tokes):
    """
    (key, num) -> word
    """
    fields = {}
    for toke in tokes:
        try:
            fullkey, val = toke.split(':')
        except ValueError:
            ugh = toke.split(':') # must be colons in the val
            fullkey = ugh[0]
            val = ''.join(ugh[1:])
        if val == "<none>":
            continue
        #try:
        keypieces = fullkey.split('_')
        if len(keypieces) == 1:
            key = fullkey
            #keynum = '0'
            keynum = 1
        else:
            keynum = int(keypieces[-1])
            key = '_'.join(keypieces[:-1])
        fields[key, keynum] = val
    return fields
