from models import *
import pickle

fnames_postfix = [('fitted_models/population/S_%03i', ''),
    ('fitted_models/population/CURE_DRN/S_%03i', '_+LDA_+cure_DRN'),
    ('fitted_models/population/CURE_DRN_LC/S_%03i', '_+LDA_+cure_DRN_+cure_LC'),
    ('fitted_models/population/CURE_LC/S_%03i', '_+LDA_+cure_LC')]


def fix_dict_key(d):
    for k, v in list(d.items()):
        if 'SNc' in k and 'SNcVTA' not in k:
            d.pop(k)
            newk = k.replace('SNc', 'SNcVTA')
            d[newk] = v
            print('   fixed %s to %s' % (k, newk))
    return d

def fix_list_element(l):
    for i,e in enumerate(l):
        if 'SNc' in e and 'SNcVTA' not in e:
            l[i] = e.replace('SNc', 'SNcVTA')
            print('   fixed list element %s to %s' % (e, l[i]))
    return l

for fname, postfix in fnames_postfix:
    for n in range(241):
        try:
            with open(fname % n + postfix, 'rb') as f:
                s = pickle.load(f)
        except FileNotFoundError:
            continue
        print(fname % n)

        # s['name'] = s['name'].replace('6OHDA','DA').replace('pCPA', '5HT').replace('DSP4', 'NE')
        # s['equations'] = [x.replace('SNc', 'SNcVTA') for x in s['equations']]
        s['equations'] = fix_list_element(s['equations'])

        keys = ['parameters', 'parameters_constraints', 'constants', 'target', 'target_constraints', 'parameters_signs']
        for k in keys:
            # print('  ', k, s[k])
            s[k] = fix_dict_key(s[k])

        with open(fname % n + postfix, 'bw') as f:
            pickle.dump(s, f)
