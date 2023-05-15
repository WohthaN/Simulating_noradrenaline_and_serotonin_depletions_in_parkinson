from models import *
import pickle

for n in range(241):
    try:
        with open('fitted_models/population/S_%03i' % n, 'rb') as f:
            s = pickle.load(f)
    except FileNotFoundError:
        continue
    p = s['parameters']
    for k in list(p.keys()):
        p[k.replace('6OHDA','DA').replace('pCPA', '5HT').replace('DSP4', 'NE')] = p.pop(k)
    s['parameters'] = p
    with open('fitted_models/population/S_%03i' % n, 'bw') as f:
        pickle.dump(s, f)
    
    
        
