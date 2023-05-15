from models import *
import pickle


fname = 'fitted_models/population/S_%03i'

for n in range(241):
    try:
        with open( fname % n, 'rb') as f:
            s = pickle.load(f)
    except FileNotFoundError:
        continue
    
    s['name'] = s['name'].replace('6OHDA','DA').replace('pCPA', '5HT').replace('DSP4', 'NE')
    
    with open(fname  % n, 'bw') as f:
        pickle.dump(s, f)
    
    
        
