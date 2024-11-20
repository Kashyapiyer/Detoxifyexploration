from re import X
from detoxify import Detoxify
import pandas as pd
import numpy as np
from collections import Counter
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)



def sentencetoxicityvalidator(contextstr, modelname='unbiased'):
    try: 
        if contextstr is not None and bool(contextstr.strip()):
            detox_model = Detoxify(modelname)
            result = detox_model.predict(str(contextstr))
            sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
            maxofthree = dict(Counter(sorted_result).most_common(3))        
            summationpercenatage = (sum(maxoftwo.values()) * 100)
            toxicityrisk = 2 if summationpercenatage>=50.00 else 1 if 35.00 <= summationpercenatage <=50.00 else 0 
            result['question'] = contextstr
            result['maxofthree'] = maxofthree
            result['summationpercenatage'] = summationpercenatage
            result['toxicityrisk'] = toxicityrisk
            result['toxicityeval'] = 1 if toxicityrisk in [1,2] else 0
            if result['toxicityeval'] == 1: 
               result['validationmessage'] = "Highly toxic content detected hence further inference is prevented"
            else: 
               result['validationmessage'] = "Validation passed"
            return result
    except Exception as e: 
        raise f'Encountered error while processing sentence toxicity{e}'
