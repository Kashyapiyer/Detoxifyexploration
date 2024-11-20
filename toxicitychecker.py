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



def calculatetoxicityratio(df, contextstr,gtlabel='label',modelname='unbiased'):
    import collections
    from collections import Counter
    detox_model = Detoxify(modelname)
    df['toxicity_result'] = df[contextstr].apply(lambda t: detox_model.predict(t))   
    df['toxicity_percentage'] = df['toxicity_result'].apply(lambda p: {k: round(v * 100, 2) for k, v in p.items()})
    #df['toxicityrisk'] = df['toxicity_percentage'].apply(lambda x: 2 if any(v > 7 for v in x.values()) else 1 if any(0.4 < v < 7 for v in x.values()) else 0)
    df['toxicityrisk'] = df['toxicity_percentage'].apply(lambda x: 2 if any(v > 5.0 for v in x.values()) else 1 if any(3.5 < v < 5.0 for v in x.values()) else 0)
    df['toxicityeval'] = df['toxicityrisk'].apply(lambda x: 1 if x in [1,2] else 0)
    df['match'] = df[gtlabel] == df['toxicityeval']
    return df


def toxicityratiovalidator(contextstr, modelname='unbiased'):
    try: 
        if contextstr is not None and bool(contextstr.strip()):
            detox_model = Detoxify(modelname)
            result = detox_model.predict(str(contextstr))
            rat_result =  {k: round(v * 100, 2) for k, v in result.items()}
            toxicityrisk = 2 if any(v > 7 for v in rat_result.values()) else 1 if any(0.4 < v < 7 for v in rat_result.values()) else 0
            result['question'] = contextstr
            result['ratpercent'] = rat_result
            result['toxicityrisk'] = toxicityrisk
            result['toxicityeval'] = 1 if toxicityrisk in [1,2] else 0
            if result['toxicityeval'] == 1: 
               result['validationmessage'] = "Highly toxic content detected hence further inference is prevented"
            else: 
               result['validationmessage'] = "Validation passed"
            return result
    except Exception as e: 
        return f'Encountered error while processing sentence toxicity{e}'
