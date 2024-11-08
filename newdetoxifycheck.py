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



def predictsentencetoxicity(texts_col, modelname='unbiased'):
    if texts_col is not None and bool(texts_col.strip()):
        detox_model = Detoxify(modelname)
        result = detox_model.predict(str(texts_col))
        sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        maxoftwo = dict(Counter(sorted_result).most_common(2))        
        #avgofmaxtwopercentage = (sum(maxoftwo.values()) / len(maxoftwo) * 100)
        avgofmaxtwopercentage = ((sum(maxoftwo.values()) / len(maxoftwo)) * 100)
        summationpercenatage = (sum(maxoftwo.values()) * 100)
        toxicityeval = 1 if summationpercenatage > 90 else 0
        result['question'] = texts_col
        result['maxoftwo'] = maxoftwo
        result['summationpercenatage'] = summationpercenatage
        #result['avgofmaxtwopercentage'] = avgofmaxtwopercentage
        result['toxicityeval'] = toxicityeval
        return result


def calculatetoxicity(df, texts_col, modelname='unbiased'):
    detox_model = Detoxify(modelname)
    df['toxic_result'] = dict(df[texts_col].apply(lambda t: detox_model.predict(str(t))))
    df['maxoftwo'] = df['toxic_result'].apply(lambda x: dict(sorted(Counter(x).most_common(2), key=lambda item: item[1], reverse=True)))
    #df['maxoftwo'] = df['toxic_result'].apply(lambda x: dict(Counter(x).most_common(2)))
    #maxoftwo  = df['maxoftwo'][0]
    #df['avgofmaxtwopercentage']= (sum(maxoftwo.values()) / len(maxoftwo) * 100)
    df['summationpercentage'] = df['values'].apply(lambda x: (sum(x.values()) * 100))
    
    #df['summationpercentage'] = (sum(df['maxoftwo'].values()) * 100)
    #df['avgofmaxtwopercentage'] = ((sum(maxoftwo.values()) / len(maxoftwo)) * 100)
    # changing to summationpercentage
    df['toxicityeval'] = df['summationpercentage'].apply(lambda x: 1 if x > 90 else 0) 
    return df


def predicttoxicity(df, texts_col, modelname='unbiased'):
    detox_model = Detoxify(modelname)
    responsedict = {}
    for sentence in df[texts_col]:
      result = detox_model.predict(sentence)
      # get two higly toxic 
      maxtwotoxic = dict(Counter(result).most_common(2))
      result['maxtwotoxic'] = maxtwotoxic 
      result['summationpercentage'] = '{:.2f}%'.format(sum(maxtwotoxic.values()))
      #result['avgofmaxtwopercentage']= '{:.2f}%'.format((sum(maxtwotoxic.values()) / len(maxtwotoxic))* 100)
      result['toxicityeval'] = result['summationpercentage'].apply(lambda x: 1 if x > 90 else 0) 
      responsedict[sentence] = result
      rdf = pd.DataFrame(list(responsedict.items()), columns=['Question', 'Response'])
      flatten_df = pd.json_normalize(rdf.to_dict(orient='records'))
      flatten_df.columns = flatten_df.columns.str.replace('Response.', '', regex=False)
    return flatten_df
    
