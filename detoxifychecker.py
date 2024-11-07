from detoxify import Detoxify
import pandas as pd
import numpy as np

#data = pd.read_csv('/content/sample.csv')

#print(f"Number of rows: {data.shape[0]}")
#print(f"Number of cols: {data.shape[1]}")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)

def predict(df,texts_col):
	# Initialize the Detoxify model
    detox_model = Detoxify('unbiased')

    responsedict = {}
    querylist=[]

    for sentence in df[texts_col]:
      result = detox_model.predict(sentence)
      rounded_result = {k: round(v, 3) for k, v in result.items()}
      rounded_result['combinedtoxicity'] = round(sum(value for value in rounded_result.values() if value > 0), 3)
      rounded_result['percentage'] = (rounded_result['combinedtoxicity'] / 7 * 100)
      
      responsedict[sentence] = rounded_result
      rdf = pd.DataFrame(list(responsedict.items()), columns=['Question', 'Response'])
      flatten_df = pd.json_normalize(rdf.to_dict(orient='records'))
      flatten_df.columns = flatten_df.columns.str.replace('Response.', '', regex=False)
    return flatten_df

def smallsetrun(df, numbervalue, colname, exportfilepath): 
    from datetime import datetime
    start_time = datetime.now()
    numbervalue = int(numbervalue)
    texts_col = df[colname][:numbervalue]
    resultsample = predict(df, texts_col)
    resultsample.to_csv(exportfilepath, index=False)
    end_time = datetime.now()
    duration_in_minutes = (end_time - start_time) / 60
    print(f"Duration in minutes: {duration_in_minutes}")
    