import pandas as pd
import string

#filtering section text
def preprocess(te_xt):
    if isinstance(te_xt, str):
        te_xt = te_xt.translate(str.maketrans('', '', string.punctuation))
        te_xt = te_xt.replace('\n', '')
        te_xt = te_xt.lower()
        return te_xt.strip()
    return ''

#dividing dataset in to chnuks and doing pre processing 
Temp=[]
for c1 in pd.read_csv("data.csv", chunksize=9500):
    c=pd.DataFrame()
    c['SECTION_TEXT']=c1.iloc[:, 3]
    c['SECTION_TEXT']=c['SECTION_TEXT'].apply(preprocess)
    c=c[c['SECTION_TEXT']!='']
    Temp.append(c)


#concating and storing process data in Process_Data.txt
df=pd.concat(Temp)
df.reset_index(drop=True, inplace=True)
df.to_csv('Process_Data.txt', index=True)
