import pandas as pd

input_filename = '/home/yyg/image_retrival/yyg/open_clip/data/train_data.csv\n'
df = pd.read_csv(input_filename, sep='\t')
print(df.head())
