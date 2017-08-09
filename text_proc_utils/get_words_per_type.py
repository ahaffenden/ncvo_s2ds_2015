#! /usr/bin/env python
#Created on Thd, 13th of Aug.
#@author: Anna

# read data with words for each entry in the data and extract the words for each income type, then print to a file
import os
import pandas

from pandas import read_csv
df = read_csv("../../data/features/data_frame.csv")

df_sub = df.groupby('type_class')

all_sets = []
all_sets_id = []

OUTPUT_DIR = "../../data/" + "OUT_words_stat" 
for typeId, data_sub in df_sub:
    out_file_name = "sep_uniq_" + typeId + '.txt'
    output_path_name = os.path.join(OUTPUT_DIR, out_file_name)    
    f = open(output_path_name, 'w')
    iv_set = set()
    df_sub_tmp = df_sub.get_group(typeId)
    tem = df_sub_tmp.iloc[0:,2:].values.tolist()
    list(map(iv_set.update, tem))
    f.writelines(str(iv_set))
    f.close()
   