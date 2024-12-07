import os
import sys
import codecs
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from random import shuffle
from bs4 import BeautifulSoup
from bs4.diagnose import diagnose
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, HiveContext, SQLContext
from pyspark.sql import functions as F
from pyspark.sql import types as T
# from pyspark.sql.functions import *

CWD = os.getcwd()
DATASET_DIR = os.path.join(CWD, "dataset")
FILE_LIST = os.listdir("dataset")

MASTER = "spark://192.168.225.128:7077"

conf = SparkConf().setAppName("HW3").setMaster(MASTER)
sc = SparkContext(conf=conf)

sc.setCheckpointDir('checkpoint')

spark = SparkSession.builder\
            .appName("HW3")\
            .master(MASTER)\
            .config("spark.sql.shuffle.partitions", 50)\
            .getOrCreate()

def sgml_files_to_csv():
    """
    - Extract the information of the tags in the SGML files
    - Combine the information in each SGML file and output a csv file
    """
    old_id_list = []
    new_id_list = []
    dataset_type_list = [] # dataset type = test or training
    title_list = []
    body_list = []
    
    print("=========== Converting the SGML files to csv ===========")
    # iterate through 22 files
    for filename in FILE_LIST:
        file_abspath = os.path.join(DATASET_DIR, filename)
        
        with codecs.open(file_abspath, 'r', encoding='utf-8', \
                 errors='ignore') as file:
            file_content = file.read()
    
        soup = BeautifulSoup(file_content, 'html.parser')
        reuters = soup.find_all('reuters')
        date_tags = soup.find_all('date')
        title_tags = soup.find_all('title')
        body_tags = soup.find_all('body')
        text_tags =  soup.find_all('text')

        # iterate through news
        for idx, r in enumerate(reuters):
            old_id = r.get('oldid')
            new_id =r.get('newid')
            dataset_type = r.get('lewissplit')
            title_tag = r.findChildren('title' , recursive=True)
            body_tag = r.findChildren('body' , recursive=True)

            if title_tag:
                title = title_tag[0].get_text()
                title_list.append(title)
            else:
                title_list.append('')

            if body_tag:
                body = body_tag[0].get_text()
                body = body.replace("\n", " ")
                body = ' '.join(body.split()) # replace multiple whitespace to one whitespace
                body_list.append(body)
            else:
                body_list.append('')

            old_id_list.append(old_id)
            new_id_list.append(new_id)
            dataset_type_list.append(dataset_type)

    output_data = {'New_ID': new_id_list,\
                        'Old_ID': old_id_list,\
                        'Train_Test': dataset_type_list,\
                        'Title': title_list,\
                        'Body': body_list}
    output_df = pd.DataFrame(data=output_data)
    
    df = spark.createDataFrame(output_df)
    
    reg_pattern = "[^[A-Za-z ]+]"
    replace_chr = " "
    df = df.withColumn("Body", F.regexp_replace("Body", reg_pattern, replace_chr))
    
    df.toPandas().to_csv("documents.csv", index=False, encoding="utf-8")

def is_contain(a, b):
    if len(set(b) - set(a)) == 0:
        return "1"
    return "0"

def shingling(k):
    df = spark.read.options().csv("documents.csv", header=True)
    df = df.withColumn('words', F.split(F.col('Body'), '\s+'))
    df = df.withColumn('words', F.array_remove(F.col('words'), ""))
    
    # convert the dataframe column to a list
    words_list = df.select('words').rdd.flatMap(lambda x: x).collect()
    doc_id_list = df.select('New_ID').rdd.flatMap(lambda x: x).collect()
    
    words_list = [x for x in words_list if x is not None]
    all_shingles = list(itertools.chain.from_iterable(words_list)) # 2d -> 1d list

    shingle_set = set()
    
    # form a shingle set
    for doc_words in words_list:
        for i in range(len(doc_words)-k+1):
            shingle = tuple(sorted(tuple((doc_words[i:i+k]))))
            shingle_set.add(shingle)
    
    shingle_list = list(shingle_set)
    
    
    # replace Null with an empty array
#     df = df.withColumn('words', when(df['words'].isNull(), array([])).otherwise(df['words']))

    partital_df = df.select('New_ID', 'words')
    
    # filter out the document without content
    partital_df = partital_df.filter(~F.col('words').isNull())
    print("The number of shingles: "+str(len(shingle_list)))
    
    shingle_bool_vec = []
    for shingle in tqdm(shingle_list):
        contain_udf = F.udf(is_contain, T.StringType())
        shingle_name = '(' + ", ".join(shingle) + ')'
        
        partital_df = partital_df.withColumn(shingle_name, contain_udf(\
                                        partital_df['words'], F.array([F.lit(x) for x in list(shingle)])))
#         partital_df = partital_df.withColumn('is_occ', contain_udf(\
#                                         partital_df['words'], F.array([F.lit(x) for x in list(shingle)])))
# #         occ = list(partital_df.select('is_occ').toPandas()['is_occ'])
#         occ = partital_df.select('is_occ').rdd.flatMap(lambda x: x).collect()
#         occ.insert(0, shingle_name)
#         shingle_bool_vec.append(occ)
        partital_df = partital_df.checkpoint()
    
#     doc_id_list.insert(0, 'shingle')
#     result_df = spark.createDataFrame(shingle_bool_vec, doc_id_list)
#     result_df.toPandas().to_csv("shingles.csv", header=True, index=False, encoding="utf-8")
    partital_df = partital_df.drop('words')
    partital_df = partital_df.checkpoint()
    partital_df.toPandas().set_index("New_ID").transpose().to_csv("shingles.csv", index=True, encoding="utf-8")

    return all_shingles

def find_signature(col_data, hash_f):
    signature = 1
    
    for idx in hash_f:
        if col_data[idx] == "1":
            break
        signature += 1
    
    return signature
        

def min_hash(h):
    df = spark.read.options().csv("shingles.csv", header=True)
    shingles = df.select('_c0').rdd.flatMap(lambda x: x).collect() # a list of all shingles
    num_shingles = len(shingles) # the number of shingles
    col_name = df.schema.names
    
    hash_f = list(range(0, num_shingles))
    
    signature_mat = []
    print("================= min-hashing ===========")
    print("The number of hash function: " + str(h))
    print("The number of documents: " + str(len(col_name)-1))
    for it in tqdm(range(h)):
        shuffle(hash_f)
        doc_sig_list = []
        for col in col_name: # iterate through each document
            if col != "_c0":
                col_data = df.select(col).rdd.flatMap(lambda x: x).collect()
                signature = find_signature(col_data, hash_f)
                doc_sig_list.append(signature)
        signature_mat.append(doc_sig_list)

    return col_name[1:], signature_mat

def cal_jaccard_sim(doc1, doc2):
    doc1_set = set(doc1)
    doc2_set = set(doc2)
    
    return len(doc1_set.intersection(doc2_set)) / len(doc1_set.union(doc2_set))
    
def find_candidate_pairs(doc_ID, num_doc, doc_band_sig, sim_threshold=0.8):
    for doc1_idx in range(num_doc-1):
        for doc2_idx in range(doc1_idx+1, num_doc):
            jaccard_sim = cal_jaccard_sim(doc_band_sig[doc1_idx], doc_band_sig[doc2_idx])
            if jaccard_sim >= sim_threshold: # candidate pair
                yield (doc_ID[doc1_idx], doc_ID[doc2_idx])

def LSH(doc_ID, signature_mat, num_bands, sim_threshold):
    if len(signature_mat) % num_bands != 0:
        print("Cannot partition the signature into " +str(num_bands)+ \
              " bands !! Please choose another band number")
        return
    
    rows = int(len(signature_mat) / num_bands)
    signature_mat = np.array(signature_mat)
    num_doc = len(doc_ID)
    
    candidate_pairs = set()
    
    print("================= LSH ================")
    for idx in tqdm(range(0, len(signature_mat), rows)):
        band = signature_mat[idx: idx+rows, :]
        doc_band_sig = np.array(band).T.tolist()
        
        for cp in find_candidate_pairs(doc_ID, num_doc, doc_band_sig, sim_threshold):
            candidate_pairs.add(cp)
        
    result_df = spark.createDataFrame(list(candidate_pairs), ["candidate_pair"])
    result_df.toPandas().to_csv("candidate_pairs.csv", header=True, index=False, encoding="utf-8")
    
    return candidate_pairs

def main():
    k = 3
    h = 20
    num_bands = 5
    sim_threshold = 0.5
#    k = int(sys.argv[1])
#    h = int(sys.argv[2])
#    num_bands = int(sys.argv[3])
#    sim_threshold = float(sys.argv[4])
#    sgml_files_to_csv()
    shingling(k)
    doc_ID, signature_mat = min_hash(h)
    candidate_pairs = LSH(doc_ID, signature_mat, num_bands=num_bands, sim_threshold=sim_threshold)
    print(candidate_pairs)
    
if __name__ == "__main__":
    main()
