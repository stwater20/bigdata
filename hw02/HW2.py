from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, HiveContext, SQLContext
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import *
import pandas as pd

AVG_POP_PH_FILENAME = "_avg_popularity_per_hour.csv"
AVG_POP_PD_FILENAME = "_avg_popularity_per_day.csv"
MASTER = "spark://192.168.225.128:7077"

conf = SparkConf().setAppName("HW2").setMaster(MASTER)
spark = SparkSession.builder\
         .appName("HW2")\
         .master(MASTER)\
         .config(conf=conf)\
         .getOrCreate()

def count_words_in_col(col_name, df, output_filename):
    result_df = df.withColumn('word', split(col(col_name), '\s+'))
    result_df = result_df.withColumn('word', array_remove(col('word'), ""))
    result_df = result_df.withColumn('word', explode(col('word')))\
                .groupBy('word')\
                .count()\
                .sort('count', ascending=False)
    result_df.show()
    result_df.toPandas().to_csv(output_filename, header=True, index=False, encoding="utf-8")
    
    return result_df

def count_words_per_day(col_name, df, output_filename):
    result_df = df.withColumn('PublishDate', date_format('PublishDate', 'yyyy-MM-dd'))
    result_df = result_df.select(col_name, 'PublishDate')
    
    result_df = result_df.withColumn('word', split(col(col_name), '\s+'))
    result_df = result_df.withColumn('word', array_remove(col('word'), ""))
    result_df = result_df.withColumn('word', explode(col('word')))\
        .groupBy('PublishDate', "word")\
        .agg({"word" : "count"})\
        .withColumnRenamed("count(word)", "word_count")\
        .orderBy(col('PublishDate').desc())
    
    win = Window.partitionBy(result_df["PublishDate"]).orderBy(result_df["word_count"].desc())
    result_df = result_df.select("*", rank().over(win).alias("rank")).filter(col("rank") <= 1)
    result_df = result_df.sort("PublishDate", ascending=False)
    result_df.show()
    result_df.toPandas().to_csv(output_filename, header=True, index=False, encoding="utf-8")

def count_words_each_topic(col_name, df, output_filename, is_q4):
    result_df = df.select(col_name, 'Topic')
    result_df = result_df.withColumn('word', split(col(col_name), '\s+'))
    
    # remove "" from the word arrays
    result_df = result_df.withColumn('word', array_remove(col('word'), ""))
    result_df = result_df.withColumn('word', explode(col('word')))\
        .groupBy('Topic', 'word')\
        .agg(count("word").alias("word_count"))

    win = Window.partitionBy(result_df["Topic"]).orderBy(result_df["word_count"].desc())
    if is_q4:
        result_df = result_df.select("*", rank().over(win).alias("rank")).filter(col("rank") <= 100)
        
        return result_df
    else:
        result_df = result_df.select("*", rank().over(win).alias("rank")).filter(col("rank") <= 1)
#     result_df = result_df.select("*", rank().over(win).alias("rank"))
    result_df = result_df.select('Topic', 'word', 'word_count')
    result_df.show(truncate=False)
    result_df.toPandas().to_csv(output_filename, header=True, index=False, encoding="utf-8")
    
    return 

def cal_avg_popularity(df, output_filename, option):
    """
    calculate the average popularity of each news by hour or by day.
    option: "d" => by day; "h" => by hour
    """
    time_unit_count = len(df.columns) - 1
    result_df = df.select("IDLink")
    
    if option == "h":
        for col_idx in range(0, time_unit_count, 3):
            avg_pop_per_hour = df.select("IDLink",\
                            ((col("TS"+str(col_idx+1)) + col("TS"+str(col_idx+2)) + col("TS"+str(col_idx+3)))\
                            / lit(3)).alias("hour("+str(col_idx/3+1)+")"))

            result_df = result_df.join(avg_pop_per_hour, ["IDLink"])
    
    elif option == "d":
        for col_idx in range(0, time_unit_count, 72):
            day_data = col("TS"+str(col_idx+1))
            for i in range(2, 73):
                day_data += col("TS"+str(col_idx+i))

            avg_pop_per_day = df.select("IDLink",(day_data / lit(72)).alias("day("+str(col_idx/72+1)+")"))
            result_df = result_df.join(avg_pop_per_day, ["IDLink"])
    
    result_df.toPandas().to_csv(output_filename, header=True, index=False, encoding="utf-8")

def union_df():
    fb_eco_df = spark.read.options().csv("dataset/Facebook_Economy.csv", header=True)
    fb_mic_df = spark.read.options().csv("dataset/Facebook_Microsoft.csv", header=True)
    fb_obama_df = spark.read.options().csv("dataset/Facebook_Obama.csv", header=True)
    fb_pal_df = spark.read.options().csv("dataset/Facebook_Palestine.csv", header=True)
    
    google_eco_df = spark.read.options().csv("dataset/GooglePlus_Economy.csv", header=True)
    google_mic_df = spark.read.options().csv("dataset/GooglePlus_Microsoft.csv", header=True)
    google_obama_df = spark.read.options().csv("dataset/GooglePlus_Obama.csv", header=True)
    google_pal_df = spark.read.options().csv("dataset/GooglePlus_Palestine.csv", header=True)
    
    link_eco_df = spark.read.options().csv("dataset/LinkedIn_Economy.csv", header=True)
    link_mic_df = spark.read.options().csv("dataset/LinkedIn_Microsoft.csv", header=True)
    link_obama_df = spark.read.options().csv("dataset/LinkedIn_Obama.csv", header=True)
    link_pal_df = spark.read.options().csv("dataset/LinkedIn_Palestine.csv", header=True)
    
    fb_df = fb_eco_df
    fb_df = fb_df.union(fb_mic_df)
    fb_df = fb_df.union(fb_obama_df)
    fb_df = fb_df.union(fb_pal_df)
    
    google_df = google_eco_df
    google_df = google_df.union(google_mic_df)
    google_df = google_df.union(google_obama_df)
    google_df = google_df.union(google_pal_df)
    
    link_df = link_eco_df
    link_df = link_df.union(link_mic_df)
    link_df = link_df.union(link_obama_df)
    link_df = link_df.union(link_pal_df)
    
    return fb_df, google_df, link_df
    

def cal_sentiment_score_sum_avg(df):
    result_df = df.select('Topic', 'SentimentTitle', 'SentimentHeadline')
    result_df = result_df\
        .groupBy('Topic')\
        .agg(sum('SentimentTitle'), sum('SentimentHeadline'), \
             mean('SentimentTitle'), mean('SentimentHeadline'))
    result_df.show()
    result_df.toPandas().to_csv("sentiment_avg_sum.csv", header=True, index=False, encoding="utf-8")
    
def generate_co_occ_matrix(top_100_words_df, title_df, col_name, topic):
    top_100_words_df = top_100_words_df.filter(col('Topic') == topic)
    top_100_words_df = top_100_words_df.select('word')
    top_100_words2_df = top_100_words_df.withColumnRenamed('word', 'word-')
    
    print("=== Generate 100 x 100 word pairs ===")
    word_pairs = top_100_words_df.crossJoin(top_100_words2_df)
    pairs = word_pairs.select('word', 'word-')\
            .where(col('word') >= col('word-'))\
            .collect()
    
    words_df = title_df.withColumn('Words', split(col(col_name), '\s+')) 
    
    co_occ_list = []
     
    for p in pairs:
        res = words_df.filter(array_contains(col('Words'), p[0]))
        res = res.filter(array_contains(col('Words'), p[1]))
        occ = res.count()

        for i in range(occ):
            co_occ_list.append((p[0], p[1]))
            
            if p[0] != p[1]:
                co_occ_list.append((p[1], p[0]))
    
    result_df = spark.createDataFrame(co_occ_list, ["word1", "word2"])
    co_occ_matrix = result_df.stat.crosstab("word1", "word2")
    print("===============DONE!!!===================")
    co_occ_matrix.toPandas().to_csv("co_occ_mat_" + topic + "_" + col_name + ".csv",\
                                    header=True, index=False, encoding="utf-8")
    
def main():
    pd_df = pd.read_csv("dataset/News_Final.csv")
    pd_df = pd_df.astype(str)
    df = spark.createDataFrame(pd_df)
    
    reg_pattern = "[^[A-Za-z ]+]"
    replace_chr = ""
    df = df.withColumn("Title", regexp_replace("Title", reg_pattern, replace_chr))\
           .withColumn("Headline", regexp_replace("Headline", reg_pattern, replace_chr))\
           .withColumn("Source", regexp_replace("Source", reg_pattern, replace_chr))
    
    """
    Q1
    """
    title_word_count = count_words_in_col('Title', df, "Title_word_count_all.csv")
    headline_word_count = count_words_in_col('Headline', df, "Headline_word_count_all.csv")

    count_words_per_day('Title', df, "Title_word_count_day.csv")
    count_words_per_day('Headline', df, "Headline_word_count_day.csv")

    count_words_each_topic('Title', df, "Title_word_count_topic.csv", False)
    count_words_each_topic('Headline', df, "Headline_word_count_topic.csv", False)
    
    """
    Q2
    """
    
    fb_df, google_df, link_df = union_df()
    
    cal_avg_popularity(fb_df, "FB"+AVG_POP_PH_FILENAME, "h")
    cal_avg_popularity(fb_df, "FB"+AVG_POP_PD_FILENAME, "d")
    
    cal_avg_popularity(google_df, "Google"+AVG_POP_PH_FILENAME, "h")
    cal_avg_popularity(google_df, "Google"+AVG_POP_PD_FILENAME, "d")
    
    cal_avg_popularity(link_df, "LinkedIn"+AVG_POP_PH_FILENAME, "h")
    cal_avg_popularity(link_df, "LinkedIn"+AVG_POP_PD_FILENAME, "d")
    
    """
    Q3
    """
    cal_sentiment_score_sum_avg(df)
    
    """
    Q4
    """
    top_100_words_title = count_words_each_topic('Title', df, "", True)
    top_100_words_headline = count_words_each_topic('Headline', df, "", True)
    title_df = df.select('Title')
    headline_df = df.select('Headline')
    
    generate_co_occ_matrix(top_100_words_title, title_df, 'Title', 'economy')
    generate_co_occ_matrix(top_100_words_headline, headline_df, 'Headline', 'economy')
    
    generate_co_occ_matrix(top_100_words_title, title_df, 'Title', 'obama')
    generate_co_occ_matrix(top_100_words_headline, headline_df, 'Headline', 'obama')

    generate_co_occ_matrix(top_100_words_title, title_df, 'Title', 'microsoft')
    generate_co_occ_matrix(top_100_words_headline, headline_df, 'Headline', 'microsoft')
    
    generate_co_occ_matrix(top_100_words_title, title_df, 'Title', 'palestine')
    generate_co_occ_matrix(top_100_words_headline, headline_df, 'Headline', 'palestine')

if __name__ == "__main__":
    main()