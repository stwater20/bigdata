from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, count, size, split, countDistinct, desc, explode, to_date, when,  regexp_extract
from pyspark.sql import functions as F

MASTER = "spark://192.168.225.128:7077"
conf = SparkConf().setAppName("HW1").setMaster(MASTER)
sc = SparkContext(conf=conf)
spark = SparkSession.builder \
    .appName("HW1") \
    .master(MASTER) \
    .getOrCreate()

# Load dataset from CSV file
df = spark.read.csv("./spacenews.csv", header=True, inferSchema=True)


date_pattern = "^[A-Za-z]+ \\d{1,2}, \\d{4}$"

df = df.withColumn(
    "Date",
    to_date(
        when(col("date").isNull() | (col("date") == "") | 
             (regexp_extract(col("date"), date_pattern, 0) == ""), "January 1, 1970")
        .otherwise(col("date")),
        "MMMM d, yyyy"
    )
)


# Preprocess - Convert all text to lower case and split into words
df = df.withColumn("TitleWords", split(lower(col("title")), "\\W+")) \
       .withColumn("ContentWords", split(lower(col("content")), "\\W+"))

# Task 1: Count words in 'Title'
# Flatten title words and count frequencies, by total and by day
df_title_words = df.select("Date", "TitleWords").withColumn("Word", explode("TitleWords"))
word_count_total = df_title_words.groupBy("Word").count().orderBy(desc("count"))
word_count_per_day = df_title_words.groupBy("Date", "Word").count().orderBy("Date", desc("count"))


with open("Q1_word_count_total.txt", "w") as f:
    for row in word_count_total.collect():
        f.write(f"{row['Word']} {row['count']}\n")

with open("Q1_word_count_per_day.txt", "w") as f:
    for row in word_count_per_day.collect():
        f.write(f"{row['Date']} {row['Word']} {row['count']}\n")


# Task 2: Count words in 'Content'
df_content_words = df.select("Date", "ContentWords").withColumn("Word", explode("ContentWords"))
content_count_total = df_content_words.groupBy("Word").count().orderBy(desc("count"))
content_count_per_day = df_content_words.groupBy("Date", "Word").count().orderBy("Date", desc("count"))


with open("Q2_content_count_total.txt", "w") as f:
    for row in content_count_total.collect():
        f.write(f"{row['Word']} {row['count']}\n")

with open("Q2_content_count_per_day.txt", "w") as f:
    for row in content_count_per_day.collect():
        f.write(f"{row['Date']} {row['Word']} {row['count']}\n")

# Task 3: Calculate percentage of published articles per day, and by authors per day
articles_per_day = df.groupBy("Date").count().withColumnRenamed("count", "TotalArticlesPerDay")
total_articles = articles_per_day.selectExpr("sum(TotalArticlesPerDay) as Total").collect()[0]["Total"]
articles_per_day = articles_per_day.withColumn("Percentage", (col("TotalArticlesPerDay") / total_articles) * 100)

articles_by_author_per_day = df.groupBy("Date", "author").count().withColumnRenamed("count", "ArticlesByAuthor")
articles_by_author_per_day = articles_by_author_per_day.join(articles_per_day.select("Date", "TotalArticlesPerDay"), on="Date")
articles_by_author_per_day = articles_by_author_per_day.withColumn("Percentage", (col("ArticlesByAuthor") / col("TotalArticlesPerDay")) * 100).orderBy("Date", "author")

with open("Q3_articles_per_day.txt", "w") as f:
    for row in articles_per_day.collect():
        f.write(f"{row['Date']} {row['Percentage']:.2f}%\n")

with open("Q3_articles_by_author_per_day.txt", "w") as f:
    for row in articles_by_author_per_day.collect():
        f.write(f"{row['Date']} {row['author']} {row['Percentage']:.2f}%\n")

# Task 4: List records where "Space" occurs both in Title and Postexcerpt
df_filtered = df.filter(lower(col("title")).contains("space") & lower(col("postexcerpt")).contains("space"))


with open("Q4_filtered_records.txt", "w") as f:
    for row in df_filtered.collect():
        f.write(f"{row}\n")

# Stop SparkSession
spark.stop()
