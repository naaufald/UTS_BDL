import streamlit as st
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
        .stApp {
            background-color: white !important;
        }
        body, .stMarkdown, .stTextInput, .stButton, .stSelectbox, .stSlider, .stDataFrame, .stTable {
            color: black !important;
        }
        h1, h2, h3, h4, h5, h6, .stHeader, .stSubheader {
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Pemrosesan Beralur Big Data")
st.markdown("""
- Nama : Naufal Fadhlullah
- NIM : 20234920001""")

st.header("Flowchart")
st.image(image="https://raw.githubusercontent.com/naaufald/UTS_BDL/main/alir.png")

st.header("Data : dataset yang digunakan merupakan data penerbangan")
st.image("https://raw.githubusercontent.com/naaufald/UTS_BDL/main/data.png")
caption = "Sumber Data: https://www.tablab.app/csv/sample?"
st.markdown(f'<p style="color:black;">{caption}</p>', unsafe_allow_html=True)
st.markdown("""
Data yang digunakan merupakan data jadwal dan keterlambatan penerbangan yang terdiri dari 1 juta baris serta 7 kolom. dimana, kolom tersebut meliputi :

1. FL_DATE = tanggal penerbangan
2. DEP_DELAY = keterlambatan penerbangan
3. ARR_DELAY = keterlambatan kedatangan (dalam menit)
4. AIR_TIME = durasi penerbangan (menit)
5. DISTANCE = jarak yang ditempuh (dalam mil)
6. DEP_TIME = waktu keberangkatan format jam desimal
7. ARR_TIME = waktu kedatangan format jam desimal. """)

st.subheader("Data Exploration with Spark")
st.image("https://raw.githubusercontent.com/naaufald/UTS_BDL/main/desc.png")
kode_python ="""
```python
df.describe().show()"""
st.markdown(kode_python)

st.markdown("cek missing value")
st.image("https://raw.githubusercontent.com/naaufald/UTS_BDL/main/miss_val.png")
kode_python = """
```python
missing_values = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
missing_values.show()"""
st.markdown(kode_python)
st.markdown("tidak terdapat missing value di data ini")

st.markdown("distribusi nilai")
st.image("https://raw.githubusercontent.com/naaufald/UTS_BDL/main/dist_nilai.png")
kode_python = """
```python
numeric_columns = ["DEP_DELAY", "ARR_DELAY", "AIR_TIME", "DISTANCE"]
pdf = df.select(numeric_columns).toPandas()
plt.figure(figsize=(12, 6))
for i, col_name in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(pdf[col_name], kde=True, bins=50)
    plt.title(f"Distribusi {col_name}")

plt.tight_layout()
plt.show()"""
st.markdown(kode_python)

st.markdown("outliers")
st.image("https://raw.githubusercontent.com/naaufald/UTS_BDL/main/outlier.png")
kode_python = """
```python
plt.figure(figsize=(12, 6))
for i, col_name in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=pdf[col_name])
    plt.title(f"Outlier pada {col_name}")

plt.tight_layout()
plt.show()"""
st.markdown(kode_python)

st.markdown("feature engineering")
st.image("https://raw.githubusercontent.com/naaufald/UTS_BDL/main/feature.png")
kode_python = """
```python
from pyspark.sql.functions import col, when, round, year, month, dayofmonth, dayofweek, expr

# Ekstraksi fitur waktu dari FL_DATE
df = df.withColumn("YEAR", year(col("FL_DATE"))) \
       .withColumn("MONTH", month(col("FL_DATE"))) \
       .withColumn("DAY", dayofmonth(col("FL_DATE"))) \
       .withColumn("DAY_OF_WEEK", dayofweek(col("FL_DATE")))

# Konversi DEP_TIME dan ARR_TIME ke format jam (HH.MM)
df = df.withColumn("DEP_HOUR", round(col("DEP_TIME") / 100, 2)) \
       .withColumn("ARR_HOUR", round(col("ARR_TIME") / 100, 2))

# Hitung kecepatan rata-rata penerbangan (miles per minute)
df = df.withColumn("SPEED", round(col("DISTANCE") / col("AIR_TIME"), 2))

# Kategorisasi delay keberangkatan (>15 menit dianggap delay)
df = df.withColumn("DELAYED", when(col("DEP_DELAY") > 15, 1).otherwise(0))

# Menampilkan skema setelah feature engineering
df.printSchema()"""
st.markdown(kode_python)

st.markdown("show")
st.image("https://raw.githubusercontent.com/naaufald/UTS_BDL/main/hasil.png")
kode_python = """
```python
df.select("FL_DATE", "YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "DEP_HOUR", "ARR_HOUR", "SPEED", "DELAYED").show(5)"""
st.markdown(kode_python)

st.markdown("pemilihan fitur yang digunakan dilakukan untuk memahami fitur penting untuk dianalisis."
" di data ini fitur yang digunakan itu DEP_DELAY & ARR_DELAY buat menganalisis si penerbangan ini tuh sering terjadi keterlambatan atau ngga; "
"selanjutnya AIR_TIME & DISTANCE dipake buat analisis seberapa jauh dan lama si pesawat bisa terbang,"
" trs ada penambahan kecepatan rata-rata dengan menggunakan DISTANCE/AIR_TIME (durasi) buat tahu ada anomali ga sih di durasinya, "
"dan juga buat kategori delay yang dimana kalo DEP_DELAY nya > 15 menit bakal di detect 1 kalau terlambat, dan 0 kalo tepat waktu.")

st.subheader("Spark Analysis")
st.image("https://raw.githubusercontent.com/naaufald/UTS_BDL/main/avg_delay.png")
kode_python = """
```python
from pyspark.sql.functions import avg

# Rata-rata delay per bulan
df.groupBy("MONTH") \
  .agg(avg("DEP_DELAY").alias("avg_dep_delay"), 
       avg("ARR_DELAY").alias("avg_arr_delay")) \
  .orderBy("MONTH") \
  .show()"""
st.markdown(kode_python)


st.image("https://raw.githubusercontent.com/naaufald/UTS_BDL/main/avgdly_week.png")
kode_python = """
```python
# Rata-rata delay per hari dalam seminggu
df.groupBy("DAY_OF_WEEK") \
  .agg(avg("DEP_DELAY").alias("avg_dep_delay"), 
       avg("ARR_DELAY").alias("avg_arr_delay")) \
  .orderBy("DAY_OF_WEEK") \
  .show()"""
st.markdown(kode_python)

st.markdown(""" 1. pada tabel pertama, didapatkan hasil bahwa rata-rata keterlambatan keberangkatan pada bulan pertama (januari) sebesar 8.05 menit dan keterlambatan kedatangan ada di angka 5.62 menit. selanjutnya, di bulan berikutnya yaitu februari, terdapat keterlambatan keberangkatan yang lebih lebih tinggi yaitu 9.4 menit dan keterlambatan kedatangan di angka 7.4 menit. dimana, hal iini menunjukkan bahwa keterlambatan di bulan februari lebih besar atau lebih tinggi dibanding dengan keterlambatan yang ada di bulan januari.""")

st.markdown("""
2. pada tabel berikutnya, ini tuh nunjukkin keterlambatan dalam seminggu yang diisyaratkan 1 sebagai senin, 2 sebagaii selasa, dan seterusnya hingga 7 sebagai minggu.
pada senin, rata-rata keterlambatan keberangkatan ada di angka 10.3 menit dan keterlambatan kedatangan ada di angka 7.9 menit pada selasa, rata-rata keterlambatan keberangkatan ada di angka 12 menit dan keterlambatan kedatangan ada di angka 10.2 menit pada rabu, rata-rata keterlambatan keberangkatan ada di angka 6.5 menit dan keterlambatan kedatangan ada di angka 3.88~4 menit pada kamis, rata-rata keterlambatan keberangkatan ada di angka 6.09 menit dan keterlambatan kedatangan ada di angka 3.65 menit pada jumat, rata-rata keterlambatan keberangkatan ada di angka 6.44 menit dan keterlambatan kedatangan ada di angka 4.12 menit pada sabtu, rata-rata keterlambatan keberangkatan ada di angka 10.8 menit dan keterlambatan kedatangan ada di angka 9.2 menit pada minggu, rata-rata keterlambatan keberangkatan ada di angka 7.7 menit dan keterlambatan kedatangan ada di angka 5.1 menit
dari 7 hari, keterlambatan keberangkatan dan kedatangan tertinggi terjadi pada hari selasa dan keterlambatan terendah ada di kamis""")

st.image("https://raw.githubusercontent.com/naaufald/UTS_BDL/main/corr.png")
kode_python = """
```python
from pyspark.sql.functions import corr

# Korelasi antara keterlambatan keberangkatan dan kedatangan
df.select(corr("DEP_DELAY", "ARR_DELAY").alias("correlation")).show()

# Korelasi antara durasi penerbangan dan jaraknya
df.select(corr("AIR_TIME", "DISTANCE").alias("correlation")).show()"""
st.markdown(kode_python)
st.markdown("""terdapat korelasi yang tinggi (hampir mendekati 1) antara DEP_DELAY dan ARR_DELAY dan juga AIR_TIME dan DISTANCE.""")

st.image("https://raw.githubusercontent.com/naaufald/UTS_BDL/main/delay_ekstrem.png")
kode_python = """
```python
from pyspark.sql.functions import col
# Cek delay ekstrem (> 300 menit)
df.filter(col("DEP_DELAY") > 300).show()"""
st.markdown(kode_python)
st.markdown("""semua penerbangan yang tertera dalam tabel mengalami keterlambatan, dilihat dari tabel delayed yang menunjukkan angka 1. terlihat juga pada tanggal 14-01-2006 terjadi keterlambatan keberangkatan selama 949 menit atau sekitar 15.8 jam dan keterlambatan kedatangan selama 923menit atau 15.3 jam dari jadwal yang seharusnya.
kevariasian dari DEP_HOUR dan ARR_HOUR menandakan adanya jadwal penerbangan yang dilakukan adalah tiap hari.""")

st.image("https://raw.githubusercontent.com/naaufald/UTS_BDL/main/outlier_speed.png")
kode_python = """
```python
# Cek outlier pada kecepatan penerbangan
df.orderBy(col("SPEED").desc()).show(10)"""
st.markdown(kode_python)
st.markdown("""beberapa tanggal penerbangan memiliki keterlambatan yang negatif, artinya penerbangan ini berangkat lebih awal, seperti pada tanggal 29 januari 2006, penerbangan berangkat lebih awal 9 menit dan datang 24 menit lebih awal juga.
seperti halnya yang terjadi tanggal 15-01-2006, penerbangan pada tanggal ini terjadi keterlambatan penerbangan 65 menit, tetapi datang lebih awal 42 menit.
untuk setiap penerbangan yang mengalami keterlambatan > 15 menit, akan di detect sebagai 1 di kolom delayed, dan apabila < 15 menit, akan di detect sebagai 0.""")

st.subheader("Discuss dan Kesimpulan")
st.markdown("""
kelebihan = kecepatan dalam pemrosesan data yang tinggi, fleksibilitas dalam menangani berbagai jenis data.

kekurangan = butuh memori yang besar, konfigurasi yang cukup kompleks

interpretasi hasil = dari analisis yang telah dilakukan, kita bisa mengetahui adanya pola keterlambatan berdasarkan waktu seperti adanya keterlambatan di bulan tertentu yang tinggi, dan juga keterlambatan di hari tertentu. selain itu, kita bisa mengetahui korelasi antar variabel, seperti contoh, variabel DEP_DELAY dan ARR_DELAYberkorelasi, artinyakalau sebuah penerbangan terlambat berangkat, kemungkinan besar akan terjadi pula keterlambatan ketibaan.""")

st.subheader("Kesimpulan")
st.markdown("""dari analisis yang telah dilakukan, ditemukan bahwa terdapat bulan-bulan dan hari-hari tertentu sebuah penerbangan mengalami keterlambatan. dapat mengetahui korelasi yang tinggi antara keterlambatan keberangkatan dengan keterlambatan kedatangan yang mengindikasikan bahwa kemungkinan besar suatu penerbangan akan mengalami terlambat tiba jika penerbangan tersebut terlambat berangkat. selain itu, menemukan adanya sebuah penerbangan yang mengalami keterlambatan yang sangat tinggi (>5jam) atau juga penerbangan yang memiliki kecepatan tidak seperti biasanya.""")
