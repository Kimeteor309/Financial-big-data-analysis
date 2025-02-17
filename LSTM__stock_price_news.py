
#%%
"""
1. 股價資料(yfinaace api)
"""
# 安裝yahoo finance套件
!pip install yfinance

# 整理台灣50成分股的股票代號 + 台股加權指數
stock_ids = [
    '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2881.TW', '2382.TW', '2303.TW',
    '2882.TW', '2891.TW', '3711.TW', '2412.TW', '2886.TW', '2884.TW', '1216.TW',
    '2357.TW', '2885.TW', '2892.TW', '3034.TW', '2890.TW', '2327.TW', '5880.TW',
    '2345.TW', '3231.TW', '2002.TW', '2880.TW', '3008.TW', '2883.TW', '1303.TW',
    '4938.TW', '2207.TW', '2887.TW', '2379.TW', '1101.TW', '2603.TW', '2301.TW',
    '1301.TW', '5871.TW', '3037.TW', '3045.TW', '2912.TW', '3017.TW', '6446.TW',
    '4904.TW', '3661.TW', '6669.TW', '1326.TW', '5876.TW', '2395.TW', '1590.TW', '6505.TW', '^TWII'
]

# 匯入必要的套件
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 定義時間範圍
start_date = '2020-01-01' 
end_date = '2024-12-31'

# 建立空的 DataFrame
taiwan50_data = pd.DataFrame()

# 逐一下載每支股票的資料並整合
scalers = {}  # 用於儲存每支股票的標準化模型
for stock_id in stock_ids:
    print(f"Downloading data for {stock_id}...")
    stock_data = yf.download(stock_id, start=start_date, end=end_date)

    # 保留收盤價 (Close)
    stock_data = stock_data[['Close']].copy()

    # 標準化
    scaler = StandardScaler()
    stock_data['Close'] = scaler.fit_transform(stock_data[['Close']])
    scalers[stock_id] = scaler  # 儲存標準化模型以便後續使用

    # 將欄位名稱改為股票代號
    stock_data.rename(columns={'Close': f'Close({stock_id})'}, inplace=True)
    # stock_data.rename(columns={'Close': stock_id}, inplace=True)

    # 合併到總表，按日期對齊
    if taiwan50_data.empty:
        taiwan50_data = stock_data
    else:
        taiwan50_data = taiwan50_data.join(stock_data, how='outer')

# 重置索引為日期，確保整潔
taiwan50_data.index.name = 'Date'

# 檢視資料
print(taiwan50_data.head())

# 將資料儲存為 CSV 檔案
taiwan50_data.to_csv('taiwan50_close_prices_normalized.csv')




#%%
"""
2. 新聞資料(鉅亨網api)
"""
import pandas as pd
import datetime as dt
import numpy as np
import requests
from time import sleep
from tqdm import tqdm

# 定義目標股票代號
stock_ids = [
    '1101', '1216', '1301', '1303', '1326', '1590', '2002', '2207',
    '2301', '2303', '2308', '2317', '2327', '2330', '2345', '2357',
    '2379', '2382', '2395', '2412', '2454', '2603', '2880', '2881',
    '2882', '2883', '2884', '2885', '2886', '2887', '2890', '2891',
    '2892', '2912', '3008', '3017', '3034', '3037', '3045', '3231',
    '3661', '3711', '4904', '4938', '5871', '5876', '5880', '6446',
    '6505', '6669', '^TWII'
]

# API 參數
cnyes_api = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock"
data_cols = ["publishAt", "title", "content", "market"]  # 要提取的資料欄位

def get_news_data(start_date, end_date):
    """
    從 API 獲取指定時間範圍內的新聞資料。
    """
    params = {
        "startAt": int(dt.datetime.strptime(start_date, "%Y-%m-%d").timestamp()),
        "endAt": int(dt.datetime.strptime(end_date, "%Y-%m-%d").timestamp()),
        "limit": 100  # 每頁最多顯示 100 篇新聞
    }

    # 獲取總頁數
    data = requests.get(url=cnyes_api, params=params).json()
    page_num = data["items"]["last_page"]

    # 儲存所有新聞的列表
    article_list = []

    # 遍歷所有頁面
    for i in tqdm(range(1, page_num + 1)):
        params["page"] = i
        data = requests.get(url=cnyes_api, params=params).json()["items"]["data"]

        # 篩選出目標欄位並添加到文章列表
        for article in data:
            # 只保留與目標股票相關的新聞
            if "market" in article and any(stock["code"] in stock_ids for stock in article["market"]):
                filtered_article = {k: article[k] for k in data_cols if k in article}
                # 添加符合條件的文章
                article_list.append(filtered_article)

        sleep(np.random.randint(3, 5)) # 避免請求過於頻繁

    return article_list  # 返回收集到的新聞資料列表

# 將時間範圍細分為按月的區間
def generate_monthly_ranges(start_date, end_date):
    start = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end = dt.datetime.strptime(end_date, "%Y-%m-%d")
    ranges = []

    while start <= end:
        month_end = (start + pd.offsets.MonthEnd(0)).to_pydatetime()
        if month_end > end:
            month_end = end
        ranges.append((start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))
        start = month_end + dt.timedelta(days=1)

    return ranges


# 定義起始日期和結束日期
start_date = '2020-01-01'
end_date = '2024-12-31'

# 生成每月的日期範圍
monthly_ranges = generate_monthly_ranges(start_date, end_date)

# 收集所有日期範圍內的新聞資料
all_articles = []
for start, end in monthly_ranges:
    articles = get_news_data(start, end)  # 呼叫函數獲取資料
    all_articles.extend(articles)  # 將獲取到的新聞資料添加到 all_articles 列表中


# 將文章分成每個股票的 DataFrame
stock_news = {stock: [] for stock in stock_ids}
for article in all_articles:
    for stock in article["market"]:
        if stock["code"] in stock_ids:
            stock_news[stock["code"]].append(article)


# 將結果轉為 DataFrame 並儲存成 CSV 檔案
import dateutil.tz as tz # 導入必要的模組
import re
!pip install html # 安裝 html 套件
import html

for stock_id, articles in stock_news.items():
    if articles:  # 確保有內容
        df = pd.DataFrame(articles)
        # 刪除 'market' 欄位
        df = df.drop('market', axis=1)
        # 添加 'date' 欄位
        df["date"] = df["publishAt"].apply(lambda x: int(dt.datetime.fromtimestamp(x, tz=tz.gettz("Asia/Taipei")).strftime("%Y%m%d")))
        # 將 'date' 設成索引
        df = df.set_index('date')
        df = df.sort_index()
        # 刪除 'publishAt' 欄位
        df = df.drop('publishAt', axis=1)
        # 套用 html 標籤清除 regex 語法
        pattern = "<.*?>|\n|&[a-z0-9]+;|http\S+"
        df["content"] = df["content"].apply(lambda x: re.sub(pattern, "", html.unescape(x)).strip())
        # 儲存成 CSV 檔案
        df.to_csv(f"{stock_id}_news.csv", index=True, encoding='utf-8')
        print(f"股票 {stock_id} 的新聞資料已儲存到 {stock_id}_news.csv")
    else:
        print(f"股票 {stock_id} 無任何新聞資料返回")

print("所有股票的新聞資料已儲存完成")


#%%
"""
載入先前存好的 {stock_id}_news.csv
"""
import pandas as pd

# 定義目標股票代號
stock_ids = [
    '1101', '1216', '1301', '1303', '1326', '1590', '2002', '2207',
    '2301', '2303', '2308', '2317', '2327', '2330', '2345', '2357',
    '2379', '2382', '2395', '2412', '2454', '2603', '2880', '2881',
    '2882', '2883', '2884', '2885', '2886', '2887', '2890', '2891',
    '2892', '2912', '3008', '3017', '3034', '3037', '3045', '3231',
    '3661', '3711', '4904', '4938', '5871', '5876', '5880', '6446',
    '6505', '6669'
]

# 建立一個空字典來儲存 DataFrame
stock_news = {}

# 載入先前存好的消息面csv
for stock_id in stock_ids:
    stock_news[stock_id] = pd.read_csv(f"data/news/original/{stock_id}_news.csv", index_col=0)
    print(f"{stock_id}_news.csv 載入完成")

# 查看 2330
stock_news["2330"]


#%%
"""
使用 finbert-chinese 進行新聞情緒分析
"""
# 載入 finbert-chinese
from transformers import TextClassificationPipeline
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import BertTokenizerFast
model_path="yiyanghkust/finbert-tone-chinese"
new_model = AutoModelForSequenceClassification.from_pretrained(model_path,output_attentions=True)
tokenizer = BertTokenizerFast.from_pretrained(model_path)
PipelineInterface = TextClassificationPipeline(model=new_model, tokenizer=tokenizer, return_all_scores=True)

# 測試
label = PipelineInterface("此外宁德时代上半年实现出口约2GWh，同比增加200%+。")
print(label)


#%%
"""
將新聞內容截斷至finbert可接受的最大長度(512 token)
"""
# 定義截斷函式
def truncate_texts(texts, tokenizer, max_length=512):
    """將多條文本截斷至模型可接受的最大長度。"""
    truncated_texts = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_length - 2:  # 考慮 [CLS] 和 [SEP]
            tokens = tokens[:max_length - 2]
        truncated_texts.append(tokenizer.convert_tokens_to_string(tokens))
    return truncated_texts


# 批量截斷文本 輸出csv
for stock_id in stock_news:
    df = stock_news[stock_id]
    df["content"] = truncate_texts(df["content"], tokenizer)
    stock_news[stock_id] = df
    df.to_csv(f"{stock_id}_news_truncated.csv", index=True, encoding='utf-8')
    print(f"{stock_id}_news_truncated.csv 輸出完成")



#%%
"""
載入先前存好的 {stock_id}_news_truncated.csv
"""
import pandas as pd
# 定義目標股票代號
stock_ids = [
    '1101', '1216', '1301', '1303', '1326', '1590', '2002', '2207',
    '2301', '2303', '2308', '2317', '2327', '2330', '2345', '2357',
    '2379', '2382', '2395', '2412', '2454', '2603', '2880', '2881',
    '2882', '2883', '2884', '2885', '2886', '2887', '2890', '2891',
    '2892', '2912', '3008', '3017', '3034', '3037', '3045', '3231',
    '3661', '3711', '4904', '4938', '5871', '5876', '5880', '6446',
    '6505', '6669'
]

# 建立一個空字典來儲存 DataFrame
stock_news = {}

# 載入 stock_news_truncated.csv
for stock_id in stock_ids:
    stock_news[stock_id] = pd.read_csv(f"./data/news/truncated/{stock_id}_news_truncated.csv", index_col=0)
    print(f"{stock_id}_news_truncated.csv 載入完成")

# 顯示完整內容(調整 Pandas 的輸出設定)
pd.set_option('display.max_colwidth', None)  # 設定列寬為無限制
pd.set_option('display.max_rows', 20)       # 設定最大顯示列數為 20（依需求調整）

# 顯示資料
stock_news["2330"]



#%%
"""
使用 finbert-chinese 進行新聞情緒分析
"""
# 批量進行情緒分析 輸出 csv
for stock_id in stock_news:
    df = stock_news[stock_id]
    # PipelineInterface 接收整個文本列表進行批量分析
    df["sentiment"] = PipelineInterface(list(df["content"]))
    stock_news[stock_id] = df
    print(f"{stock_id} 分析完成")
    df.to_csv(f"{stock_id}_news_sentiment.csv", index=True, encoding='utf-8')
    print(f"{stock_id}_news_sentiment.csv 輸出完成")



#%%
"""
載入先前存好的 {stock_id}_news_sentiment.csv
"""
import pandas as pd
# 定義目標股票代號
stock_ids = [
    '1101', '1216', '1301', '1303', '1326', '1590', '2002', '2207',
    '2301', '2303', '2308', '2317', '2327', '2330', '2345', '2357',
    '2379', '2382', '2395', '2412', '2454', '2603', '2880', '2881',
    '2882', '2883', '2884', '2885', '2886', '2887', '2890', '2891',
    '2892', '2912', '3008', '3017', '3034', '3037', '3045', '3231',
    '3661', '3711', '4904', '4938', '5871', '5876', '5880', '6446',
    '6505', '6669'
]

# 建立一個空字典來儲存 DataFrame
stock_news = {}

# 載入 stock_news_sentiment.csv
for stock_id in stock_ids:
    stock_news[stock_id] = pd.read_csv(f"./data/news/sentiment/{stock_id}_news_sentiment.csv", index_col=0)
    print(f"{stock_id}_news_sentiment.csv 載入完成")

# 顯示完整內容(調整 Pandas 的輸出設定)
# pd.set_option('display.max_colwidth', None)  # 設定列寬為無限制
# pd.set_option('display.max_rows', 20)       # 設定最大顯示列數為 20（依需求調整）

# 顯示資料
stock_news["2330"]


#%%
"""
整合相同股票、同一天的情緒分數，並計算每日情緒值（範圍：-1 到 1，沒有新聞則補0）
"""

from collections import defaultdict
import pandas as pd

def aggregate_sentiments_with_daily_score(stock_news):
    """
    整合相同股票、同一天的情緒分數，並計算每日情緒值（範圍：-1 到 1）。
    """
    from collections import defaultdict
    
    aggregated_results = defaultdict(dict)  # 用於存儲整合後的結果

    for stock_id, news_list in stock_news.items():
        daily_sentiments = defaultdict(list)
        
        # 確保 DataFrame 的索引重置，避免意外問題
        news_list.reset_index(inplace=True)

        # 將相同日期的新聞情緒分組
        for _, row in news_list.iterrows():  # 使用 iterrows 逐行處理
            date = row["date"]  # 單一日期值
            sentiment = eval(row["sentiment"])  # 假設 sentiment 是字串需轉回字典
            
            # 按日期收集情緒分數
            for entry in sentiment:
                daily_sentiments[date].append(entry)

        # 計算每種情緒的平均分數和每日情緒值
        for date, sentiments in daily_sentiments.items():
            total_counts = len(sentiments) // 3  # 每個新聞有三種情緒分數
            sentiment_totals = {"Neutral": 0, "Positive": 0, "Negative": 0}

            # 加總每種情緒的分數
            for entry in sentiments:
                sentiment_totals[entry['label']] += entry['score']

            # 平均分數
            averaged_scores = {
                label: sentiment_totals[label] / total_counts
                for label in sentiment_totals
            }

            # 計算每日情緒值：Positive - Negative
            daily_score = averaged_scores["Positive"] - averaged_scores["Negative"]

            # 保存結果
            aggregated_results[stock_id][date] = {
                "Neutral": averaged_scores["Neutral"],
                "Positive": averaged_scores["Positive"],
                "Negative": averaged_scores["Negative"],
                "Daily Sentiment": daily_score
            }

    return aggregated_results

# 執行整合
result = aggregate_sentiments_with_daily_score(stock_news)

# 將結果轉為 DataFrame 以方便查看
stock_sentiment = pd.DataFrame.from_dict(
    {(stock_id, date): sentiments for stock_id, dates in result.items() for date, sentiments in dates.items()},
    orient="index"
)
stock_sentiment.index = pd.MultiIndex.from_tuples(stock_sentiment.index, names=["Stock ID", "Date"])
print(stock_sentiment)

# 輸出csv
stock_sentiment.to_csv("stock_sentiment.csv")




#%%
"""
3. 前處理
"""
"""
載入 股價標準化後的資料
"""
# 讀取股票標準化後的資料
stocks_df = pd.read_csv('./data/taiwan50_close_prices_normalized.csv', index_col=0)
# 將索引名稱設定為“日期”
stocks_df.index.name = 'Date'
# 去除前兩個row
stocks_df = stocks_df[2:]
# 轉換型態float32
stocks_df = stocks_df.astype('float32')
# 日期格式轉為 datetime
stocks_df.index = pd.to_datetime(stocks_df.index)

stocks_df['Date'] = stocks_df.index
# 去除索引
stocks_df = stocks_df.reset_index(drop=True)

stocks_df


#%%
"""載入 股票新聞情緒資料

"""
sentiment_df = pd.read_csv('./data/stock_sentiment.csv', index_col=0)

sentiment_df = sentiment_df.reset_index()  # 將原本的索引變成欄位
sentiment_df = sentiment_df.pivot(index='Date', columns='Stock ID', values='Daily Sentiment')

# 調整欄位名稱格式為 "Daily Sentiment(股票代號)"
sentiment_df.columns = [f"Daily Sentiment({col})" for col in sentiment_df.columns]

# 將索引轉換為日期格式
sentiment_df.index = pd.to_datetime(sentiment_df.index, format='%Y%m%d')  # 指定日期格式
sentiment_df['Date'] = sentiment_df.index

# 去除索引
sentiment_df = sentiment_df.reset_index(drop=True)

sentiment_df


#%%
"""整合 股票標準化後股價資料 與 股票新聞情緒

"""
import pandas as pd

# 確保 full_dates 與 sentiment_df['Date'] 資料類型一致
full_dates = pd.DataFrame({'Date': pd.to_datetime(stocks_df['Date'])})

# 確保 sentiment_df 的日期格式正確
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

# 找出多餘情緒值（不在有效交易日中的日期）
extra_sentiment = sentiment_df[~sentiment_df['Date'].isin(stocks_df['Date'])]

# 定義有效交易日
valid_dates = pd.to_datetime(stocks_df['Date']).sort_values().unique()

# 將多餘情緒移動到最近有效交易日
def shift_sentiment(df, valid_dates):
    df['Date'] = df['Date'].apply(lambda x: valid_dates[valid_dates >= x].min())
    return df

# 將多餘情緒值移動到最近的有效交易日
shifted_sentiment = shift_sentiment(extra_sentiment, valid_dates)

# 合併原始情緒數據與移動後的情緒數據
sentiment_df = pd.concat([sentiment_df, shifted_sentiment], ignore_index=True)

# 按日期聚合情緒數據（保留每支股票的獨立欄位，取平均值）
sentiment_df = sentiment_df.groupby('Date', as_index=False).mean()

# 確保 sentiment_df 日期與 full_dates 對齊，並補全缺失值為 0
sentiment_df = pd.merge(full_dates, sentiment_df, on='Date', how='left')
sentiment_df.fillna(0, inplace=True)  # 將所有缺失值補為 0

# 將 sentiment_df 與 stocks_df 合併
merged_df = pd.merge(stocks_df, sentiment_df, on='Date', how='left')

# 結果
print(merged_df.head())

merged_df.index = pd.to_datetime(merged_df['Date'])
merged_df = merged_df.drop(columns=['Date'])
stock_price_news_data = merged_df.sort_index()  # 確保資料按日期排序
stock_price_news_data

# 儲存csv
stock_price_news_data.to_csv("./data/stock_price_news_data.csv")


#%%
"""4. 準備訓練資料

"""
"""
準備多重輸入的資料
"""
# 讀取 stock_price_news_data csv
stock_price_news_data = pd.read_csv("./data/stock_price_news_data.csv")
# 去除date欄位
stock_price_news_data = stock_price_news_data.drop(columns=["Date"])
stock_price_news_data

# 定義目標股票代號
stock_ids = [
    '1101', '1216', '1301', '1303', '1326', '1590', '2002', '2207',
    '2301', '2303', '2308', '2317', '2327', '2330', '2345', '2357',
    '2379', '2382', '2395', '2412', '2454', '2603', '2880', '2881',
    '2882', '2883', '2884', '2885', '2886', '2887', '2890', '2891',
    '2892', '2912', '3008', '3017', '3034', '3037', '3045', '3231',
    '3661', '3711', '4904', '4938', '5871', '5876', '5880', '6446',
    '6505', '6669'
]

stock_ids = [f"Close({stock_id}.TW)" for stock_id in [
    '1101', '1216', '1301', '1303', '1326', '1590', '2002', '2207',
    '2301', '2303', '2308', '2317', '2327', '2330', '2345', '2357',
    '2379', '2382', '2395', '2412', '2454', '2603', '2880', '2881',
    '2882', '2883', '2884', '2885', '2886', '2887', '2890', '2891',
    '2892', '2912', '3008', '3017', '3034', '3037', '3045', '3231',
    '3661', '3711', '4904', '4938', '5871', '5876', '5880', '6446',
    '6505', '6669'
]]

import pandas as pd
import numpy as np

# 定義參數
window_size = 20

# 計算所有股票之間的相關係數
correlation_matrix = stock_price_news_data.corr()

# print("Stock IDs:", stock_ids)
# print("Correlation Matrix Columns:", correlation_matrix.columns)

# 提取正相關和負相關的股票清單
positive_corr_dict = {}
negative_corr_dict = {}

# 準備多重輸入的數據
X_target, X_index, X_positive, X_negative, X_sentiment, y = [], [], [], [], [], []

# 處理資料
for i in range(window_size, len(stock_price_news_data)):  # 從 window_size 開始，因為需要足夠的滑動視窗
    for target_stock in stock_ids:
        
        # 取得正相關和負相關的股票
        correlations = correlation_matrix[target_stock].drop(target_stock)  # 排除自身的相關係數
        positive_corr_dict[target_stock] = correlations.nlargest(10).index  # 正相關前10
        negative_corr_dict[target_stock] = correlations.nsmallest(10).index  # 負相關前10

        positive_corr_stocks = positive_corr_dict[target_stock]
        negative_corr_stocks = negative_corr_dict[target_stock]

        # 滑動視窗特徵
        target_window = stock_price_news_data[target_stock].iloc[i-window_size:i].values
        index_window = stock_price_news_data["Close(^TWII)"].iloc[i-window_size:i].values  # 台股加權指數
        positive_window = stock_price_news_data[positive_corr_stocks].iloc[i-window_size:i].values
        negative_window = stock_price_news_data[negative_corr_stocks].iloc[i-window_size:i].values
        # 處理情緒數據
        sentiment_column = f"Daily Sentiment({target_stock.split('(')[1].split('.')[0]})"
        if sentiment_column in stock_price_news_data.columns:
            sentiment_window = stock_price_news_data[sentiment_column].iloc[i - window_size:i].values
        else:
            sentiment_window = np.zeros(window_size) # 若無情緒資料，補 0

        # 將資料加入輸入
        X_target.append(target_window)
        X_index.append(index_window)
        X_positive.append(positive_window)
        X_negative.append(negative_window)
        X_sentiment.append(sentiment_window)
        y.append(stock_price_news_data[target_stock].iloc[i])  # 預測目標是當天的股價

# 轉換為 NumPy 陣列
X_target = np.array(X_target)
X_index = np.array(X_index)
X_positive = np.array(X_positive)
X_negative = np.array(X_negative)
X_sentiment = np.array(X_sentiment)
y = np.array(y)

# 輸出資料形狀
print("X_target shape:", X_target.shape)
print("X_index shape:", X_index.shape)
print("X_positive shape:", X_positive.shape)
print("X_negative shape:", X_negative.shape)
print("X_sentiment shape:", X_sentiment.shape)
print("y shape:", y.shape)


#%%
""" 資料集以 8:2 的比率切割訓練及測試集

"""
train_size = int(len(X_sentiment) * 0.8)
test_size = len(X_sentiment) - train_size  # 測試集大小為剩餘部分

X_train_target, X_test_target = X_target[:train_size], X_target[train_size:]
X_train_index, X_test_index = X_index[:train_size], X_index[train_size:]
X_train_positive, X_test_positive = X_positive[:train_size], X_positive[train_size:]
X_train_negative, X_test_negative = X_negative[:train_size], X_negative[train_size:]
X_train_sentiment, X_test_sentiment = X_sentiment[:train_size], X_sentiment[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("X_train_target shape:", X_train_target.shape)
print("X_test_target shape:", X_test_target.shape)
print("- - - - - - - - -")
print("X_train_index shape:", X_train_index.shape)
print("X_test_index shape:", X_test_index.shape)
print("- - - - - - - - -")
print("X_train_positive shape:", X_train_positive.shape)
print("X_test_positive shape:", X_test_positive.shape)
print("- - - - - - - - -")
print("X_train_negative shape:", X_train_negative.shape)
print("X_test_negative shape:", X_test_negative.shape)
print("- - - - - - - - -")
print("X_train_sentiment shape:", X_train_sentiment.shape)
print("X_test_sentiment shape:", X_test_sentiment.shape)
print("- - - - - - - - -")
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


#%%
"""為了符合 LSTM 的輸入，重塑輸入為 (samples, time steps, features)

"""
X_train = {
    "target_stock": X_train_target.reshape(-1, window_size, 1).astype(np.float32),
    "benchmark": X_train_index.reshape(-1, window_size, 1).astype(np.float32),
    "positive_stocks": X_train_positive.astype(np.float32),
    "negative_stocks": X_train_negative.astype(np.float32),
    "sentiment": X_train_sentiment.reshape(-1, window_size, 1).astype(np.float32),
}
X_test = {
    "target_stock": X_test_target.reshape(-1, window_size, 1).astype(np.float32),
    "benchmark": X_test_index.reshape(-1, window_size, 1).astype(np.float32),
    "positive_stocks": X_test_positive.astype(np.float32),
    "negative_stocks": X_test_negative.astype(np.float32),
    "sentiment": X_test_sentiment.reshape(-1, window_size, 1).astype(np.float32),
}

print("X_train_target shape:", X_train["target_stock"].shape)
print("X_train_index shape:", X_train["benchmark"].shape)
print("X_train_positive shape:", X_train["positive_stocks"].shape)
print("X_train_negative shape:", X_train["negative_stocks"].shape)
print("X_train_sentiment shape:", X_train["sentiment"].shape)
print("- - - - - - - - -")

print("X_test_target shape:", X_test["target_stock"].shape)
print("X_test_index shape:", X_test["benchmark"].shape)
print("X_test_positive shape:", X_test["positive_stocks"].shape)
print("X_test_negative shape:", X_test["negative_stocks"].shape)
print("X_test_sentiment shape:", X_test["sentiment"].shape)
print("- - - - - - - - -")

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



#%%
"""5. 建立LSTM模型

"""
"""
建立模型架構
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Flatten, Dropout

# 定義輸入層shape (窗口大小，特徵數)
input_target = Input(shape=(window_size, 1), name="target_input")
input_index = Input(shape=(window_size, 1), name="index_input")
input_positive = Input(shape=(window_size, 10), name="positive_input")
input_negative = Input(shape=(window_size, 10), name="negative_input")
input_sentiment = Input(shape=(window_size, 1), name="sentiment_input")

# LSTM 層
h = 64
lstm_target = LSTM(h, name="lstm_target")(input_target)
lstm_index = LSTM(h, name="lstm_index")(input_index)
lstm_positive = LSTM(h, name="lstm_positive")(input_positive)
lstm_negative = LSTM(h, name="lstm_negative")(input_negative)
lstm_sentiment = LSTM(h, name="lstm_sentiment")(input_sentiment)

# 連接 所有LSTM層 輸出
# concatenated = Concatenate(name="concat_features")([lstm_target, lstm_index])
# concatenated = Concatenate(name="concat_features")([lstm_target, lstm_positive, lstm_negative])
# concatenated = Concatenate(name="concat_features")([lstm_target, lstm_sentiment])
concatenated = Concatenate(name="concat_features")([lstm_target, lstm_index, lstm_positive, lstm_negative, lstm_sentiment])

# 展平串聯輸出
flatten_concatenated = Flatten(name="flatten_concatenated")(concatenated)

# 全連接層 和 dropout
# fc1 = Dense(64, activation="relu", name="fc1")(lstm_target)  # 全連接層 (64 個神經元)
fc1 = Dense(64, activation="relu", name="fc1")(flatten_concatenated)  # 全連接層 (64 個神經元)
dropout1 = Dropout(0.2, name="dropout1")(fc1)  # Dropout 層 (0.2) 隨機關閉 20% 神經元
fc2 = Dense(32, activation="relu", name="fc2")(dropout1)  # 全連接層 (32 個神經元)
dropout2 = Dropout(0.2, name="dropout2")(fc2)  # Dropout 層 (0.2) 隨機關閉 20% 神經元

# 輸出層
output = Dense(1, name="output")(dropout2)  # 輸出層 (1 個神經元)

# 建立模型
# model_target = Model(inputs=[input_target], outputs=output)
# model_target_index = Model(inputs=[input_target, input_index], outputs=output)
# model_target_corr = Model(inputs=[input_target, input_positive, input_negative], outputs=output)
# model_target_news = Model(inputs=[input_target, input_sentiment], outputs=output)
model_target_index_corr_news = Model(inputs=[input_target, input_index, input_positive, input_negative, input_sentiment], outputs=output)

# 編譯模型
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# model_target.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
# model_target_index.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
# model_target_corr.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
# model_target_news.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
model_target_index_corr_news.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

# 顯示模型結構
# model_target.summary()
# model_target_index.summary()
# model_target_corr.summary()
# model_target_news.summary()
model_target_index_corr_news.summary()



#%%
"""LSTM 訓練模型

"""
# 早停機制
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
       monitor='val_loss',  # 監控驗證集的損失
       patience=5,          # 容忍的 epoch 數，若連續 5 個 epoch 驗證集損失沒有改善，則停止訓練
       restore_best_weights=True  # 恢復最佳模型的權重
   )

# 訓練
# model_target.fit(
#        [X_train['target_stock']],
#        y_train,
#        batch_size=32,
#        epochs=100,
#        validation_split=0.2,  # 驗證集比例
#        callbacks=[early_stopping]  # 加入 EarlyStopping 回調函數
#    )

# model_target_index.fit(
#        [X_train['target_stock'], X_train["benchmark"]],
#        y_train,
#        batch_size=32,
#        epochs=100,
#        validation_split=0.2,  # 驗證集比例
#        callbacks=[early_stopping]  # 加入 EarlyStopping 回調函數
#    )

# model_target_corr.fit(
#        [X_train['target_stock'], X_train["positive_stocks"], X_train["negative_stocks"]],
#        y_train,
#        batch_size=32,
#        epochs=100,
#        validation_split=0.2,  # 驗證集比例
#        callbacks=[early_stopping]  # 加入 EarlyStopping 回調函數
#    )

# model_target_news.fit(
#        [X_train['target_stock'], X_train["sentiment"]],
#        y_train,
#        batch_size=32,
#        epochs=100,
#        validation_split=0.2,  # 驗證集比例
#        callbacks=[early_stopping]  # 加入 EarlyStopping 回調函數
#    )

model_target_index_corr_news.fit(
       [X_train['target_stock'], X_train["benchmark"], X_train["positive_stocks"], X_train["negative_stocks"], X_train["sentiment"]],
       y_train,
       batch_size=32,
       epochs=100,
       validation_split=0.2,  # 驗證集比例
       callbacks=[early_stopping]  # 加入 EarlyStopping 回調函數
   )


#%%
"""LSTM 評估模型

"""
import matplotlib.pyplot as plt
import pandas as pd

# 評估測試集
# mse, mae = model_target.evaluate(
#     [X_test['target_stock']],
#     y_test
# )
# mse, mae = model_target_index.evaluate(
#     [X_test['target_stock'], X_test["benchmark"]],
#     y_test
# )
# mse, mae = model_target_corr.evaluate(
#     [X_test['target_stock'], X_test["positive_stocks"], X_test["negative_stocks"]],
#     y_test
# )
# mse, mae = model_target_news.evaluate(
#     [X_test['target_stock'],  X_test["sentiment"]],
#     y_test
# )
mse, mae = model_target_index_corr_news.evaluate(
    [X_test['target_stock'], X_test["benchmark"], X_test["positive_stocks"], X_test["negative_stocks"], X_test["sentiment"]],
    y_test
)
print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)  # 也印出 MAE



# %%
""" 6. 測試期間漲最多和跌最多的股票

"""
# 
# 匯入必要的套件
import yfinance as yf
import pandas as pd

# 定義目標股票代號
stock_ids = [
    '1101.TW', '1216.TW', '1301.TW', '1303.TW', '1326.TW', '1590.TW', '2002.TW', '2207.TW',
    '2301.TW', '2303.TW', '2308.TW', '2317.TW', '2327.TW', '2330.TW', '2345.TW', '2357.TW',
    '2379.TW', '2382.TW', '2395.TW', '2412.TW', '2454.TW', '2603.TW', '2880.TW', '2881.TW',
    '2882.TW', '2883.TW', '2884.TW', '2885.TW', '2886.TW', '2887.TW', '2890.TW', '2891.TW',
    '2892.TW', '2912.TW', '3008.TW', '3017.TW', '3034.TW', '3037.TW', '3045.TW', '3231.TW',
    '3661.TW', '3711.TW', '4904.TW', '4938.TW', '5871.TW', '5876.TW', '5880.TW', '6446.TW',
    '6505.TW', '6669.TW'
 ]


# 定義時間範圍
start_date = '2024-01-04' 
end_date = '2024-12-31'

# 建立空的 DataFrame
test_taiwan50_data = pd.DataFrame()

# 逐一下載每支股票的資料並整合
scalers = {}  # 用於儲存每支股票的標準化模型
for stock_id in stock_ids:
    print(f"Downloading data for {stock_id}...")
    stock_data = yf.download(stock_id, start=start_date, end=end_date)

    # 保留收盤價 (Close)
    stock_data = stock_data[['Close']].copy()

    # 將欄位名稱改為股票代號
    stock_data.rename(columns={'Close': f'Close({stock_id})'}, inplace=True)
    # stock_data.rename(columns={'Close': stock_id}, inplace=True)

    # 合併到總表，按日期對齊
    if test_taiwan50_data.empty:
        test_taiwan50_data = stock_data
    else:
        test_taiwan50_data = test_taiwan50_data.join(stock_data, how='outer')

# 重置索引為日期，確保整潔
test_taiwan50_data.index.name = 'Date'

# 檢視資料
print(test_taiwan50_data.head())
print(test_taiwan50_data.tail())


#%%
# 計算每支股票的漲跌幅
# 對每支股票計算 2024-01-04 和 2024-12-31 的價格變動百分比
price_change = (test_taiwan50_data.iloc[-1] - test_taiwan50_data.iloc[0]) / test_taiwan50_data.iloc[0]

# 找出漲幅最大和跌幅最小的股票
max_gain_stock = price_change.idxmax()  # 漲幅最大的股票
max_loss_stock = price_change.idxmin()  # 跌幅最大的股票

# 輸出結果
print(f"2024年01月04日到2024年12月31日，漲幅最大的是: {max_gain_stock}")
print(f"2024年01月04日到2024年12月31日，跌幅最大的是: {max_loss_stock}")

# 如果想知道具體的漲幅值，可以這樣：
print(f"{max_gain_stock} 的漲幅是: {price_change[max_gain_stock] * 100:.2f}%")
print(f"{max_loss_stock} 的跌幅是: {price_change[max_loss_stock] * 100:.2f}%")



#%%
"""7. 預測個股

"""
"""
準備奇鋐科技3017、台塑化6505資料
"""
# 讀取 stock_price_news_data csv
stock_price_news_data = pd.read_csv("./data/stock_price_news_data.csv")
# 去除date欄位
stock_price_news_data = stock_price_news_data.drop(columns=["Date"])
stock_price_news_data

# 預測台積電股價
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# 定義參數
window_size = 20
# target_stock = "Close(3017.TW)"  # 奇鋐科技
# target_stock_news = "Daily Sentiment(3017)"  # 奇鋐科技
target_stock = "Close(6505.TW)"  # 台塑化
target_stock_news = "Daily Sentiment(6505)"  # 台塑化

# 計算所有股票之間的相關係數
correlation_matrix = stock_price_news_data.corr()

# 提取正相關和負相關的股票清單
positive_corr_dict_one = {}
negative_corr_dict_one = {}

# 準備多重輸入的數據
X_target_one, X_index_one, X_positive_one, X_negative_one, X_sentiment_one, y_one = [], [], [], [], [], []

# 從第 n 日到第 t 日處理資料
for i in range(window_size, len(stock_price_news_data)):  # 從 window_size 開始，因為需要足夠的滑動視窗
    # 取得正相關和負相關的股票
    correlations_one = correlation_matrix[target_stock].drop(target_stock)  # 排除自身的相關係數
    positive_corr_dict_one[target_stock] = correlations_one.nlargest(10).index  # 正相關前10
    negative_corr_dict_one[target_stock] = correlations_one.nsmallest(10).index  # 負相關前10

    positive_corr_stocks_one = positive_corr_dict_one[target_stock]
    negative_corr_stocks_one = negative_corr_dict_one[target_stock]

    # 滑動視窗特徵
    target_window_one = stock_price_news_data[target_stock].iloc[i-window_size:i].values
    index_window_one = stock_price_news_data["Close(^TWII)"].iloc[i-window_size:i].values  # 台股加權指數
    positive_window_one = stock_price_news_data[positive_corr_stocks_one].iloc[i-window_size:i].values
    negative_window_one = stock_price_news_data[negative_corr_stocks_one].iloc[i-window_size:i].values
    sentiment_window_one = stock_price_news_data[target_stock_news].iloc[i-window_size:i].values

    # 將結果加入輸入
    X_target_one.append(target_window_one)
    X_index_one.append(index_window_one)
    X_positive_one.append(positive_window_one)
    X_negative_one.append(negative_window_one)
    X_sentiment_one.append(sentiment_window_one)
    y_one.append(stock_price_news_data[target_stock].iloc[i])  # 預測目標是當天的股價

# 轉換為 NumPy 陣列
X_target_one = np.array(X_target_one)
X_index_one = np.array(X_index_one)
X_positive_one = np.array(X_positive_one)
X_negative_one = np.array(X_negative_one)
X_sentiment_one = np.array(X_sentiment_one)
y_one = np.array(y_one)

# 輸出資料形狀
print("X_target shape:", X_target_one.shape)
print("X_index shape:", X_index_one.shape)
print("X_positive shape:", X_positive_one.shape)
print("X_negative shape:", X_negative_one.shape)
print("X_sentiment_one shape:", X_sentiment_one.shape)
print("y shape:", y_one.shape)


from sklearn.model_selection import train_test_split

# 資料集以 8:2 的比率切割訓練及測試集
train_size_one = int(len(X_sentiment_one) * 0.8)
test_size_one = len(X_sentiment_one) - train_size  # 測試集大小為剩餘部分

X_train_target_one, X_test_target_one = X_target_one[:train_size_one], X_target_one[train_size_one:]
X_train_index_one, X_test_index_one = X_index_one[:train_size_one], X_index_one[train_size_one:]
X_train_positive_one, X_test_positive_one = X_positive_one[:train_size_one], X_positive_one[train_size_one:]
X_train_negative_one, X_test_negative_one = X_negative_one[:train_size_one], X_negative_one[train_size_one:]
X_train_sentiment_one, X_test_sentiment_one = X_sentiment_one[:train_size_one], X_sentiment_one[train_size_one:]
y_train_one, y_test_one = y_one[:train_size_one], y_one[train_size_one:]

# 輸出資料形狀
print("X_train_target shape:", X_train_target_one.shape)
print("X_test_target shape:", X_test_target_one.shape)
print("- - - - - - - - -")
print("X_train_index shape:", X_train_index_one.shape)
print("X_test_index shape:", X_test_index_one.shape)
print("- - - - - - - - -")
print("X_train_positive shape:", X_train_positive_one.shape)
print("X_test_positive shape:", X_test_positive_one.shape)
print("- - - - - - - - -")
print("X_train_negative shape:", X_train_negative_one.shape)
print("X_test_negative shape:", X_test_negative_one.shape)
print("- - - - - - - - -")
print("X_train_sentiment shape:", X_train_sentiment_one.shape)
print("X_test_sentiment shape:", X_test_sentiment_one.shape)
print("- - - - - - - - -")
print("y_train shape:", y_train_one.shape)
print("y_test shape:", y_test_one.shape)


# 為了符合 LSTM 的輸入，重塑輸入為 (samples, time steps, features)
X_train_one = {
    "target_stock": X_train_target_one.reshape(-1, window_size, 1).astype(np.float32),
    "benchmark": X_train_index_one.reshape(-1, window_size, 1).astype(np.float32),
    "positive_stocks": X_train_positive_one.astype(np.float32),
    "negative_stocks": X_train_negative_one.astype(np.float32),
    "sentiment": X_train_sentiment_one.reshape(-1, window_size, 1).astype(np.float32),
}
X_test_one = {
    "target_stock": X_test_target_one.reshape(-1, window_size, 1).astype(np.float32),
    "benchmark": X_test_index_one.reshape(-1, window_size, 1).astype(np.float32),
    "positive_stocks": X_test_positive_one.astype(np.float32),
    "negative_stocks": X_test_negative_one.astype(np.float32),
    "sentiment": X_test_sentiment_one.reshape(-1, window_size, 1).astype(np.float32),
}

print("X_train_target shape:", X_train_one["target_stock"].shape)
print("X_train_index shape:", X_train_one["benchmark"].shape)
print("X_train_positive shape:", X_train_one["positive_stocks"].shape)
print("X_train_negative shape:", X_train_one["negative_stocks"].shape)
print("X_train_sentiment shape:", X_train_one["sentiment"].shape)
print("- - - - - - - - -")

print("X_test_target shape:", X_test_one["target_stock"].shape)
print("X_test_index shape:", X_test_one["benchmark"].shape)
print("X_test_positive shape:", X_test_one["positive_stocks"].shape)
print("X_test_negative shape:", X_test_one["negative_stocks"].shape)
print("X_test_sentiment shape:", X_test_one["sentiment"].shape)
print("- - - - - - - - -")

print("y_train shape:", y_train_one.shape)
print("y_test shape:", y_test_one.shape)



#%%
"""真實股價

"""
# realprice = test_taiwan50_data['Close(3017.TW)']
realprice = test_taiwan50_data['Close(6505.TW)']

realprice



#%%
"""LSTM_target+index+corr+news 模型 預測個股

"""
# 預測台積電股價
import matplotlib.pyplot as plt

y_pred_one = model_target_index_corr_news.predict(
    [X_test_one["target_stock"], X_test_one["benchmark"], X_test_one["positive_stocks"], X_test_one["negative_stocks"], X_test_one["sentiment"]]
)


# 反正規化預測值 (若有使用標準化，請取消註解並修改)
# y_pred_one = scalers['3017.TW'].inverse_transform(y_pred_one)
y_pred_one = scalers['6505.TW'].inverse_transform(y_pred_one)

# 預測值轉為 DataFrame 格式 (使用 taiwan50_data 的索引)
test_predict_one = pd.DataFrame(
    y_pred_one,
    columns=['Predicted Price'],
    index=taiwan50_data.index[train_size_one + window_size:]
)
test_predict_one



#%%
"""LSTM_target+index+corr+news 繪製測試集

"""
# 只繪製測試集
plt.figure(figsize=(14, 5))

# 測試集部分
plt.plot(realprice,
         label='Testing Data', color='green')

# 預測部分
plt.plot(test_predict_one, label='Predicted Data', color='red')

# plt.title('3017 Stock Prices')
plt.title('6505 Stock Prices')
plt.xlabel('Days')
plt.ylabel('Prices')
plt.legend()
plt.show()




#%%
"""8. 回測

"""
import pandas as pd

# 只取測試集
realprice_backtesting = realprice.reset_index().rename(columns={"index": "Date", "6505.TW": "Actual Price"})
# 確保 Date 是索引或欄位對齊
realprice_backtesting.set_index('Date', inplace=True)  # 將 Date 設為索引
test_predict_2330_backtesting = test_predict_one

# 假設 test_predict_one 和 realprice 已按日期排序
# test_predict_one["Shifted Predicted Price"] = test_predict_one["Predicted Price"].shift(-1)  # 將預測值向前移一天
realprice_backtesting["Shifted Actual Price"] = realprice_backtesting["Actual Price"].shift(1)  # 將真實值向後移一天

# 去除最後一筆沒有對應預測值的資料
valid_data = pd.concat([test_predict_2330_backtesting, realprice_backtesting], axis=1).iloc[1:]

# 初始化交易設定
initial_capital = 1_000_000  # 期初資金
trade_unit = 1000  # 每次交易的基準單位

# 將回測結果儲存在 DataFrame 中
backtest_results = {
    "Date": [],
    "Action": [],
    "Stock Price": [],
    "Shares Held": [],
    "Cash": [],
    "Portfolio Value": []
}

# 初始化變數
cash = initial_capital
shares_held = 0

# 交易回測
for _, row in valid_data.iterrows():
    predicted_price = row["Predicted Price"]
    actual_price = row["Shifted Actual Price"]
    action = "Hold"

    # 判斷交易行為
    if predicted_price > actual_price:
        if shares_held == 0:  # 空手狀態，買入
            action = "Buy"
            shares_held = cash // (actual_price * trade_unit) * trade_unit
            cash -= shares_held * actual_price
    elif predicted_price < actual_price:
        if shares_held > 0:  # 持有股票，賣出
            action = "Sell"
            cash += shares_held * actual_price
            shares_held = 0

    # 計算投資組合價值
    portfolio_value = cash + shares_held * actual_price

    # 儲存結果
    backtest_results["Date"].append(row.name)
    backtest_results["Action"].append(action)
    backtest_results["Stock Price"].append(actual_price)
    backtest_results["Shares Held"].append(shares_held)
    backtest_results["Cash"].append(cash)
    backtest_results["Portfolio Value"].append(portfolio_value)

# 將回測結果轉為 DataFrame
backtest_df = pd.DataFrame(backtest_results)

# 輸出回測結果
print(backtest_df)


#%%
"""回測 繪製資產變化圖

"""

buy_and_hold_shares = initial_capital // (valid_data["Actual Price"].iloc[0] * trade_unit) * trade_unit
buy_and_hold_cash = initial_capital - buy_and_hold_shares * valid_data["Actual Price"].iloc[0]
buy_and_hold_portfolio = buy_and_hold_cash + buy_and_hold_shares * valid_data["Actual Price"]

# 繪製資產變化圖
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(backtest_df["Date"], backtest_df["Portfolio Value"], label="Backtest Portfolio Value")
plt.plot(backtest_df["Date"], buy_and_hold_portfolio, label="Buy-and-Hold Portfolio Value", linestyle="--")
plt.axhline(initial_capital, color="red", linestyle="--", label="Initial Capital")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("Backtest vs. Buy-and-Hold Portfolio Value Over Time")
plt.legend()
plt.grid(True)
plt.show()

# 
# %%
# 計算交易回測報酬率
backtest_return = (backtest_df["Portfolio Value"].iloc[-1] - initial_capital) / initial_capital 

# 計算買入後持有報酬率
holding_return = (buy_and_hold_portfolio.iloc[-1] - initial_capital) / initial_capital 

# 輸出結果
print(f"Backtest Return: {backtest_return * 100:.2f}%")
print(f"Holding Return: {holding_return * 100:.2f}%")



