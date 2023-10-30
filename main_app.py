#streamlit runapp.py

import streamlit as st #Webアプリを簡単に作れるライブラリ
import io #データ化したいページのURLを入力するだけで、自動でデータ箇所を判断して情報を集めてくれるスクレイピングサービス
import json # JSON
import gzip #ファイルを圧縮、展開するシンプルなインターフェイス
import requests #HTTPライブラリ

import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def main():
    st.title('キーワードで人気のタイトル検索 ')

    key = st.text_input(label='キーワードを入力してください！！:')
    keyword = list(map(str,key.split()))
    keywords = ' '.join(keyword)

    # バリデーション処理
    if len(keyword) < 1:
        st.warning('Please input keyword')
        # 条件を満たないときは処理を停止する
        st.stop()


    # 欲しい情報
    attrs = ["title", "ncode"]

    # 項目から of パラメータへの対応付け
    attr2of = {
        "title": "t",
        "ncode": "n",
        "userid": "u",
        "writer": "w",
        "story": "s",
        "biggenre": "bg",
        "genre": "g",
        "keyword": "k",
        "general_firstup": "gf",
        "general_lastup": "gl",
        "noveltype": "nt",
        "end": "e",
        "general_all_no": "ga",
        "length": "l",
        "time": "ti",
        "isstop": "i",
        "isr15": "ir",
        "isbl": "ibl",
        "isgl": "igl",
        "iszankoku": "izk",
        "istensei": "its",
        "istenni": "iti",
        "pc_or_k": "p",
        "global_point": "gp",
        "daily_point": "dp",
        "weekly_point": "wp",
        "monthly_point": "mp",
        "quarter_point": "qp",
        "yearly_point": "yp",
        "fav_novel_cnt": "f",
        "impression_cnt": "imp",
        "review_cnt": "r",
        "all_point": "a",
        "all_hyoka_cnt": "ah",
        "sasie_cnt": "sa",
        "kaiwaritu": "ka",
        "novelupdated_at": "nu",
        "updated_at": "ua",
    }

    def get_allcount_for_keyword(url):
        payload = {
            "gzip": 5,
            "out": "json",
            "lim": 1,
            "order": "hyoka",
        }
        res = requests.get(url, params=payload)
        if res.status_code != 200:
            return None
        gzip_file = io.BytesIO(res.content)
        with gzip.open(gzip_file, "rt") as f:
            json_data = f.read()
        json_data = json.loads(json_data)
        return json_data[0]["allcount"]


    def get_info(url,  allcount, lim_per_page, page_no):
        # 1 ページあたり `lim_per_page` 件表示として、`page_no` ページ目の情報を表示
        st = (page_no - 1) * lim_per_page + 1

        payload = {
            "gzip": 5,
            "out": "json",
            "of": "-".join([attr2of[attr] for attr in attrs]),
            "lim": lim_per_page,
            "st": st,
            "order": "hyoka",
        }
        res = requests.get(url, params=payload)
        if res.status_code != 200:
            return []
        gzip_file = io.BytesIO(res.content)
        with gzip.open(gzip_file, "rt") as f:
            json_data = f.read()
        json_data = json.loads(json_data)
        return json_data[1:]

    url = "https://api.syosetu.com/novelapi/api/"


    allcount = get_allcount_for_keyword(url)
    lim_per_page = 20

    data_title = []
    data_ncode = []
    for p in range(1,50):
        for i, info in enumerate(get_info(url, allcount, lim_per_page, p)):
            for attr in attrs:
                if attr == 'title':
                    value = info[attr]
                    data_title.append(value)
                else:
                    value = info[attr]
                    data_ncode.append(value)


    from janome.tokenizer import Tokenizer #形態素解析ライブラリ

    tokenizer = Tokenizer()

    def get_token(text):
        t = Tokenizer()
        tokens = t.tokenize(text)
        word = ""
        for token in tokens:
            part_of_speech = token.part_of_speech.split(",")[0]
            if part_of_speech == "名詞":
                word +=token.surface + " "
            if part_of_speech == "動詞":
                word +=token.base_form+ " "
            if part_of_speech == "形容詞":
                word +=token.base_form+ " "
        return word

    documents=[]
    for item in data_title:
        token=get_token(item)
        documents.append(token)

    def stems(doc):
        result = []
        t = Tokenizer()
        tokens = t.tokenize(doc)
        for token in tokens:
            result.append(token.surface.strip())
        return result

    novel_dic = []
    documents.insert(0, keywords)
    data_title.insert(0, keywords)
    data_ncode.insert(0, keywords)
    vectorizer = TfidfVectorizer(analyzer=stems)
    t = Tokenizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    ix = 0
    similarities = cosine_similarity(tfidf_matrix[ix], tfidf_matrix)
    for similarity in similarities:
        for ix2 in range(len(similarity)):
            link = f'https://ncode.syosetu.com/{data_ncode[ix2].lower()}/'
            novel = {'小説名': data_title[ix2], 'tf-idf': similarity[ix2], 'URL': link}
            novel_dic.append(novel)


    del documents[0]
    del data_title[0]
    del data_ncode[0]

    novel_dic.pop(0)
    df_novel = pd.DataFrame()
    df_new = pd.DataFrame(novel_dic)
    df_novel = pd.concat([df_novel, df_new], ignore_index=True)
    df_s = df_novel.sort_values('tf-idf', ascending=False)
    df_r = df_s.reset_index(drop=True)

    st.markdown(df_r.head(50).to_html(render_links=True),unsafe_allow_html=True)




if __name__ == '__main__':
    main()