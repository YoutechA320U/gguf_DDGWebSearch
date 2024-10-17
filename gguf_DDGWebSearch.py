import datetime
from langchain.document_loaders import WebBaseLoader
#llama2
#system_bos_token="<s>[INST] <<SYS>>\n"
#system_eos_token="\n<</SYS>>\n\n"
#user_bos_token=""
#assistant_bos_token=" [/INST]"
#eos_token="</s>"

#llama3
#system_bos_token="<|begin_of_text|><<|start_header_id|>system<|end_header_id|>\n"
#system_eos_token="<|eot_id|>\n"
#user_bos_token="<|start_header_id|>user <|end_header_id|>\n\n"
#assistant_bos_token="<|start_header_id|>assistant <|end_header_id|>\n\n"
#eos_token="<|eot_id|>\n"

#calm3,qwen
system_bos_token="<|im_start|>system\n"
system_eos_token="<|im_end|>\n"
user_bos_token="<|im_start|>user\n"
assistant_bos_token="<|im_start|>assistant\n"
eos_token="<|im_end|>\n"

#gemma2
#system_bos_token=""
#system_eos_token=""
#user_bos_token="<start_of_turn>user\n"
#assistant_bos_token="<start_of_turn>model\n"
#eos_token="<end_of_turn>\n"

def Run(n_ctx,llm,prompt):#Web検索の判断、実行
   if prompt !="":#プロンプトに入力があったら各種推論実行
#############################################################
#############################################################
#############################################################
      day = (str(datetime.datetime.now().year)\
+"年"+str(datetime.datetime.now().month)\
   +"月"+str(datetime.datetime.now().day)\
      +"日"+str(datetime.datetime.now().strftime(" %a "))\
         +str(datetime.datetime.now().hour)\
            +"時"+str(datetime.datetime.now().minute)+"分")\
               .replace("Sun", "日曜日").replace("Mon", "月曜日").replace("Tue", "火曜日").replace("Wed", "水曜日")\
                  .replace("Thu", "木曜日").replace("Fri", "金曜日").replace("Sat", "土曜日")
#1段目の推論。ツールの使用を判断
      first_prompt=f"{system_bos_token}\
あなたは文章の理解、要約、ロールプレイなど様々なタスクをこなせる日本語の大規模言語モデルです。{system_eos_token}\
{user_bos_token}\
あなたは会話の際にWeb・インターネット検索ができます。\n\
以下の会話履歴であなたの知らないことを質問されたら、Web・インターネット検索が必要と判断したら1を出力してください。\n\
Web・インターネット検索が不要と判断したら0を出力してください。\n\
\n\
** 現在時刻 **:\n\
{day}\n\
** 会話履歴 **:\n\
{prompt}{eos_token}\
{assistant_bos_token}"
      print(first_prompt)
      tool_use_flag = llm(
           prompt=first_prompt,
           max_tokens=1024,
           temperature = 0.77,
           top_p=0.95, 
           top_k=40, 
           #stop=[system_bos_token,system_eos_token,user_bos_token,assistant_bos_token,eos_token] 
      )
      try:
       tool_use_flag=int((tool_use_flag["choices"][0]["text"]).replace("\n",""))
      except:
       tool_use_flag=0
      print(tool_use_flag)
#2-1段目の推論。web検索ワード抽出
   if tool_use_flag ==1:
        print("インターネット検索必要\n")
        websearch_prompt=f"{system_bos_token}\
あなたは文章の理解、要約、ロールプレイなど様々なタスクをこなせる日本語の大規模言語モデルです。{system_eos_token}\
{user_bos_token}\
あなたは会話の際にWeb検索ができます。\n\
以下の会話履歴の最後userからの質問に対してWeb検索に使える単語を抽出し、単語だけ出力してください。複数ある場合は半角スペースで区切りって別々に抽出してください。\n\
質問にURLを含む場合はURLだけを出力してください。\n\
\n\
** 会話履歴 **:\n\
{prompt}{eos_token}\
{assistant_bos_token}"
        print(websearch_prompt)
        websearch_word = llm(
               prompt=websearch_prompt,
               max_tokens=1024,
               temperature = 0.77,
               top_p=0.95, 
               top_k=40, 
               #stop=[system_bos_token,system_eos_token,user_bos_token,assistant_bos_token,eos_token] 
        )
        websearch_word= websearch_word["choices"][0]["text"]
        print(websearch_word)
        from duckduckgo_search import DDGS
        import json
        # クエリ
        with DDGS() as ddgs:
         results = list(ddgs.text(
            keywords=websearch_word,      # 検索ワード
            region='jp-jp',       # リージョン 日本は"jp-jp",指定なしの場合は"wt-wt"
            safesearch='moderate',     # セーフサーチOFF->"off",ON->"on",標準->"moderate"
            timelimit=None,       # 期間指定 指定なし->None,過去1日->"d",過去1週間->"w",
                                  # 過去1か月->"m",過去1年->"y"
            max_results=10         # 取得件数
         ))
        titles=[]
        urls=[]
        discriptiopns=[]
        # レスポンスの表示
        for line in results:
            a=json.dumps(
                line,
                indent=2,
                ensure_ascii=False
            )
            json_data = json.loads(a)
            titles.append(json_data["title"])
            urls.append(json_data["href"])
            discriptiopns.append(json_data["body"])
        # 結果を表示
        print("インターネット検索結果\n")
        print(titles)
        print(urls)
        print(discriptiopns)
#############################################################
#############################################################
#############################################################
#2-2段目の推論。webサイト要約判断
        website_summary_check=f"{system_bos_token}\
あなたは文章の理解、要約、ロールプレイなど様々なタスクをこなせる日本語の大規模言語モデルです。{system_eos_token}\
{user_bos_token}\
次の会話履歴を踏まえて{websearch_word}を検索ワードにWeb検索を行い次のタイトルと説明のリストを取得しました。\n\
会話履歴の最後userからの質問に回答するのにWeb検索結果だけで十分と判断した場合は1を、不十分と判断したら2を出力してください。\n\
\n\
** 現在時刻 **:\n\
{day}\n\
** 会話履歴 **:\n\
{prompt}\n\
** Web検索結果 **:\n\
 * タイトル *:\n\
 {titles}\n\
 * 説明 *:\n\
 {discriptiopns}{eos_token}\
{assistant_bos_token}"
        print(website_summary_check)
        website_summary_flag = llm(
               prompt=website_summary_check,
               max_tokens=1024,
               temperature = 0.77,
               top_p=0.95, 
               top_k=40, 
               #stop=[system_bos_token,system_eos_token,user_bos_token,assistant_bos_token,eos_token] 
        )
        try:
         website_summary_flag=int((website_summary_flag["choices"][0]["text"]).replace("\n",""))
        except:
         website_summary_flag=1  
        print(website_summary_flag)
#############################################################
#############################################################
#############################################################
#2-3段目の推論。検索結果要約
        if website_summary_flag == 1:
            print("インターネット検索結果を要約する\n")
            web_search_summary_prompt=f"{system_bos_token}\
あなたは文章の理解、要約、ロールプレイなど様々なタスクをこなせる日本語の大規模言語モデルです。{system_eos_token}\
{user_bos_token}\
次の会話履歴を踏まえて{websearch_word}を検索ワードにWeb検索を行い次のタイトルと説明のリストを取得しました。\n\
会話履歴の最後userからの質問にWeb検索結果をもとに詳しく答えてください。\n\
\n\
** 現在時刻 **:\n\
{day}\n\
** 会話履歴 **:\n\
{prompt}\n\
\n\
** Web検索結果 **:\n\
 * タイトル *:\n\
 {titles}\n\
 * 説明 *:\n\
 {discriptiopns}{eos_token}\
{assistant_bos_token}"
            print(web_search_summary_prompt)
            web_search_summary = llm(
               prompt=web_search_summary_prompt, # Prompt
               max_tokens=1024,
               temperature = 0.77,
               top_p=0.95, 
               top_k=40, 
               #stop=[system_bos_token,system_eos_token,user_bos_token,assistant_bos_token,eos_token] 
            )
            web_search_summary= web_search_summary["choices"][0]["text"]
            site_summary=web_search_summary
            print(site_summary)
#############################################################
#############################################################
#############################################################
#2-4段目の推論。要約するwebサイト選択
        if website_summary_flag == 2:
            print("詳細に要約するサイトを選択\n")
            website_jumpcheck=f"{system_bos_token}\
あなたは文章の理解、要約、ロールプレイなど様々なタスクをこなせる日本語の大規模言語モデルです。{system_eos_token}\
{user_bos_token}\
次の会話履歴を踏まえて{websearch_word}を検索ワードにWeb検索を行い次のタイトルと説明のリストを取得しました。\n\
会話履歴の最後userからの質問に回答するために閲覧するWebサイトを* タイトル *もしくは* 説明 *のリストから0を先頭として整数で出力し、他は出力しないでください。\n\
\n\
** 会話履歴 **:\n\
{prompt}\n\
** Web検索結果 **:\n\
 * タイトル *:\n\
{titles}\n\
 * 説明 *:\n\
{discriptiopns}{eos_token}\
{assistant_bos_token}"
            print(website_jumpcheck)
            website_No = llm(
               prompt=website_jumpcheck,
               max_tokens=1024,
               temperature = 0.77,
               top_p=0.95, 
               top_k=40, 
               stop=["<|im_"] 
            )
            try:
             website_No=int((website_No["choices"][0]["text"]).replace("\n",""))
            except:
             website_No=0
            website_No=0
            #print(urls)
            #print(website_No)
            print(urls[website_No]+"を要約する\n")
#############################################################
#############################################################
#############################################################
#2-5段目の推論。選したwebサイトの要約（分割）
            loader=WebBaseLoader(urls[website_No])
            documents=loader.load()
            #print(documents)
            content_list = [doc.page_content for doc in documents]
            #print(content_list)
            content=("".join(content_list))
            #print(content)
            content =content.replace("\\n", "\n").replace("\\n", "\n").replace("\\u3000", "\u3000")\
            .replace("！","?").replace("？","?")
            if len(content)>80:
            #区切り文字を残してリスト化するのに正規表現の後読み肯定が上手くいかなかったので消えてもいい一時的文字を付与してそれで区切る
               content=content.replace("。","。区r切tりc文g字b一o時d").replace(".",".区r切tりc文g字b一o時d").replace("!","!区r切tりc文g字b一o時d").replace("?","?区r切tりc文g字b一o時d").replace("\n","\n区r切tりc文g字b一o時d")
               split_text=content.split("区r切tりc文g字b一o時d")
               #print(split_text[1031])
               #print(len(split_text))
               re_join_text=[]
               tmp_text=""
               tmp_text2=""
               for i in range(len(split_text)):
                  tmp_text+=split_text[i] 
                  if len(split_text[i]) >=n_ctx-200:#もし一区切りがコンテキスト長を超えた場合は強制的に分割する
                     force_split_text = [split_text[i][x:x+n_ctx-200] for x in range(0, len(split_text[i]), n_ctx-200)]
                     re_join_text+= force_split_text
                     tmp_text=""
                     tmp_text2=""
                  if len(tmp_text) <n_ctx-200:
                     tmp_text2+=split_text[i] 
                  if len(tmp_text) >=n_ctx-200:
                     re_join_text.append(tmp_text2)
                     tmp_text2+=split_text[i] 
                     tmp_text=""
                     tmp_text2=""
                  if tmp_text!="":
                     re_join_text.append(tmp_text)
                  split_text=re_join_text
                  #print(split_text)
                  #print(len(split_text))
                  site_summary_split_join=""
                  for i in range(len(split_text)):
                     site_summary=f"{system_bos_token}\
あなたは文章の理解、要約、ロールプレイなど様々なタスクをこなせる日本語の大規模言語モデルです。{system_eos_token}\
{user_bos_token}\
会話履歴の最後userからの質問を踏まえて{websearch_word}を検索ワードにWeb検索を行い次のWebサイトの文章を{day}に取得しました。\n\
質問に関連度が高い部分をWebサイトの文章から要約して抽出してください。質問に関連する部分を抽出できない場合は「なし」とだけ出力してください。\n\
** 会話履歴 **:\n\
{prompt}\n\
\n\
**Webサイトの文章:**\n\
{split_text[i]}{eos_token}\
{assistant_bos_token}"
                     site_summary_split_prompt = site_summary.replace("{prompt}",prompt).replace("{split_text}",split_text[i])
                     #print(site_summary)
                     site_summary_split = llm(
                        prompt=site_summary_split_prompt, # Prompt
                        max_tokens=1024,
                        temperature = 0.77,
                        top_p=0.95, 
                        top_k=40, 
                        #stop=[system_bos_token,system_eos_token,user_bos_token,assistant_bos_token,eos_token] 
                     )
                     site_summary_split= site_summary_split["choices"][0]["text"]
                     site_summary_split_join=site_summary_split_join+site_summary_split
                     site_summary_split_join=site_summary_split_join.replace("なし","\n")
                     #print(site_summary_split_join)
                  site_summary=site_summary_split_join
                  #print("結論を出力する\n"+site_summary)
#############################################################
#############################################################
#############################################################
            if len(content)<=80:
              print("サイトの内容が薄かったのでやっぱりインターネット検索結果を要約する\n")
              web_search_summary_prompt=f"{system_bos_token}\
あなたは文章の理解、要約、ロールプレイなど様々なタスクをこなせる日本語の大規模言語モデルです。{system_eos_token}\
{user_bos_token}\
次の会話履歴を踏まえて{websearch_word}を検索ワードにWeb検索を行い次のタイトルと説明のリストを取得しました。\n\
会話履歴の最後userからの質問にWeb検索結果をもとに詳しく答えてください。\n\
\n\
** 現在時刻 **:\n\
{day}\n\
** 会話履歴 **:\n\
{prompt}\n\
\n\
** Web検索結果 **:\n\
 * タイトル *:\n\
 {titles}\n\
 * 説明 *:\n\
 {discriptiopns}{eos_token}\
{assistant_bos_token}"
            print(web_search_summary_prompt)
            web_search_summary = llm(
               prompt=web_search_summary_prompt, # Prompt
               max_tokens=1024,
               temperature = 0.77,
               top_p=0.95, 
               top_k=40, 
               #stop=[system_bos_token,system_eos_token,user_bos_token,assistant_bos_token,eos_token] 
            )
            web_search_summary= web_search_summary["choices"][0]["text"]
            site_summary=web_search_summary
            print(site_summary)      
   if tool_use_flag !=1 or prompt =="":
      print("インターネット検索をしない\n")
      site_summary="なし"
   return site_summary