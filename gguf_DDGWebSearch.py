import datetime
from duckduckgo_search import DDGS
import re
from langchain_community.document_loaders import WebBaseLoader #LangChainがこっちでインポートしてって言っている

def Run(llm=None,prompt=None,model_type=1,site_summary="",n_ctx=1024):#Web検索の判断、実行
   """
   Web検索関数

   Args:
        llm (variable) : 事前にロードしたllama-cpp-python (デフォルト: None)
        prompt (str): プロント (デフォルト: None)
        model_type (int): 独自実装のプロンプトトークン (デフォルト: 1(llama2系))
        site_summary (str): インターネット検索結果キャッシュ  (デフォルト: インターネットインターネット検索しません)
        n_ctx (int): Web検索で取得する文字数 (デフォルト: 1024)
   """   
   if model_type == 1:
     #llama2
     system_bos_token="[INST] <<SYS>>\n"
     system_eos_token="\n<</SYS>>\n\n"
     user_bos_token=""
     assistant_bos_token=" [/INST]"
     user_eos_token="</s>"
     assistant_eos_token="</s>"

   if model_type == 2:
     #llama3
     system_bos_token="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
     system_eos_token="<|eot_id|>\n"
     user_bos_token="<|start_header_id|>user <|end_header_id|>\n\n"
     assistant_bos_token="<|start_header_id|>assistant <|end_header_id|>\n\n"
     user_eos_token="<|eot_id|>\n"
     assistant_eos_token="<|eot_id|>\n"

   if model_type == 3:
     #chatml,calm3,qwen
     system_bos_token="<|im_start|>system\n"
     system_eos_token="<|im_end|>\n"
     user_bos_token="<|im_start|>user\n"
     assistant_bos_token="<|im_start|>assistant\n"
     user_eos_token="<|im_end|>\n"
     assistant_eos_token="<|im_end|>\n"

   if model_type == 4:
     #gemma
     system_bos_token=""
     system_eos_token=""
     user_bos_token="<start_of_turn>user\n"
     assistant_bos_token="<start_of_turn>model\n"
     user_eos_token="<end_of_turn>\n"
     assistant_eos_token="<end_of_turn>\n"

   if model_type == 5:
     #phi-4
     system_bos_token="<|im_start|>system<|im_sep|>\n"
     system_eos_token="<|END_OF_TURN_TOKEN|>\n"
     user_bos_token="<|im_start|>user<|im_sep|>\n"
     assistant_bos_token="<|im_start|>assistant<|im_sep|>\n"
     user_eos_token="<|im_end|>\n" 
     assistant_eos_token="<|im_end|>\n"

   if model_type == 6:
     #aya
     system_bos_token="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\n"
     system_eos_token="<|END_OF_TURN_TOKEN|>\n"
     user_bos_token="<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\n"
     assistant_bos_token="<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\n"
     user_eos_token="<|END_OF_TURN_TOKEN|>\n"  
     assistant_eos_token= "<|END_OF_TURN_TOKEN|>\n"  

   if model_type == 7:
     #DeepSeek
     system_bos_token=""
     system_eos_token=""
     user_bos_token="<｜User｜>"
     assistant_bos_token="<｜Assistant｜>"
     user_eos_token="\n"  
     assistant_eos_token= "<｜end▁of▁sentence｜>"

   if model_type == 8:
     #sarashina2.2
     system_bos_token="<|system|>\n"
     system_eos_token="<|end|>\n"
     user_bos_token="<|user|>\n"
     assistant_bos_token="<|assistant|>\n"
     user_eos_token="<|end|>\n"
     assistant_eos_token="<|end|>\n"
     
   if prompt !="":#プロンプトに入力があったら各種推論実行
#############################################################
#############################################################
#############################################################
      day = (str(datetime.datetime.now().year)+"年"\
           +str(datetime.datetime.now().month)+"月"\
            +str(datetime.datetime.now().day)+"日"\
            +str(datetime.datetime.now().strftime("%a "))\
            +str(datetime.datetime.now().hour)+"時"\
            +str(datetime.datetime.now().minute)+"分")\
               .replace("Sun", "（日）").replace("Mon", "（月）").replace("Tue", "（火）").replace("Wed", "（水）")\
                  .replace("Thu", "（木）").replace("Fri", "（金）").replace("Sat", "（土）")
#1段目の推論。ツールの使用を判断
      if model_type==5:
       first_prompt=f"{user_bos_token}\
あなたはWeb・インターネット検索で最新の情報を取得できます。\n\
\n\
以下の会話履歴とキャッシュされている検索結果を見て、それらにも、あなたの知識にもない新たにWeb・インターネット検索が必要な質問をされていると判断したら1とだけ出力してください。\n\
新たにWeb・インターネット検索が必要な質問はされていないと判断したら0とだけ出力してください。\n\
\n\
** 現在時刻 **:\n\
{day}\n\
** 会話履歴 **:\n\
{prompt}{user_eos_token}\
** キャッシュされている検索結果 **:\n\
{site_summary}\n\
{assistant_bos_token}"
      if model_type!=5:
       first_prompt=f"{system_bos_token}\
あなたは会話の際にWeb・インターネット検索で最新の情報を取得できるアシスタントです。{system_eos_token}\
{user_bos_token}\
\n\
以下の会話履歴とキャッシュされている検索結果を見て、それらにも、あなたの知識にもない新たにWeb・インターネット検索が必要な質問をされていると判断したら1とだけ出力してください。\n\
新たにWeb・インターネット検索が必要な質問はされていないと判断したら0とだけ出力してください。\n\
\n\
** 現在時刻 **:\n\
{day}\n\
** 会話履歴 **:\n\
{prompt}\
** キャッシュされている検索結果 **:\n\
{site_summary}{user_eos_token}\n\
{assistant_bos_token}"
      print(first_prompt)
      tool_use_flag = llm(
           prompt=first_prompt,
           max_tokens=1024,
           temperature = 0.8,
           top_p=0.95, 
           top_k=40, 
      )
      try:
       tool_use_flag=int((tool_use_flag["choices"][0]["text"]).replace("\n",""))
      except:
       tool_use_flag=0
      #tool_use_flag=1
      print(tool_use_flag)
#1段目の推論。web検索ワード生成
   if tool_use_flag ==1:
        print("インターネット検索必要\n")
        if model_type==5:
         websearch_prompt=f"{user_bos_token}\n\
あなたはWeb・インターネット検索で最新の情報を取得できます。\n\
\n\
以下の会話履歴の最後userからの質問に対してWeb検索クエリに使う単語を抽出及び生成し、単語だけ出力してください。複数ある場合は半角スペースで区切って別々に抽出、生成してください。\n\
質問にURLを含む場合はURLだけを出力してください。\n\
\n\
** 会話履歴 **:\n\
{prompt}{user_eos_token}\n\
{assistant_bos_token}"
        else:
         websearch_prompt=f"{system_bos_token}\n\
あなたは会話の際にWeb・インターネット検索で最新の情報を取得できるアシスタントです。{system_eos_token}\n\
{user_bos_token}\n\
\n\
以下の会話履歴の最後userからの質問に対してWeb検索クエリに使う単語を生成し、単語だけ出力してください。複数ある場合は半角スペースで区切って別々に生成してください。\n\
質問にURLを含む場合はURLだけを出力してください。\n\
\n\
** 会話履歴 **:\n\
{prompt}{user_eos_token}\n\
{assistant_bos_token}"
        print(websearch_prompt)
        websearch_word = llm(
               prompt=websearch_prompt,
               max_tokens=1024,
               temperature = 0.8,
               top_p=0.95, 
               top_k=40, 
               stop=["\n"],
               stream=True
        )
        output = ""
        chunk=""
        for chunk in websearch_word:
         output += chunk['choices'][0]['text']
         print(chunk['choices'][0]['text'],end="",flush=True)
         yield output
        websearch_word=output
        print(websearch_word)
        # クエリ
        with DDGS() as ddgs:
         results = list(ddgs.text(
          keywords=websearch_word,  # 検索ワード
          region='jp-jp',  # リージョン 日本は"jp-jp",指定なしの場合は"wt-wt"
          safesearch='moderate',  # セーフサーチOFF->"off",ON->"on",標準->"moderate"
          timelimit=None,  # 期間指定 指定なし->None,過去1日->"d",過去1週間->"w",
                         # 過去1か月->"m",過去1年->"y"
          max_results=3,
          backend="lite"  # 取得件数
         ))

        # URLリストの生成
        titles = [result["title"] for result in results]
        urls = [result["href"] for result in results]
        discriptiopns = [result["body"] for result in results]
        urls_titles=urls,titles
        urls_titles_discriptiopns=urls,titles,discriptiopns
        print(urls_titles_discriptiopns)
        yield urls_titles_discriptiopns
        # Webページの読み込み
        import time
        time.sleep(2)
        documents = []
        for url in urls:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())

        # 各ドキュメントのテキストに対して処理を行う
        for i, doc in enumerate(documents):
            # 2つ以上の連続した改行を1つの改行に置換
            cleaned_text = re.sub(r'\n{2,}', '\n', doc.page_content)
            # 置換したテキストを元のドキュメントに戻す
            documents[i].page_content = cleaned_text

        content_list = [doc.page_content for doc in documents]
        content = "".join(content_list)
        content=str(urls_titles)+"\n"+content
        content = content.replace("\\n", "\n").replace("\\u3000", "\u3000").replace("！", "?").replace("？", "?")

        # n_ctx文字に最も近い位置で分割
        def split_text_at_nearest(content, target_length):
            # 分割のための句読点や改行などをリストアップ
            delimiters = ['。', '.', '!', '?', '\n']
            # 最も近い位置を見つける
            closest_position = float('inf')
            for delimiter in delimiters:
                pos = content.find(delimiter, target_length)
                if pos != -1 and pos < closest_position:
                   closest_position = pos

            # 最も近い位置で分割
            if closest_position != float('inf'):
               return content[:closest_position + 1]
            else:
            # 適切な区切りが見つからなかった場合は、強制的にn_ctx文字で分割
              return content[:target_length]

        # n_ctx文字に最も近い位置で分割した最初の部分を取得
        result = split_text_at_nearest(content, n_ctx)
        yield result

        # 結果の表示
        print(result)
        site_summary=result
#############################################################
#############################################################
#############################################################
        if model_type==5:
         web_search_summary_prompt=f"{user_bos_token}\n\
あなたはWeb・インターネット検索で最新の情報を取得できます。\n\
\n\
次の会話履歴を踏まえて{websearch_word}を検索ワードにWeb検索結果を取得しました。\n\
これを会話履歴の流れに対して助けになるよう要約してまとめてください。「具体的に」「詳しく」「一言で」などの指示がある場合はそれに従ってください。\n\
ソースコードのように省略できないものはそのまま出力してください。要約内容だけ出力し、他は出力しないでください。\n\
\n\
** 現在時刻 **:\n\
{day}\n\
** 会話履歴 **:\n\
{prompt}\n\
\n\
** Web検索結果 **:\n\
{result}\n\
{user_eos_token}\n\
{assistant_bos_token}"
        if model_type!=5:
         web_search_summary_prompt=f"{system_bos_token}\n\
あなたは会話の際にWeb・インターネット検索で最新の情報を取得できるアシスタントです。{system_eos_token}\n\
{user_bos_token}\n\
次の会話履歴を踏まえて{websearch_word}を検索ワードにWeb検索結果を取得しました。\n\
これを基に会話履歴に対しての回答を行ってください。特に指定がない場合は短く簡潔に結果をまとめてください。「具体的に」「詳しく」「一言で」などの指示がある場合はそれに従ってください。\n\
ソースコードのように省略できないものはそのまま出力してください。\n\
\n\
** 現在時刻 **:\n\
{day}\n\
** 会話履歴 **:\n\
{prompt}\n\
\n\
** Web検索結果 **:\n\
{result}\n\
{user_eos_token}\n\
{assistant_bos_token}"
        web_search_summary = llm(
               prompt=web_search_summary_prompt, # Prompt
               max_tokens=1024,
               temperature = 0.8,
               top_p=0.95, 
               top_k=40, 
               stream=True
        )
        output = ""
        chunk=""
        for chunk in web_search_summary:
         output += chunk['choices'][0]['text']
         print(chunk['choices'][0]['text'],end="",flush=True)
         yield output
        site_summary="・検索したWebサイト\n"+str(urls_titles)+"\n・まとめ\n"+output
        print(site_summary)
   if tool_use_flag !=1 or prompt =="":
      print("インターネット検索をしない\n")
   yield site_summary