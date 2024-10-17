from llama_cpp import Llama

model="Qwen-2.5-32B-Instruct-Q4_K_M.gguf"

n_ctx=4096
llm = Llama(
      model_path=model,
      n_gpu_layers=-1,
      n_ctx=n_ctx,
       verbose=False,
)
import datetime
import gguf_DDGWebSearch

site_summary="なし"
day = str(datetime.datetime.now().year)\
+"/"+str(datetime.datetime.now().month)\
   +"/"+str(datetime.datetime.now().day)\
      +"/"+str(datetime.datetime.now().strftime(" %a "))\
         +str(datetime.datetime.now().hour)\
            +"/"+str(datetime.datetime.now().minute)
#プロンプト
prompt = "2024年10月現在の日本の総理大臣をインターネット検索して"
#ツール実行
site_summary=gguf_DDGWebSearch.Run(n_ctx,llm,prompt)
#ツールの実行結果
#2024年10月現在の日本の総理大臣は石破茂氏です。石破氏は自民党総裁選挙2024で選出され、2024年10月1日に正式に総理大臣としての職に就いています。
#この選挙では、決選投票の末に高市氏を破り、新総裁に選ばれました。
#石破茂総裁は、10月1日の衆議院本会議で内閣総理大臣に指名され、第214回臨時国会の冒頭で第102代内閣総理大臣に選出されました。
#また、石破総理は10月4日に第214回国会で所信表明演説を行い、日本の将来に向けた政策を述べています。

#システムプロンプト
system = f"<|im_start|>system\n\
あなたは文章の理解、要約、ロールプレイ、Web検索でリアルタイムでの情報提供ができるなど様々なタスクをこなせる日本語の大規模言語モデルです。\n\
言語は日本語です。\n\
現在の日付・曜日・時刻はそれぞれ{day}です\n\
\n\
**Web検索結果:**\n\
* 最新のWeb検索結果が以下に格納される。知識に無いWeb検索結果はここを参照する。参照しても答えられない場合は調べられなかった旨を伝える。\n\
**Web検索結果ここから\n\
{site_summary}\n\
**Web検索結果ここまで<|im_end|>"

prompt = system+"\n"+ "<|im_start|>user\n"+"2024年10月現在の日本の総理大臣をインターネット検索して"+"<|im_end|>\n<|im_start|>assistant\n"
print(prompt)
#推論
output = llm(
           prompt=prompt,
           max_tokens=1024,
           temperature = 0.77,
           top_p=0.95, 
           top_k=40, 
           stop=["<|im_"]
        )
output= output["choices"][0]["text"]
print("#最終的な回答\n"+output)
#最終的な回答
#2024年10月現在の日本の総理大臣は石破茂氏です。石破氏は2024年10月1日に正式に総理大臣としての職に就いています。
#また、石破総理は10月4日に第214回国会で所信表明演説を行い、日本の将来に向けた政策を述べています。