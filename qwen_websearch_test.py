from llama_cpp import Llama

model ="F:\ABEJA-Qwen2.5-32b-Japanese-v0.1-IQ3_M.gguf"

n_ctx=8192
llm = Llama(
      model_path=model,
      n_gpu_layers=-1,
      n_ctx=n_ctx,
      #verbose=False,
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
prompt = "2025年4月現在の日本の総理大臣をインターネットで検索して結果を要約して"
#ツール実行
for output in gguf_DDGWebSearch.Run(llm=llm,prompt=prompt,model_type=3,n_ctx=n_ctx):#model_type=3はChatML系
    pass
site_summary=output
#ツールの実行結果
#(['https://www.kantei.go.jp/jp/103/statement/2025/0401kaiken.html', 'https://ja.wikipedia.org/wiki/内閣総理大臣の一覧', 'https://www.kantei.go.jp/jp/103/statement/2025/0101nentou.html'], ['令和7年4月1日 石破内閣総理大臣記者会見 | 総理の演説・記者 ...', '内閣総理大臣の一覧 - Wikipedia', '石破内閣総理大臣 令和7年 年頭所感 - 首相官邸ホームページ'])
#・まとめ
#2025年4月現在、日本の総理大臣は石破茂氏です。石破総理は、令和7年4月1日の記者会見で、現在の経済や社会の課題に対する取り組みについて説明しており、特に賃金の引き上げ、物価高 への対応、地方創生、そして少子高齢化への対策などを重点的に進めていることを強調しています。

#システムプロンプト
system = f"<|im_start|>system\n\
あなたは優秀なアシスタントです\n\
現在の日付・曜日・時刻はそれぞれ{day}です\n\
\n\
**Web検索結果:**\n\
* 最新のWeb検索結果が以下に格納される。知識に無いWeb検索結果はここを参照する。参照しても答えられない場合は調べられなかった旨を伝える。\n\
**Web検索結果ここから\n\
{site_summary}\n\
**Web検索結果ここまで<|im_end|>"

send_prompt = system+"\n"+ "<|im_start|>user\n"+prompt+"<|im_end|>\n<|im_start|>assistant\n"
print(send_prompt)
#推論
output = llm(
           prompt=send_prompt,
           max_tokens=1024,
           temperature = 0.8,
           top_p=0.95, 
           top_k=40, 
        )
output= output["choices"][0]["text"]
print("#最終的な回答\n"+output)
#最終的な回答
#2025年4月現在、日本の総理大臣は石破茂氏です。石破総理は、経済や社会の課題に対する取り組み、特に賃金の引き上げ、物価高への対応、地方創生、少子高齢化対策などを重点的に進めています。