## gguf_DDGWebSearch

極力LangChainに頼らず実装した日本語LLM用DuckDuckGo検索ツール。

Web検索を頼むような口ぶりで入力すると検索した結果の要約を返します

既存のllama-cpp-python実装に1行足すだけでWeb検索が可能になる作りです。

ツールで検索の必要性判断、検索ワードの抽出、検索結果一覧（日本語：10件）もしくは参考Webサイト1件の要約を行います。

## 使い方
```python
　　from llama_cpp import Llama
    model="Qwen-2.5-32B-Instruct-Q4_K_M.gguf"

    n_ctx=4096
    llm = Llama(
        model_path=model,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        verbose=False,
    )
    import gguf_DDGWebSearch

    site_summary="なし"
    #プロンプト
    prompt = "2024年10月現在の日本の総理大臣をインターネット検索して"
    #ツール実行
    site_summary=gguf_DDGWebSearch.Run(n_ctx,llm,prompt)
    #ツールの実行結果
    print(site_summary)
    #2024年10月現在の日本の総理大臣は石破茂氏です。石破氏は自民党総裁選挙2024で選出され、2024年10月1日に正式に総理大臣としての職に就いています。
    #この選挙では、決選投票の末に高市氏を破り、新総裁に選ばれました。
    #石破茂総裁は、10月1日の衆議院本会議で内閣総理大臣に指名され、第214回臨時国会の冒頭で第102代内閣総理大臣に選出されました。
    #また、石破総理は10月4日に第214回国会で所信表明演説を行い、日本の将来に向けた政策を述べています。
```

## 履歴
    [2024/10/17] - 初回リリース
