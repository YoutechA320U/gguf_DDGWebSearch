## gguf_DDGWebSearch

極力LangChainに頼らず実装した日本語LLM用DuckDuckGo検索ツール。`pip install -U duckduckgo_search`が必要です。

Web検索を頼むような口ぶりで入力すると検索した結果の要約を返します

既存のllama-cpp-python実装に少し足すだけでWeb検索が可能になる作りです。

ツールで検索の必要性判断、検索ワードの抽出、検索結果一覧（日本語：3件）の本文を設定した文字数に収まるよう取得して要約します。

公式のJinja2テンプレート対応前から作成しているため独自のプロンプトトークン付与のシステムを使います。

## 使い方
```python
　　from llama_cpp import Llama
    model="ABEJA-Qwen2.5-32b-Japanese-v0.1-IQ3_M.gguf"

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
    prompt = "2025年4月現在の日本の総理大臣をインターネット検索して"
    #ツール実行
    for output in gguf_DDGWebSearch.Run(llm=llm,prompt=prompt,model_type=3,n_ctx=n_ctx):#model_typeでモデルに応じたプロンプトトークンを付与。3はChatML系
    pass
    site_summary=output
    #ツールの実行結果
    print(site_summary)
    #(['https://www.kantei.go.jp/jp/103/statement/2025/0401kaiken.html', 'https://ja.wikipedia.org/wiki/内閣総理大臣の一覧', 'https://www.kantei.go.jp/jp/103/statement/2025/0101nentou.html'], ['令和7年4月1日 石破内閣総理大臣記者会見 | 総理の演説・記者 ...', '内閣総理大臣の一覧 - Wikipedia', '石破内閣総理大臣 令和7年 年頭所感 - 首相官邸ホームページ'])
    #・まとめ
    #2025年4月現在、日本の総理大臣は石破茂氏です。石破総理は、令和7年4月1日の記者会見で、現在の経済や社会の課題に対する取り組みについて説明しており、特に賃金の引き上げ、物価高 への対応、地方創生、そして少子高齢化への対策などを重点的に進めていることを強調しています。
```
## 備考
正常に日本のサイトがヒットしない場合はインストールしたライブラリから`duckduckgo_search.py`を探して`api: 'd.js'`と記載されている行を削除してみてください。

## 履歴
    [2024/10/17] - 初回リリース
    [2025/04/20] - 検索フローを大幅改修