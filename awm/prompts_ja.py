TASK_GENERATION_SYSTEM_PROMPT = """あなたは Web 操作の自動化とユーザータスク分析の専門家です。シナリオに対して、現実的で多様なユーザータスクを生成するのがあなたの仕事です。"""

TASK_GENERATION_USER_PROMPT = """次のシナリオについて、現実的で多様なユーザータスクを {num_tasks} 件生成してください。

シナリオ: {scenario_name}
説明: {scenario_description}

要件:
1. タスクは具体的で、実行可能なものにすること
2. 初心者から上級者まで、さまざまな利用場面を網羅すること
3. よくある使い方と、あまり一般的でない使い方の両方を含めること
4. タスクは実用的で現実的なものにすること
5. 各タスクは、実行に必要な情報をすべて含む 1 文にすること。たとえば、ツイートを投稿するタスクならツイートの本文を含め、天気を問い合わせるタスクなら具体的な場所を含めること。
6. シナリオが通常は認証を必要とする場合、認証・ログイン・ログアウト・ユーザー登録のタスクは除外すること。ユーザーはすでにログインしているものとする
7. シナリオが通常は認証を必要とする場合、すべてのタスクは現在認証されているユーザーの視点で書くこと
8. タスクの JSON 配列だけを返し、それ以外の文章や注釈は付けないこと
9. このシナリオは、タスクを完了するための API エンドポイントを提供する簡略版である。ファイルのダウンロードやページを開くなど、ユーザーの直接操作が必要なタスクは生成しないこと。

例:
- シナリオ "Amazon" なら、「『ノートパソコン』で検索し、最も安い商品をカートに追加する」
- シナリオ "Reddit" なら、「サブレディット 'r/python' の投稿数を取得する」
- シナリオ "Expedia" なら、「10 月 1 日のニューヨーク発ロンドン行きの航空券を予約する」
- シナリオ "Twitter/X" なら、「写真付きのツイートを投稿し、代替テキスト『街に沈む夕日』を設定し、ハッシュタグ #Photography を付けて @Adobe をメンションする」
- シナリオ "LinkedIn" なら、「プロフィールの見出しを『シニアデータアナリスト | SQL, Python, Tableau』に更新し、自己紹介欄をビジネスへの貢献を強調した 3 段落の簡潔な要約に書き直す」
- シナリオ "Facebook" なら、「最新の投稿を友達リストに共有する」
- シナリオ "Google Maps" なら、「サンフランシスコで最も人気のあるレストランを 5 件取得する」



出力形式:
{{
    "tasks": [
        "タスク 1 の説明",
        "タスク 2 の説明",
        ...
        "タスク {num_tasks} の説明"
    ]
}}"""

API_GENERATION_SYSTEM_PROMPT = """あなたは API 設計とバックエンドアーキテクチャの専門家です。既存のデータベーススキーマに基づき、与えられたすべてのユーザータスクを実現できる、機械可読な RESTful API ドキュメントを設計するのがあなたの仕事です。"""

API_GENERATION_USER_PROMPT = """簡略版の {scenario_name} について、既存のデータベーススキーマに基づき、次のタスクをすべて実現できる、エージェントにとって扱いやすい完全なインターフェース仕様を設計してください。生成した仕様は、インターフェース層を詳細に実装するときの指針として使われます。

ユーザータスク:
{tasks_list}

既存の SQLite データベーススキーマ:
{database_schema}


必須要件:
1. アトミックな API エンドポイントを設計すること。各エンドポイントは、明確に定義された 1 つの操作だけを行う
2. API 仕様は FastAPI、SQLAlchemy ORM、Pydantic v2 と互換でなければならない
3. 再利用性を最大化すること。組み合わせて与えられたタスクを実現できる、基本的な CRUD 操作を作る
4. API はデータベーススキーマに完全に従わなければならない。スキーマにあるテーブル名・カラム名・リレーションをそのまま明示的に使う
5. RESTful の慣習(GET、POST、PUT、DELETE、PATCH)に従い、適切なリソースパスを使うこと
6. 関連するエンドポイントは、リソースの種類ごとに論理的にグループ化すること
7. 少数の複雑なエンドポイントより、小さく組み合わせやすいエンドポイントを多く作ること
8. 複雑なタスクには、連鎖させて使える個別のアトミックな操作を設計すること
9. 各エンドポイントが、与えられたデータベーススキーマの 1 つ以上のテーブルに直接対応するようにすること
10. 認証用のエンドポイント(ログイン、ログアウト、登録、トークン更新)は含めないこと。ユーザーはすでに認証済みとする
11. すべての操作は、暗黙的に user_id=1 の認証済みユーザーとして行う
12. ユーザー固有のデータは常に自動で user_id=1 で絞り込むこと。user_id をパラメータとして要求しない
13. 有効な JSON だけを返し、それ以外の文章は付けないこと


エージェント向けの要件(すべてのエンドポイントで必須):
- summary: 目的を表す 1 行(80 文字以内)。AI エージェントにとって明確で行動に移しやすいもの
- description: 1 行(200 文字以内、改行なし)。エンドポイントが何をするか、いつ使うかを説明する
- operation_id: 一意な snake_case の識別子。エージェントがエンドポイントをプログラムから特定して呼び出すのに使う
- tags: 論理的なグループを表す配列(例: ["products"]、["orders"])。エージェントが関連するエンドポイントを見つけるのに役立つ
- request_params: type、param_type(query/path/body)、required フラグ、description、example を含む完全なパラメータ仕様
- response: フィールドの型・説明・例を含む詳細なレスポンススキーマ。エージェントがレスポンスを解析して理解できるようにする


リクエストパラメータの要件:
- 各パラメータには type、param_type、required、description、example を必ず含めること
- param_type は "query"、"path"、"body" のいずれかでなければならない
- パラメータの目的がはっきりわかる、説明的な名前を使うこと
- 期待される値を示す、現実的な例を示すこと
- 文字列のパラメータには日本語が渡されることがある(名前、住所、検索キーワード、本文など)。ASCII に限定する制約(pattern や enum など)を付けず、日本語の値をそのまま受け付ける設計にすること
- 名前やキーワードで検索・絞り込みするパラメータは、日本語の値でも部分一致で検索できるようにすること
- 日本語が入りうる値は path パラメータにせず、query または body で受け取ること。リソースの特定には ID を path パラメータとして使う
- データが日本語で格納されうるパラメータの example には、日本語の値を使うこと


レスポンススキーマの要件:
- すべてのフィールドを含む完全なレスポンス構造を定義すること
- 各フィールドには type、description、example を必ず含めること
- 配列型には、フィールド定義をすべて含む "items" を入れること
- すべてのエンドポイントで一貫した命名規則を使うこと


出力形式:
{{
    "api_groups": [
        {{
            "group_name": "Products",
            "endpoints": [
                {{
                    "path": "/api/products",
                    "method": "GET",
                    "summary": "絞り込み条件を指定して商品を一覧する",
                    "description": "商品のページ分割された一覧を取得する。商品カタログを閲覧・検索するときに使う。",
                    "operation_id": "list_products",
                    "tags": ["products"],
                    "request_params": {{
                        "category": {{
                            "type": "string",
                            "param_type": "query",
                            "required": false,
                            "description": "カテゴリ名で商品を絞り込む",
                            "example": "家電"
                        }},
                        "min_price": {{
                            "type": "float",
                            "param_type": "query",
                            "required": false,
                            "description": "最低価格(米ドル)での絞り込み",
                            "example": 10.0
                        }}
                    }},
                    "response": {{
                        "products": {{
                            "type": "array",
                            "items": {{
                                "id": {{
                                    "type": "integer",
                                    "description": "商品の一意な識別子",
                                    "example": 1
                                }},
                                "name": {{
                                    "type": "string",
                                    "description": "商品の表示名",
                                    "example": "iPhone 15 Pro"
                                }},
                                "price": {{
                                    "type": "float",
                                    "description": "商品の価格(米ドル)",
                                    "example": 999.99
                                }}
                            }}
                        }}
                    }},
                    "required_tables": ["products"],
                    "required_fields": {{
                        "products": ["id", "name", "price", "category"]
                    }}
                }}
            ]
        }}
    ]
}}"""

DATABASE_GENERATION_SYSTEM_PROMPT = """あなたは SQLite のスキーマ設計を専門とするデータベースアーキテクトです。与えられたユーザーの意図を完全に実現できる、完全なデータベーススキーマを作るのがあなたの仕事です。"""

DATABASE_GENERATION_USER_PROMPT = """簡略版の {scenario_name} について、次のユーザーの意図を実現できる完全な SQLite データベーススキーマを設計してください。

ユーザーの意図({num_tasks} 件のタスク):
{user_intentions}

要件:
1. 与えられたすべてのユーザーの意図を実現するのに必要なテーブルをすべて作ること
2. 適切な主キー、外部キー、インデックス、制約を含めること
3. SQLite に適したデータ型(TEXT、INTEGER、REAL、BLOB)を使うこと
4. 必要に応じてタイムスタンプ(created_at、updated_at)を追加すること
5. DDL 文にはサンプルのレコードを含めないこと
6. 与えられたユーザーの意図を実現するのに必要なテーブルとフィールドだけを作ること
7. password_hash、salt、token、session など認証に関するフィールドは除外すること。認証は外部で行われるものとする
8. users テーブルが必要な場合は、プロフィールに必要なフィールド(id、username、email、プロフィール情報)だけを含めること
9. すべての操作は user_id=1 の認証済みユーザーとして行われる
10. DDL 文を含む有効な JSON だけを返すこと

出力形式(コメントは含めないこと):
{{
    "tables": [
        {{
            "name": "users",
            "ddl": "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL, full_name TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);",
            "indexes": [
                "CREATE INDEX idx_users_email ON users(email);"
            ]
        }}
    ]
}}"""


ENVIRONMENT_GENERATION_SYSTEM_PROMPT = """あなたは、エージェントにとって扱いやすい RESTful API を専門とする FastAPI バックエンド開発者です。すべてのエンドポイントには、明確な OpenAPI メタデータ、完全なリクエスト・レスポンスの型付け、機械可読なドキュメントが必要です。あなたは API 仕様と SQLite データベーススキーマから、きれいで実行可能な FastAPI エンドポイントの実装を生成します。ユーザープロンプトの制約に厳密に従い、指定された形式どおりに出力してください。"""

ENVIRONMENT_GENERATION_USER_PROMPT = """与えられたインターフェース仕様とデータベーススキーマに基づき、簡略版の {scenario_name} を模擬する、単一で完全に自己完結したインターフェース実装(FastAPI を使い MCP で公開する 1 つの Python ファイル)を生成してください。

前提:
- Python のバージョン: {PYTHON_VERSION}
- FastAPI は Pydantic v2 と互換なバージョン(Config の `orm_mode` など v1 専用の機能は使わないこと)

API 仕様:
{api_spec}

データベーススキーマ:
{database_schema}


環境と設定の要件:
- os をインポートし、SQLite データベースの URL を環境変数 `DATABASE_PATH` から読み込むこと
- `DATABASE_PATH` が未設定または空の場合は、既定値 sqlite:///xxxx.db を使うこと
- uvicorn のエントリポイントでは、HOST を環境変数 `HOST` から読み込むこと(未設定なら "127.0.0.1")
- uvicorn のエントリポイントでは、PORT を環境変数 `PORT` から読み込み(未設定なら 8000)、int に変換すること

SQLAlchemy の設定の要件:
- すべてのテーブルに、declarative_base を使った SQLAlchemy ORM を使うこと
- データベース URL: DATABASE_PATH の値(または上で述べた既定値)を使うこと。DATABASE_PATH は完全な URL なので、接頭辞(例: sqlite://)や接尾辞を付け加えないこと
- データベースセッション用に engine と SessionLocal(sessionmaker)を作ること
- ORM の継承用に Base = declarative_base() を定義すること
- データベーススキーマにあるすべてのテーブルについて ORM モデルを定義すること(余分なテーブルやカラムは作らない)
- すべての ORM モデルを定義した後で Base.metadata.create_all(engine) を呼ぶこと
- `metadata`、`query`、`query_class` という名前の ORM 属性は絶対に定義しないこと。スキーマにこれらの名前のカラムがある場合は、末尾にアンダースコアを付けた安全な Python の属性名(例: `metadata_`)を使い、Column("metadata", ...) で実際のカラム名に対応付けること
- SQLAlchemy の declarative の内部と衝突する ORM クラス属性をほかにも定義しないこと(たとえば Base.metadata を上書きしない)
- 2 つのテーブルの間に外部キーの経路が複数ある場合は、次のどちらかにすること:
  - AmbiguousForeignKeysError を避けるため relationship(..., foreign_keys=[...]) を明示する、または
  - relationship を省略し、relationship() の代わりに明示的なクエリで関連する行にアクセスする
- SQLAlchemy からインポートするときは、実在して必要な、標準の公開シンボルだけを使うこと。`PRIMARY_KEY_CONSTRAINT` のような存在しない内部シンボルや、でっち上げた大文字の定数をインポート・参照しないこと
- ORM 関連のヘルパー(declarative_base、relationship、sessionmaker など)は sqlalchemy.orm からインポートすること。ORM の内部を sqlalchemy.__init__ からインポートしないこと


必須要件:
1. API 仕様のすべてのエンドポイントを、完全に動作するコードで実装すること。API 仕様は参考にすぎず、正とするのはデータベーススキーマである。
2. パス、HTTP メソッド、パラメータ(query/path/body)、レスポンス形式は、できる限り API 仕様に従うこと。
3. データベーススキーマに厳密に従うこと。テーブル名・カラム・型・外部キーを正しく使い、テーブルやカラムをでっち上げないこと。
4. エンドポイントのハンドラ関数は、すべて async で自己完結していなければならない。
5. SQLAlchemy ORM だけを使うこと(生の SQL は使わない)。
6. ユーザー固有の操作は、該当する場合は暗黙的に user_id=1 で絞り込まなければならない。
7. エンドポイントごとのセッションのライフサイクル:
   - 最初に session = SessionLocal() とする
   - INSERT/UPDATE/DELETE では session.commit() を呼ぶ
   - 返す前に session.close() を呼ぶ
8. プレースホルダは使わず、完全に実行可能なコードを書くこと。ダミーやプレースホルダのエンドポイントは作らないこと。
9. ルートのパスパラメータ(例: /api/products/{{product_id}})は、関数の引数として現れなければならない。
10. コードでは次のものを厳禁とする: コメント、try/except、エラー処理、型以外の検証、HTTPException、JSONResponse、グローバルな例外ハンドラ、同じパスと HTTP メソッドへのルートの重複登録、未定義のモデル・フィールド・テーブルへの参照、スキーマにない FTS テーブルや補助テーブル、レスポンスモデルを組み立てるための動的・内省的な小細工。
11. Pydantic のレスポンスでは、真偽値は 0/1 ではなく本物の bool フィールドにすること。
12. FastAPI の app は、どのルートデコレータよりも前に定義しなければならない。
13. 最後に、環境変数の HOST と PORT を使う uvicorn のエントリポイントを含めること:

    if __name__ == "__main__":
        import uvicorn, os
        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host=host, port=port)

14. 生成する Python ファイルは、構文エラーのない有効な Python {PYTHON_VERSION} でなければならない。Python の予約語を関数の引数名やキーワード引数名に使わないこと。データベースのカラム名や API のフィールド名が Python のキーワード(例: "return"、"class"、"global")である場合は、末尾にアンダースコアを付けた安全な Python の識別子(例: return_)を使い、Column("return", ...) や Field(..., alias="return") で実際のカラム名や JSON フィールドに対応付けること。
15. (パス, HTTP メソッド)の組ごとに、ルート関数はちょうど 1 つでなければならない。同じパスと HTTP メソッドに対して、一時的であっても複数のハンドラを宣言しないこと。


Pydantic モデルの要件(Pydantic v2):
- Pydantic v2 からインポートすること(例: from pydantic import BaseModel, Field, ConfigDict)
- すべてのエンドポイントについて、リクエストとレスポンスのモデルを定義すること
- すべてのフィールドに、description と example の両方を指定した Field を使うこと
- 各ルートデコレータで指定する response_model は、関数が返すものと正確に一致しなければならない
- 各ルートデコレータの response_model 引数は、このファイルで定義した具体的な Pydantic BaseModel のサブクラスへの直接の参照(例: MyResponseModel、List[MyResponseModel])でなければならず、動的に計算・内省した式であってはならない。response_model を生成するために __annotations__、__mro__、.__class__、メタクラスなどの小細工を使わないこと
- エンドポイントの戻り値の型注釈は response_model と一致し(例: MyResponseModel、List[MyResponseModel])、有効な Pydantic のフィールド型でなければならない。Union[...] の戻り値型、Response 型、Union[Response, dict, None] のような混合を使わないこと
- ORM オブジェクトを返す場合は、model_config を使って Pydantic v2 用にモデルを設定すること。例:
```
  class SomeModel(BaseModel):
      model_config = ConfigDict(from_attributes=True)
      ...
```

- 古い Pydantic v1 の Config と orm_mode = True は使わないこと
- 評価できない型注釈の原因になりうる Annotated などの高度な型の小細工は使わないこと
- 標準の typing の型だけを使うこと: int、float、str、bool、Optional[T]、List[T]、Dict[str, T]
- フィールド名は、そのフィールド自身の型注釈の名前と(大文字小文字を区別しても区別しなくても)決して同じにしてはならない。API 仕様やデータベーススキーマでそれが必要な場合は、必ず別の snake_case のフィールド名(例: `schedule_date: date`、`event_datetime: datetime`)を選び、シリアライズ用のエイリアスを設定すること。例: `schedule_date: date = Field(..., serialization_alias="date", ...)`
- Pydantic のフィールド名が、同じモジュールで使われている型名やクラス名と衝突しないようにすること


エージェント向けの拡張(すべてのエンドポイントのデコレータで必須):
- summary: 目的を表す 1 行(80 文字以内)
- description: 1 行(200 文字以内、改行なし)
- tags: 論理的なグループを表す配列(例: ["products"]、["orders"])
- operation_id: 一意な snake_case の識別子
- response_model: 返す値に直接対応する具体的な Pydantic モデルのクラス(または List[Model] のような typing)


コードスタイル(必須):
- app = FastAPI(...) は、どの @app.get/post/put/patch/delete デコレータよりも前に置かなければならない
- 必要に応じて fastapi の Query/Path/Body/Depends を使うこと
- ORM のリレーションに relationship を使うのは、曖昧さがない場合か、SQLAlchemy の AmbiguousForeignKeysError を避けるため foreign_keys を明示した場合だけにすること
- SELECT: session.query(Model).filter(...).all()/first()
- INSERT: session.add(...); session.commit()
- UPDATE: ORM オブジェクトを取得し、属性を変更してから session.commit()
- DELETE: session.delete(obj); session.commit()
- 宣言した response_model に正確に従う Python の dict/list か Pydantic モデルを返すこと
- Python の予約語を属性名、関数の引数、キーワード引数名に使わないこと。スキーマや API がそのような名前を使う場合は、末尾にアンダースコアを付けた安全な Python の名前を使い、Column("name", ...) や Field(..., alias="name") で対応付けること
- コメント、例外ハンドラ、防御的プログラミング、プレースホルダのコード、FastAPI や Pydantic まわりの動的なハックは書かないこと


出力形式(最重要):
- FastAPI アプリの完全で有効な Python ソースコードだけを返さなければならない
- 応答は Python のコードだけで構成しなければならない。Markdown、文章、説明、JSON のラッパー、キー、コードフェンスを含めないこと
- コードを JSON で包まないこと。外側の構造を付け加えないこと。応答そのものがファイルの中身である
- 応答の最初の行は、`import` や `from` などの有効な Python の文でなければならない(例: `import os`、`from fastapi import FastAPI`)
- 改行をエスケープしないこと。.py ファイルに書かれるとおりにコードを出力すること
- Python 以外の文字列(「コードは次のとおりです:」、バッククォート、JSON など)を少しでも含めると、出力は無効になる
- 応答全体が、そのまま .py ファイルとして保存して実行できる、単一の自己完結した Python モジュールでなければならない


念押し:
- 与えられたデータベーススキーマにある実体だけを使うこと。API 仕様は参考にすぎない。API 仕様とデータベーススキーマが矛盾する場合は、データベーススキーマを優先すること。API 仕様どおりに書くとバグになる場合は、必ずデータベーススキーマを正とすること。API 仕様の文面に従うことよりも、コードがエラーなく実行できることを優先すること
- すべてのエンドポイントが、未定義の名前、インポートの欠落、モデルの欠落なしに動くようにすること
- コードが初回実行時に環境変数 DATABASE_PATH で指定された SQLite データベースを作成し、指定したレスポンスモデルですべてのエンドポイントを正しく提供できるようにすること
"""


SAMPLE_DATA_GENERATION_SYSTEM_PROMPT = """あなたは、AI エージェントのタスク実行用の網羅的なテストデータセットを作ることを専門とする、データベースエンジニア兼データ生成の専門家です。エージェントが API 呼び出しを通じて与えられたすべてのユーザータスクを確実に完了できる、現実的で多様なサンプルデータを生成するのがあなたの仕事です。

あなたは次のことを守らなければなりません:
- 与えられたデータベーススキーマに厳密に従う
- スキーマにないテーブル名やカラム名を、でっち上げたり推測したりしない
- ユーザープロンプトで指定された出力 JSON の形式に正確に従う
- 求められた JSON だけを出力し、余分な説明や前後の文章を付けない
"""

SAMPLE_DATA_GENERATION_USER_PROMPT = """簡略版の {scenario_name} について、与えられたすべてのユーザータスクをエージェントが実行できるよう完全に支える、網羅的なサンプルデータを生成してください。

支えるべきユーザータスク:
{tasks_list}

既存のデータベーススキーマ:
{database_schema}


最重要: スキーマ準拠のルール(必ず守ること):
1. 各テーブルの CREATE TABLE 文を注意深く読み、すべてのカラム、既定値、自動採番の挙動を理解すること。
2. INSERT 文を書くときは、テーブル名の後の括弧内に、挿入するカラム名を必ず明示すること。括弧内には、スキーマにある有効なカラム名だけを入れ、値やリテラルを入れてはならない。
3. 列挙するカラム名はすべてスキーマに存在し、定義どおりの綴りでなければならない。カラムをでっち上げたり、短縮したり、複数形にしたり、改名したりしないこと。
4. INSERT 文を生成してよいのは、「既存のデータベーススキーマ」の節にあるテーブルだけである。スキーマにないテーブル名は使わないこと。
5. すべての INSERT 文で、列挙したカラムの数は VALUES(...) 句の値の数と等しくなければならない。
6. 各 INSERT 文を見直すこと。列挙したカラムを数え、値を数え、両者が等しくなければならない。
7. カラムが多い(10 個以上)テーブルでは、列挙したすべてのカラムに VALUES(...) 句の値がちょうど 1 つずつ対応していることを特に注意して確認すること。
8. 特殊なテーブル・仮想テーブル・FTS テーブル・設定用テーブルでは、その CREATE TABLE(または CREATE VIRTUAL TABLE)文で定義されたカラムだけを使うこと。*_id のような外部キーのカラムは、スキーマに明示的に存在しない限り追加しないこと。


データ生成の方針:
上に挙げた各タスクについて、必要なデータを分析し、次のことを保証すること:
1. タスクで言及されるすべての実体がデータベースに存在する
2. タスクを完了するのに必要なリレーションがすべて正しく張られている
3. クエリの結果が、意味のある空でないデータを返す
4. 堅牢なテストのために、エッジケースやバリエーションが網羅されている


必須要件:
1. 与えられたデータベーススキーマに厳密に従うこと。テーブル名・カラム名を正確に使い、カラムの数を正しくすること。新しいテーブルやカラムをでっち上げないこと。
2. タスクを支えるのに必要なすべてのテーブルについて INSERT 文を生成すること。ただし、スキーマに存在するテーブルだけを対象にする。
3. データの整合性を保つこと。外部キーのリレーションと制約を守る。
4. INSERT 文は SQLite の構文に従うこと。
5. 外部キー制約を満たす正しい順序でデータを挿入すること。
6. users テーブルがある場合は、必ず最初のレコードとして id=1 のユーザーを作ること。これが現在認証されているユーザーである。
7. password_hash、token、session など認証用のフィールドは、スキーマにあっても含めないこと。
8. ユーザーが所有するデータ(注文、投稿など)は、大部分を user_id=1 のものとして作ること。
9. 出力の各テーブルについて、そのテーブルの INSERT 文がどのように与えられたユーザータスクを支え、データベーススキーマに準拠しているかを、"reasoning" フィールドで簡潔に(100 語以内で)説明すること。
10. 有効な JSON だけを返し、それ以外の文章は付けないこと。コメント、省略記号("...")を含めたり、JSON をバッククォートやコードフェンスで囲んだりしないこと。
11. "insert_statements" の各要素は、末尾にセミコロンを 1 つだけ付けた単一の SQL INSERT 文の文字列でなければならず、複数の文や、SQL の前後の JSON・文章を含めてはならない。
12. "insert_statements" の要素はすべて素の文字列(SQL 文のみ)でなければならず、オブジェクトや入れ子の JSON 構造であってはならない。


INSERT 文の形式(必須):
- 各 INSERT 文で、挿入するカラム名を必ず明示すること
- 形式: INSERT INTO table_name (col1, col2, ..., colN) VALUES (val1, val2, ..., valN);
- 列挙したカラムの数は、値の数と等しくなければならない
- テーブル名の後の括弧には、カラム名だけを入れ、リテラル値を入れてはならない
- NULL 値には NULL を使うこと(空文字列ではなく)
- 真偽値のカラムには 0 か 1 を使うこと
- 既定値のある任意のカラムや自動採番の主キーは、値を指定して含めるか、カラムと対応する値の両方を INSERT から省くこと


エージェントのタスクを網羅するための要件:
1. 検索・絞り込みのタスク: 検索条件に一致するデータと一致しないデータの両方を、多様に作ること
2. 一覧・取得のタスク: 意味のある結果を返せるよう、複数のレコード(少なくとも 5〜10 件)を作ること
3. 作成(POST)のタスク: 参照されるすべての実体(ユーザー、カテゴリなど)が存在するようにすること
4. 更新(UPDATE/PATCH)のタスク: 変更できる既存のレコードを作ること
5. 削除のタスク: 安全に削除できる、消してもかまわないレコードを作ること
6. 集計のタスク(件数、合計、平均): 意味のある統計になるよう、十分な量のデータを作ること
7. リレーションのタスク: すべての外部キーの参照が有効で、クエリで辿れるようにすること


データ品質の要件:
1. 現実的な値を使うこと(実在する商品名、正しいメールアドレスの形式、現実的な価格など)
2. 時間的な多様性を持たせること(異なる日時のレコード)
3. ステータスのバリエーションを含めること(有効/無効、保留中/完了など)
4. 数値の範囲を網羅すること(低・中・高の価格、数量、評価)
5. テキストのバリエーションを含めること(短い説明と長い説明、異なるカテゴリ)
6. タイムスタンプには ISO 8601 形式を使うこと: YYYY-MM-DD HH:MM:SS または datetime('now', '-N days')
7. 堅牢なテストを支えられるだけのデータ量を作ること


出力形式:
{{
    "tables": [
        {{
            "table_name": "users",
            "reasoning": "これらの INSERT 文がどのように与えられたユーザータスクを支え、データベーススキーマに厳密に準拠しているかの簡潔な説明(100 語以内)。",
            "insert_statements": [
                "INSERT INTO users (id, username, email, full_name, created_at) VALUES (1, 'current_user', 'user@example.com', 'Current User', datetime('now', '-60 days'));",
                "INSERT INTO users (id, username, email, full_name, created_at) VALUES (2, 'other_user', 'other@example.com', 'Other User', datetime('now', '-30 days'));"
            ]
        }},
        {{
            "table_name": "products",
            "reasoning": "これらの INSERT 文がどのように与えられたユーザータスクを支え、データベーススキーマに厳密に準拠しているかの簡潔な説明(100 語以内)。",
            "insert_statements": [
                "INSERT INTO products (name, description, price, user_id, created_at) VALUES ('Product A', 'Description A', 29.99, 1, datetime('now', '-20 days'));",
                "INSERT INTO products (name, description, price, user_id, created_at) VALUES ('Product B', 'Description B', 49.99, 1, datetime('now', '-15 days'));"
            ]
        }},
        ...
    ]
}}
"""


SCENARIO_CATEGORY_LIST = [
    "E-commerce/Marketplace",
    "Booking/Reservation",
    "Social/Community",
    "Task/Project Management",
    "Finance/Banking",
    "Subscription/Membership",
    "Inventory/Catalog",
    "Messaging/Communication",
    "Lists/Collections",
    "Scheduling/Calendar",
    "Forms/Surveys",
    "Settings/Configuration",
    "Healthcare/Medical",
    "Education/Learning",
    "Real Estate/Property",
    "HR/Recruiting",
    "Legal/Compliance",
    "Logistics/Shipping",
    "Food/Restaurant",
    "Entertainment/Gaming",
    "Fitness/Wellness",
    "Travel/Hospitality",
    "Automotive",
    "IoT/Smart Devices",
    "Developer Tools",
    "CRM/Sales",
    "Content Management",
    "Analytics/Reporting",
]


SCENARIO_GENERATION_SYSTEM_PROMPT = """あなたは、API 環境のシミュレーションとデータベースを使ったやり取りに非常に適した Web サイト・アプリ・デジタルプラットフォームを見つける専門家です。

## 重要な原則: データを合成できるか

架空のデータでも、現実的で役に立つものにできるプラットフォームが必要です。

### 合成できるデータ:
- 数値: 気温、価格、数量、評価、座標、株価
- 実体: ユーザー、商品、注文、予約、従業員、機器、センサー
- ステータスの値: 注文状況、フライトの状況、機器の状態
- 短いテキスト: 名前、タイトル、住所、カテゴリ、説明
- タイムスタンプ: 日付、スケジュール、締め切り、履歴の記録
- 地理: 都市、空港、座標(静的なデータ)

### 意味のある形で合成できないデータ:
- 長い記事: ニュースのコンテンツ、ブログ記事、百科事典の項目
- メディア: 動画や音声のコンテンツそのもの(メタデータではなく)
- AI の推論: チャットボットの応答、ML に基づく推薦
- 実在の検索: ランキング付きの検索エンジンの結果

## 高い適性の例(これらを生成する):

| プラットフォームの種類 | 適している理由 | 主な実体(すべて合成できる) |
|---------------|--------------|----------------------------------|
| EC | 注文やレビューの CRUD | 商品、注文、カート、レビュー |
| タスク管理 | タスクの CRUD | タスク、ボード、担当の割り当て |
| 銀行・フィンテック | 取引の CRUD | 口座、取引、振込 |
| 予約 | 予約の CRUD | 掲載物件、予約、空き状況 |
| 気象サービス | 気象データの照会 | 地点、予報、警報、履歴 |
| 航空・旅行 | フライトの検索と予約 | フライト、空港、予約、価格 |
| 株式取引 | 売買とポートフォリオ | 銘柄、売買、ポートフォリオ、価格 |
| IoT・スマートホーム | 機器の制御 | 機器、センサー、計測値、スケジュール |
| フィットネス記録 | トレーニングの記録 | トレーニング、種目、指標、目標 |
| 医療 | 予約の管理 | 患者、予約、記録、処方 |
| 人事・給与 | 従業員の管理 | 従業員、勤怠表、給与 |
| CRM・営業 | 見込み客の管理 | 見込み客、商談、連絡先、活動 |
| 在庫 | 在庫の管理 | 商品、在庫、倉庫 |
| 物流 | 配送の追跡 | 出荷、荷物、ルート、状況 |
| 飲食店・フード | 注文とメニュー | メニュー項目、注文、予約 |
| 不動産 | 物件と内見 | 物件、掲載、内見、申し込み |
| 教育・LMS | 講座と成績 | 講座、受講登録、課題、成績 |
| イベント管理 | イベントとチケット | イベント、チケット、参加者、会場 |

## 低い適性(避ける):

| プラットフォームの種類 | 適していない理由 | 問題点 |
|---------------|------------------|---------|
| ニュースサイト(BBC、CNN) | 記事のコンテンツそのものが商品 | 意味のある記事を合成できない |
| Wikipedia・百科事典 | 記事のコンテンツそのものが商品 | 知識を偽造できない |
| 検索エンジン | 実在の検索インデックスとランキングが必要 | 関連度を模擬できない |
| AI アシスタント(ChatGPT) | 実在の AI の推論が必要 | AI の応答を偽造できない |
| 翻訳(Google 翻訳) | 実在の NLP モデルが必要 | 翻訳を偽造できない |
| コンテンツ中心のプラットフォーム | コンテンツが中心的な価値の場合 | メタデータだけでは足りない |

## 中程度の適性(CRUD の側面に焦点を当てる):

| プラットフォーム | 模擬できるもの | 模擬できないもの |
|----------|----------------------|-------------|
| YouTube | 再生リスト、登録、コメント、チャンネル管理 | 動画そのもの、推薦 |
| Spotify | 再生リスト、ライブラリ、フォロー | 音声そのもの、音楽の発見 |
| Reddit | 投稿(短いもの)、コメント、投票 | 長文の質の高いコンテンツ |

## 網羅すべきカテゴリ(多様性のため):

E-commerce, Booking/Reservation, Task/Project Management, Finance/Banking,
Weather/Environmental, Flight/Travel, Stock/Investment, IoT/Smart Home,
Fitness/Health Tracking, Healthcare/Medical, HR/Recruiting, CRM/Sales,
Inventory/Warehouse, Logistics/Shipping, Restaurant/Food, Real Estate,
Education/LMS, Event/Ticketing, Legal/Contracts, Subscription Management,
Customer Support, Forms/Surveys, Analytics Dashboards, Fleet Management,
Utility Management (electricity, water), Insurance, Pet Services, Automotive

## あなたのタスク:
非常に適性の高い新しいプラットフォームを生成してください。つまり、データを現実的に合成でき、ユーザーが意味のある CRUD 操作を行うプラットフォームです。

## 出力言語:
シナリオは日本語で生成してください。description は必ず日本語で書き、name だけは英語(ASCII の英数字と空白)で書いてください。例として示すシナリオが英語で書かれていても、出力は日本語にしてください。"""


SCENARIO_GENERATION_USER_PROMPT_TEMPLATE = """適した Web サイト・アプリのシナリオの例を {num_examples} 件示します:

{examples}

---

では、上のどの例とも異なる、新しく多様な Web サイト・アプリ・プラットフォームのシナリオを {num_to_generate} 件生成してください。

要件:
1. 各シナリオは API 環境の合成に適していなければならない(CRUD 操作、データベースとのやり取り)
2. 例とは異なるインタラクションパターンや業界を扱うこと
3. 有名なプラットフォーム、ニッチなサービス、B2B ツール、モバイルアプリ、専門的なツールを織り交ぜること
4. 説明では、ユーザーが管理できる実体、使える操作、存在するワークフローを強調すること
5. コンテンツ中心のサイト、リアルタイムデータのサイト、検索エンジン、AI の推論サービスは避けること
6. 各説明は 300〜600 字程度とし、実行できる機能に焦点を当てること
7. name は英語(ASCII の英数字と空白)で書き、description は日本語で書くこと(例が英語でも description は日本語にする)

出力形式(JSON 配列、Markdown のコードフェンスなし):
[
    {{"name": "Platform Name", "description": "CRUD 操作、実体、ワークフローに焦点を当てた説明..."}},
    ...
]

新しいシナリオをちょうど {num_to_generate} 件生成してください:"""


SCENARIO_FOCUSED_GENERATION_USER_PROMPT_TEMPLATE = """適した Web サイト・アプリのシナリオの例を {num_examples} 件示します:

{examples}

---

## 重要: 次の特定のカテゴリに属するシナリオの生成に集中してください:
{focus_categories}

上のカテゴリを重視して、新しく多様な Web サイト・アプリ・プラットフォームのシナリオを {num_to_generate} 件生成してください。

要件:
1. 各シナリオは API 環境の合成に適していなければならない(CRUD 操作、データベースとのやり取り)
2. 上の重点カテゴリを優先しつつ、その中でも多様性を確保すること
3. 各カテゴリの中で異なるサブニッチを含めること(例: EC ならファッション、家電、B2B 卸売、オークション、レンタルマーケットプレイス)
4. 説明では、ユーザーが管理できる実体、使える操作、存在するワークフローを強調すること
5. コンテンツ中心のサイト、リアルタイムデータのサイト、検索エンジン、AI の推論サービスは避けること
6. 各説明は 300〜600 字程度とし、実行できる機能に焦点を当てること
7. 実在するプラットフォーム、または存在しうる現実的なプラットフォームを考えること
8. name は英語(ASCII の英数字と空白)で書き、description は日本語で書くこと(例が英語でも description は日本語にする)

出力形式(JSON 配列、Markdown のコードフェンスなし):
[
    {{"name": "Platform Name", "description": "CRUD 操作、実体、ワークフローに焦点を当てた説明..."}},
    ...
]

新しいシナリオをちょうど {num_to_generate} 件生成してください:"""


SCENARIO_FOCUS_CATEGORIES = [
    ["E-commerce/Marketplace", "Retail", "Auction", "Rental marketplace"],
    ["Booking/Reservation", "Travel", "Hospitality", "Appointments"],
    ["Social/Community", "Forums", "Professional networks", "Dating"],
    ["Task/Project Management", "Collaboration", "Workflow automation"],
    ["Finance/Banking", "Investment", "Cryptocurrency", "Payments"],
    ["Subscription/Membership", "Loyalty programs", "Digital content"],
    ["Inventory/Catalog", "Warehouse management", "Asset tracking"],
    ["Messaging/Communication", "Customer support", "Ticketing systems"],
    ["Lists/Collections", "Bookmarks", "Curation", "Wishlists"],
    ["Scheduling/Calendar", "Event management", "Resource booking"],
    ["Forms/Surveys", "Feedback", "Applications", "Registrations"],
    ["Healthcare/Medical", "Telehealth", "Patient portals", "Pharmacy"],
    ["Education/Learning", "LMS", "Tutoring", "Course platforms"],
    ["Real Estate/Property", "Rentals", "Property management"],
    ["HR/Recruiting", "Job boards", "Applicant tracking", "Payroll"],
    ["Legal/Compliance", "Contract management", "Document signing"],
    ["Logistics/Shipping", "Fleet management", "Delivery tracking"],
    ["Food/Restaurant", "Ordering", "Menu management", "Kitchen operations"],
    ["Entertainment/Gaming", "Streaming", "Game platforms", "Virtual goods"],
    ["IoT/Smart devices", "Home automation", "Device management"],
    ["Fitness/Wellness", "Workout tracking", "Gym management", "Nutrition"],
    ["Pet services", "Veterinary", "Pet sitting", "Pet supplies"],
    ["Automotive", "Car dealership", "Service scheduling", "Parts inventory"],
    ["Non-profit/Charity", "Donations", "Volunteer management", "Fundraising"],
]


SCENARIO_DIVERSITY_CHECK_SYSTEM_PROMPT = """あなたは Web サイト・アプリのシナリオの多様性を確認する担当者です。新しく生成されたシナリオのうち、既存のシナリオや互いに似すぎているものを見つけるのがあなたの仕事です。

2 つのシナリオは、次の場合に似すぎているとみなします:
1. 同じ特定のニッチを対象にしている(例: 2 つのレストラン予約システム)
2. 中心的な機能がほぼ同じである(例: 2 つの汎用的なプロジェクト管理ツール)
3. 意味のある差別化が述べられていない、直接の競合である

2 つのシナリオは、次の場合に十分に異なるとみなします:
1. 異なる業界やユーザー層を対象にしている
2. 主要な機能やワークフローがはっきり異なる
3. 異なる規模を対象にしている(個人向けと企業向け、個人とチーム)"""


SCENARIO_DIVERSITY_CHECK_USER_PROMPT_TEMPLATE = """## 既存のシナリオ(すでにデータセットに含まれているもの):
{existing_scenarios}

## 確認する新しい候補:
{new_candidates}

---

新しい候補それぞれについて、次のどちらかを判定してください:
- "keep": 既存のどのシナリオとも十分に異なる
- "reject": 既存のシナリオと似すぎている(どのシナリオかを示す)

新しい候補どうしの重複も確認してください。

出力形式(JSON オブジェクト、Markdown のコードフェンスなし):
{{
    "decisions": [
        {{"index": 0, "name": "Candidate Name", "decision": "keep" or "reject", "reason": "簡潔な理由", "similar_to": "reject の場合は似ているシナリオの名前、そうでなければ null"}},
        ...
    ],
    "summary": {{
        "total": N,
        "kept": M,
        "rejected": K
    }}
}}"""


SCENARIO_CATEGORY_DIVERSITY_PROMPT = """次の Web サイト・アプリのシナリオの一覧を分析し、どのインタラクションパターンのカテゴリが不足しているかを特定してください。

現在のシナリオ:
{scenarios}

インタラクションパターンのカテゴリ:
1. E-commerce/Marketplace
2. Booking/Reservation
3. Social/Community
4. Task/Project Management
5. Finance/Banking
6. Subscription/Membership
7. Inventory/Catalog
8. Messaging/Communication
9. Lists/Collections
10. Scheduling/Calendar
11. Forms/Surveys
12. Settings/Configuration
13. Healthcare/Medical
14. Education/Learning
15. Real Estate/Property
16. HR/Recruiting
17. Legal/Compliance
18. Logistics/Shipping
19. Food/Restaurant
20. Entertainment/Gaming

出力形式(JSON、Markdown のコードフェンスなし):
{{
    "category_counts": {{"category_name": count, ...}},
    "underrepresented": ["category1", "category2", ...],
    "suggestions": ["X の種類のシナリオをもっと生成する", ...]
}}"""


VERIFIER_SQL_GENERATION_SYSTEM_PROMPT = """あなたは Python と SQL の専門家です。簡略化した API サーバー上でユーザータスクが正しく完了したかを検証するために、SQLite のクエリを使う Python コードを生成するのがあなたの仕事です。UTF-8 でエンコードされた文字列だけを出力し、絵文字や特殊文字は避けてください。reasoning などの説明文は日本語で書いてください。"""

VERIFIER_SQL_GENERATION_USER_PROMPT = """ユーザータスクが完了したかを検証するために、与えられたデータベースから有用な情報を集める、sqlite3 のクエリを使った Python 関数を生成してください。Python 関数の生成に役立つよう、簡略化した API サーバーの環境名、検証するユーザータスク、データベーススキーマ、データベースの初期状態を与えます。

簡略化した API サーバーの名前:
{scenario_name}

検証するユーザータスク:
{user_task}

データベースのダンプ(初期状態。エージェントが何か操作する前):
{db_dump}
--------------------------------

要件:
1. initial_db_path と final_db_path を引数に取る、完全な Python 関数を生成すること。
2. 関数は SQLite データベースに接続してクエリを実行し、別の LLM がタスクの完了を判定するのに役立つ情報を返すこと。
3. sqlite3 の複雑なクエリの組み合わせや Python のロジックを使ってよい。関数は、タスクの完了を判定するのに役立つ情報を含む辞書を返さなければならない。
4. Python 関数とクエリはデータベースのダンプに従うこと。新しいテーブルやカラムをでっち上げないこと。
5. sqlite3 ライブラリを使い、ほかに必要なライブラリがあればインポートすること
6. データベースの状態は 2 つ与えられる: 初期状態と、エージェントがタスクを実行した後の最終状態である。初期状態を最終状態との比較に使ってよい。
7. データベースは決して変更してはならず、読み取りだけを行うこと。
8. Python 関数が、JSON にシリアライズできる辞書を返せるようにすること。たとえば、辞書のキーにタプルを使わず、datetime などの JSON にシリアライズできないオブジェクトを使わないこと。使えるのは string、int、float、bool、list、dict だけである。
9. 次を含む辞書を出力しなければならない:
    - reasoning: この関数がタスクの成功・失敗の検証にどう役立つかの簡潔な説明。関数の実行結果に基づいて別の LLM が判定するための指針も含める。
    - python_code: initial_db_path と final_db_path を引数に取る完全な Python 関数のコード(文字列)
    - function_name: 関数の名前
    - success_criteria: タスクが成功したことを示す、関数の戻り値の期待される内容の説明
    - failure_criteria: タスクが失敗したことを示す、関数の戻り値の期待される内容の説明

構造の例:
```python
def verify_task(initial_db_path: str, final_db_path: str) -> dict:
    import sqlite3
    import any other libraries the function needs

    conn_initial = sqlite3.connect(initial_db_path)
    conn_final = sqlite3.connect(final_db_path)

    # complex logic here
    # ...

    conn_initial.close()
    conn_final.close()

    return {{
        タスクが成功したか失敗したかを別の LLM が判定するのに役立つものであれば、価値のあるデータや情報を何でも返してよい。ただし、この辞書は JSON にシリアライズできるようにし、datetime や bytes など JSON にシリアライズできないオブジェクトは含めないこと。
    }}
```

出力形式(有効な JSON でなければならない。Markdown のコードフェンスなし):
{{
    "reasoning": "この関数がタスクの成功・失敗の検証にどう使えるかの簡潔な説明",
    "python_code": "initial_db_path と final_db_path を引数に取る完全な Python 関数のコード(文字列)",
    "function_name": "verify_task",
    "success_criteria": "タスクが成功したことを示す、関数の戻り値の期待される内容の説明",
    "failure_criteria": "タスクが失敗したことを示す、関数の戻り値の期待される内容の説明"
}}
"""


CODE_VERIFICATION_SYSTEM_PROMPT = """あなたは、データベースの検証とテストの自動化を専門とする Python 開発者です。AI エージェントが与えられたタスクを正しく完了したかを確実に判定できる、Python の検証コードを生成するのがあなたの仕事です。

UTF-8 でエンコードされた文字列だけを出力し、絵文字や特殊文字は避けてください。reasoning などの説明文は日本語で書いてください。

最重要の要件:
1. 検証コードは 100% 決定的で、ルールに基づくものでなければならない。結果を判定できない場合は、{"result": "others"} を返すだけにすること。
2. コードが {"result": "complete"} を返すのは、タスクが完了したと 100% 確信できる場合だけでなければならない
3. それ以外のすべての場合(不確か、部分的な完了、エラー)は {"result": "others"} を返すこと
4. コードは例外を出さずに、エッジケースを適切に扱うこと
5. 防御的プログラミングを行うこと。入力は不正な形式だったり欠けていたりする可能性があるものとする"""

CODE_VERIFICATION_USER_PROMPT = """AI エージェントがタスクを正しく完了したかを直接判定する、Python の検証関数を生成してください。この関数は、追加の LLM による判定がない強化学習で使われます。コードの出力が最終的な判定になります。

## 環境の情報

シナリオ名: {scenario}

検証するユーザータスク:
{task}

データベースのダンプ(初期状態。エージェントが何か操作する前):
{db_dump}

## 関数のシグネチャ

```python
def verify_task_completion(initial_db_path: str, final_db_path: str, final_answer: str | None = None) -> dict:
    \"\"\"
    Verify if the agent successfully completed the task.

    Args:
        initial_db_path: Path to the database BEFORE agent execution
        final_db_path: Path to the database AFTER agent execution
        final_answer: The agent's final text response/summary (may be None)

    Returns:
        {{"result": "complete"}} if 100% certain task was completed successfully
        {{"result": "others"}} for any other case (incomplete, uncertain, error, partial)
    \"\"\"
```

## 検証の方針

### 照会・読み取りのタスク(例: 「〜を探す」「〜を一覧する」「〜を取得する」「〜は何か」):
- final_answer に、求められた情報が含まれているべきである
- 正規表現のパターンで final_answer から重要な情報を取り出し、検証すること
- 取り出したデータをデータベースの内容と突き合わせ、正確さを確かめること
- 回答の形式が求められたものと一致しているかを確認すること

### 変更のタスク(例: 「〜を追加する」「〜を作成する」「〜を更新する」「〜を削除する」):
- initial_db と final_db を比べ、期待される変更を検出すること
- 指定されたレコードが求められたとおりに作成・変更・削除されたかを確かめること
- 変更がタスクの要件と正確に一致しているかを確認すること
- データの整合性(正しい値、リレーションなど)を検証すること

### 組み合わせのタスク(照会 + 操作):
- データベースの変更と final_answer の内容の両方を検証すること
- すべてのサブタスクが完了していることを確かめること

## コードの要件

1. 必要なライブラリはすべて関数の先頭でインポートすること
2. データベースの操作には sqlite3 を使うこと
3. final_answer の正規表現によるパターンマッチには re モジュールを使うこと
4. final_answer が None や空の場合も適切に扱うこと
5. try-except はデータベースの接続エラーにだけ使い、ロジックには使わないこと
6. エッジケースを扱える堅牢なコードにすること
7. {{"result": "complete"}} を返すのは、すべての条件が 100% 確実に満たされている場合だけにすること
8. それ以外のあらゆる状況では {{"result": "others"}} を返すこと

## 出力形式

有効な JSON オブジェクトを出力しなければなりません(Markdown のコードフェンスなし):
{{
    "reasoning": "このタスクを検証するのにどのような確認が必要かの説明",
    "verification_strategy": "query_based" | "modification_based" | "combined",
    "key_checks": ["コードが行う具体的な確認の一覧"],
    "python_code": "完全な Python 関数のコード(文字列)",
    "function_name": "verify_task_completion"
}}

## コードの構造の例

```python
def verify_task_completion(initial_db_path: str, final_db_path: str, final_answer: str | None = None) -> dict:
    import sqlite3
    import re

    # Helper to safely query database
    def safe_query(db_path: str, query: str, params: tuple = ()) -> list:
        try:
            conn = sqlite3.connect(f"file:{{db_path}}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception:
            return []

    # For query tasks: validate final_answer contains expected info
    # For modification tasks: compare database states
    # For combined: do both

    # Example check for modification task:
    # initial_count = len(safe_query(initial_db_path, "SELECT * FROM orders WHERE user_id = ?", (1,)))
    # final_count = len(safe_query(final_db_path, "SELECT * FROM orders WHERE user_id = ?", (1,)))
    # if final_count > initial_count:
    #     return {{"result": "complete"}}

    # Example check for query task:
    # if final_answer and re.search(r"price.*\\$[\\d.]+", final_answer, re.IGNORECASE):
    #     return {{"result": "complete"}}

    return {{"result": "others"}}
```

では、与えられたタスクの検証コードを生成してください。"""
