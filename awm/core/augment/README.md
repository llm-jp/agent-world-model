# タスク拡張 (`awm gen aug`)

合成済みのタスクから、ツール利用の頑健性を鍛える派生タスクを作る。
報酬は LLM 判定を使わず、code モードの verifier だけで出す。

拡張タスクの正解の振る舞いは 2 種類ある。

- **見た目だけ変える**: ツールの見せ方・名前やタスク文を変えても、DB に起きるべき変化は元タスクと同じ。元の verifier をそのまま使う。
- **すぐ実行させない**: 1 ターン目はツールか情報が足りない。エージェントは DB を変えずにユーザーへ聞き返し、2 ターン目の追加発話を受けてからタスクをこなすのが正解。元の verifier を包んだ verifier を使う。

## 処理の流れ

1. **参照軌跡を作る** (`awm gen reference`, `awm/core/reference.py`)
   元タスクを一度解かせ、verifier で `complete` になった実行のツール呼び出し列を `gen_reference.jsonl` に、シナリオごとのツール一覧を `gen_tools.jsonl` に保存する。
   どのツールを隠すか、どの値を抜くかはこの軌跡から決める。軌跡が要るパーツでは、解けなかったタスクは対象外になる。
2. **出発点を作る** (`load_drafts`, `__init__.py`)
   verifier のある元タスクごとに `Draft` を作る。中身は次のとおり。
   - タスク文
   - 元の verifier
   - ツール一覧
   - 参照軌跡
   - 他シナリオのツール (ダミーツールの候補)
3. **レシピを適用する** (`run_recipe`, `recipe.py`)
   `--parts` で指定したレシピ (1 つのパーツか、`+` でつないだパーツの列) ごとに、パーツを順に当てる (後述)。
   - 適用できない元タスクもあるので、シナリオごとに `num_per_part × 2` 件を試す (`source_drafts`)。最後のパーツまで残ったものを元タスクの番号順に見て、環境の確認 (手順 6) を通ったものから `num_per_part` 件を採用する (`served_envs`)。確認で落ちたタスクの分は、次のタスクで埋める。
   - 試すタスクの選び方は (seed, レシピ, シナリオ)、各パーツの乱数は (seed, パーツ, シナリオ, タスク) から決まる (`PartContext.rng`) ので、何度実行しても結果は同じ。
4. **LLM の返答を作り直させる** (`refine`, `refine.py`)
   LLM を使うパーツの返答は、使えるものになるまで作り直させる (後述)。
5. **データとして組み立てる** (`finalize`, `draft.py`)
   最後のパーツが返した `Draft` から、タスク文、`TaskMeta` (適用したパーツ、追加発話まで隠すツール `hidden_tools`、追加発話のターンとそこで見せるツール `turns[].shown_tools`)、verifier コードを作る。ツールの見せ方は `TaskMeta` には入れず、環境のサーバーのコードに組み込む (次の手順)。
6. **見せ方をサーバーに組み込む** (`run_recipe` / `bake_view`, `tool_view.py`)
   レシピがツールの見せ方を変えるとき、環境のサーバーのコードに見せ方を組み込む。外部の RL トレーナーなど、`task_meta` を読まない MCP クライアントにも拡張後のツールが見えるようにするため (後述の「ツールの見せ方を変える」)。組み込んだサーバーを起動し、見せ方で足したツールと引数、言い換えた説明文を一覧すること (`view_listing_problems`)、見せ方に合わせた参照軌跡の呼び出しでタスクが解けることを確かめ、満たさない環境は捨てる (`view_problems`)。見せ方なしでも解けないタスクは見せ方のせいにしない。見せ方なしで解けるかは元のサーバーでの再生結果で判断し、その結果は実行中は使い回す (`source_call_errors`, `replay.py`)。
7. **タスクごとの環境として別のディレクトリに書き出す** (`run_recipe` / `merge_env_variants`, `dataset.py`)
   入力ファイルには手を加えない。`--output_dir` (既定は入力のディレクトリ名に `_aug` を付けたもの。例: `outputs/<model>_ja` → `outputs/<model>_ja_aug`) に、入力と同じ名前で書き出す (`output_path`)。
   - 書き出すファイル: `gen_tasks.jsonl` / verifier ファイル (`gen_verifier.pure_code.jsonl`) / `gen_envs.jsonl` / `gen_db.jsonl` / `gen_sample.jsonl` / `gen_scenario.jsonl` (`--scenario_path`、既定は `--envs_path` と同じディレクトリのもの)
   - 出力先が入力と同じファイルになる場合は、実行前にエラーにする。

   出力ディレクトリの中身は、元のデータに拡張データを足したデータセット全体になっている。RL はこれを 1 つのデータセットとして読む。
   - 拡張タスクは 1 件ごとに独立した環境 `<シナリオ>__<レシピ>__<元タスクの番号>` (例: `Shop__add_params__3`、`Shop__tool_rename+miss_param__0`) にする。見せ方はタスクごとに違うので、環境を共有させると、どのタスクのための環境か見分けにくくなるため。環境のタスクは 1 件 (`task_idx` 0) で、元シナリオの行には手を加えない。
   - `gen_envs.jsonl` / `gen_db.jsonl` / `gen_sample.jsonl`: 元シナリオの行を写し、`scenario` を新しい名前にして `source_scenario` と `augmentation` (レシピ名) を付ける。DB ファイルはシナリオ名から決まる (`<正規化したシナリオ名>.db`) ので、元シナリオとは別のファイルになる。サーバーのコードは、名前を変えるパーツを含むレシピでは書き直したもの、それ以外では元のもの。見せ方を変えるレシピでは、そのコードに見せ方を組み込んだもの。
   - `gen_tasks.jsonl`: 新しいシナリオの行に拡張タスクを置き、設定 (追加発話まで隠すツール、追加発話のターン、`source_scenario`) を `task_meta` に入れる。見せ方を `tool_view` に持つ以前の形式の `task_meta` は読み込み時にエラーにする (作り直しが必要)。
   - `gen_scenario.jsonl`: 元シナリオの行を写し、`name` を新しい名前にして `source_scenario` と `augmentation` を付ける。OpenEnv はこのファイルからシナリオの一覧を作るので、ここにない環境は一覧に出ない。
   - verifier ファイル: 新しいシナリオの `task_idx` 0 から、`"augmented": true` と `source_scenario` / `source_task_idx` を付けた行として足す。
   - 正規化したシナリオ名が既存のシナリオと重なる (DB ファイルを共有してしまう) 場合はエラーにする。
   - 再実行すると、出力ファイルを作り直す。`--allowed_scenarios` を指定したときは、選ばなかったシナリオから作った環境を前回の出力から引き継ぐ。

## 既存の生成ステージとの関係

拡張データは出力ディレクトリにだけ書くので、生成ステージを実行し直しても拡張データは変わらない。新しい内容に揃えるには、生成ステージのあとで `awm gen aug` を実行し直す。

入力ファイルに拡張データが含まれていても壊れないよう、次の取り決めにしている。判定の関数は `awm/tools.py` にまとめている。

| 関数 | 中身 | 使う場所 |
|---|---|---|
| `original_task_rows` | 拡張環境の行を除き、`task_meta` 付きの拡張タスクを外した `gen_tasks.jsonl` の行 | `gen db` / `gen sample` / `gen spec` / `gen verifier` / `gen reference` の入力 |
| `with_augmented_env_rows` | 出力ファイルにあった拡張環境の行を、書き直した結果に残す | `gen db` / `gen sample` / `gen env` の出力 |
| `is_augmented_env_row` | 行が拡張環境のものか (`source_scenario` を持つか) | 上の 2 つと `awm gen aug` |

`awm gen aug` も、入力ファイルにある拡張タスクと拡張環境の行は読み飛ばす。

- **拡張環境の扱い**: コード・DB・タスク・verifier をまるごと写したものなので、元のシナリオを作り直してもそのまま動く。ただし更新はされない。
- **元シナリオの行に足された拡張タスク**: `task_meta` 付きで元シナリオの `tasks` の末尾に並んだ拡張タスクは、どのステージも読み飛ばす。`--allowed_scenarios` で再実行しても引き継がない。

## パーツの共通インターフェース (`part.py`)

どのパーツも `Part` を継承し、`apply(drafts, ctx) -> drafts` を実装する。受け取った `Draft` の束を変えて返し、扱えないものは落とす。LLM の依頼は束ごとに 1 バッチで送る。

| クラス | 単位 | サブクラスが実装するもの | パーツ |
|---|---|---|---|
| `TaskPart` | タスクごと | `apply_one(draft, rng)`: 変えた `Draft`、それを作る `LLMStep`、扱えなければ `None` | ツールの見せ方・タスク文・追加発話 |
| `EnvPart` (`env_parts.py`) | シナリオごと | `renames(source, rng)`: 新しい名前、それを LLM に聞く `LLMStep`、変える名前がなければ `None` | ツール名・引数名の変更 |

クラス属性で、パーツの性質を宣言する。
- `needs_reference` / `needs_tools`: 参照軌跡やツール一覧のないタスクを除く (`applicable`)
- `writes`: `Draft` に書くもののうち、同じレシピのほかのパーツと重なってはいけないもの (`ToolView` の各フィールド、追加発話、追加発話まで隠すツール、タスク文で触れるツール名、変えるツール名・引数名)
- `rebuilds_env`: サーバーのコードを書き直すか (`EnvPart` は真)

`PartContext` は、乱数 (`rng`)、LLM の実行 (`run_llm`、本番は `refine`)、再生の設定 (`ReplayContext`)、元シナリオのサーバーのコードを渡す。

パーツは `recipe.py` の `PARTS` に登録する。

## レシピ (`recipe.py`)

`--parts` はカンマ区切りのレシピを受け取る。レシピは 1 つのパーツか、`+` でつないだパーツの列 (例: `tool_rename+task_ambiguous+miss_param`)。パーツは並べた順に当て、各パーツは前のパーツが返した `Draft` を受け取る。

実行前に `parse_recipe` が次を確かめ、当てはまればエラーにする (`recipe_problems`)。

| 規則 | 理由 |
|---|---|
| 環境を作り直すパーツ (`rebuilds_env`) は、ほかのパーツより前に置く | ツールの見せ方やタスク文は、その時点のツール名・引数名を参照する。あとから名前を変えると参照先がなくなる |
| 同じパーツを 2 回使わない | |
| `writes` が重なるパーツを組み合わせない | 例: `miss_func` と `miss_param` はどちらも追加発話を書く。`task_no_tool_names` と `task_distractor_names` はツール名を消すか入れるかで逆の条件になる |

パーツをつないでも前のパーツの結果が消えないよう、次のように扱う。
- **ツールの見せ方**: `ToolView` の、自分の `writes` にあるフィールドだけを足す (`presenting`)。`param_enum` と `param_format` は、もう一方が使っている引数を選ばない。`miss_func` が隠すツールは見せ方とは別に持ち (`Draft.hidden_tool`)、1 ターン目だけ隠す。
- **タスク文の条件**: 書き換えた文に入れるべき文言 (`required_mentions`)、出してはいけないツール名 (`banned_mentions`)、表記を変えた値 (`value_rewrites`) は上書きせずに積み重ねる。後のパーツの書き換えも、これらを守っているかを確かめる (`constraint_problems`)。LLM に渡す「そのまま残す値」の一覧にも含める (`must_keep`)。`task_no_tool_names` は、前のパーツが入れるよう求めたツール名でも、文中に出てくれば消す。
- **環境**: 名前を変えたパーツのあとのパーツは、書き直したコード・ツール一覧・新しい名前の参照軌跡を持つ `Draft` を受け取る。2 つ目の名前を変えるパーツは、1 つ目が書き直したコードをさらに書き直す。

## パーツ一覧

### ツールや引数の名前を変える (`env_parts.py`)

見せ方を差し替えるのではなく、**環境を作り直す**。サーバーのコードを新しい名前で書き直すので、エージェントに見えるもの (ツール名、説明文、エラー文) に古い名前が残らない。
作り直したサーバーのコードは、そのシナリオから作るタスクごとの環境 (`<シナリオ>__<レシピ>__<元タスクの番号>`、例: `Shop__tool_rename__3`) がそれぞれ写して持つ。報酬は元の verifier。

| パーツ | クラス | 名前の決め方 | LLM |
|---|---|---|---|
| `tool_rename` | `ToolRename` | 機能が伝わる別の名前を LLM が考える。大文字・小文字や区切りだけの変更、英数字と `_` 以外を含む名前、重なる名前は理由付きで送り返す (`rename_problems`) | 使う |
| `tool_rename_random` | `ToolRenameRandom` | 英小文字 4〜14 文字のランダムな名前にする (長さもツールごとにランダム。元のコードに単語として出てくる名前は引き直す)。ツールの区別は説明文だけが頼りになる | 使わない |
| `param_rename` | `ParamRename` | 解けた元タスクの参照軌跡が渡す引数に、意味が伝わる別の名前を LLM が考える。同じ名前の引数はどのツールでも同じ新しい名前にする。大文字・小文字や区切りだけの変更、使えない名前、Python の予約語、重なる名前、そのツールの既存の引数名と同じ名前は送り返す (`param_rename_problems`)。サーバーは `alias=` などで新しい名前だけを受け付けるように書き直す | 使う |

どのパーツも、受け取った `Draft` をシナリオごとにまとめ、次の流れで環境を作る (`EnvPart.apply`)。

1. **確認用のタスクを選ぶ** (`scenario_sources` / `replayed_drafts`, `replay.py`): 受け取った `Draft` の呼び出しを、そのシナリオのサーバーで再生する (`replay`)。verifier で `complete` になったものをすべて残す (あとのパーツが選べるように、ここでは `num_per_part` 件に絞らない)。各呼び出しが失敗したかどうかを `Draft.call_errors` に記録する。元のサーバーでの再生結果は実行中は使い回す。前のパーツが書き直したサーバーの `Draft` は、書き直しの確認で再生済みなので再生しない。
2. **コードを書き直す** (`code_step`)
   - `operation_id="旧名"` は機械的に新しい名前にする (`rename_operation_ids`)。
   - 残り (説明文、エラー文、引数の `alias` など) は、LLM に置き換えの一覧 (`{"edits": [{"old", "new"}]}`) で返させる (`apply_edits`)。コード全体を書かせないので、返答が短く済む。
3. **機械チェック**: どれかに当たれば、理由を付けて作り直させる。
   - 置き換え前の文字列が、コードにちょうど 1 回だけ出てくる。
   - 書き直したコードが構文として正しい。
   - コードの文字列リテラルに、古いツール名が残っていない。
   - 名前以外を変えていない (`unrelated_change_problems`)。元のコードと書き換え後のコードの両方で、旧名・新名の言及 (大文字・小文字の違いや、`_` を空白にした形を含む) を新しい名前に揃え、関数名を除いて比べる。差が残れば、その行を返す。名前の一部の単語 (例: `create_order` に対する "Order total" の "order") を言い換えると、ここで弾かれる。
   - サーバーが起動する (起動しなければ、サーバーの出力の末尾を返す)。
   - ツール一覧が、新しいツール名と元の引数名にちょうど一致し、説明文に古いツール名が残っていない (`listing_problems`)。
   - 確認用のタスクの呼び出しを新しい名前で再生すると、各呼び出しの成否が元のサーバーと同じになり、verifier で `complete` になる (`replay_problems`)。
4. **自己チェック**: コードの差分を LLM に見せ、名前以外を変えていないか、名前を指す表現 (表記ゆれを含む) がすべて直っているかを答えさせる。
5. **依頼文を書き換える** (`task_step`): 元の依頼文に古いツール名が出てくれば、LLM が新しい名前に書き換える。機械チェック (古い名前が残らない、新しい名前が入る、参照軌跡の値が残る) と自己チェック (`dialogue_check_prompt`) をかける。出てこなければそのまま使う。
6. **次のパーツに渡す**: 書き直したコード、サーバーが一覧するツール、新しい名前の参照軌跡を `Draft` に入れる。レシピの最後なら、ほかのパーツと同じく出力ディレクトリの次のファイルにそのシナリオの行を足す (`merge_env_variants`, `dataset.py`)。
   - `gen_envs.jsonl`: 書き直したコード
   - `gen_db.jsonl` / `gen_sample.jsonl`: 元のシナリオの行をそのまま写す (DB は変えないため)
   - `gen_tasks.jsonl`: 書き換えた依頼文と `task_meta` (`source_scenario` に元のシナリオ)
   - verifier ファイル: 元タスクの verifier

再生用のサーバーとデータベースは `--work_dir` (既定は出力ディレクトリの `aug_env_runs/`) に作り、再生が終わるたびに消す。再生は `--max_concurrency` 個まで並行して動かす。

### ツールの見せ方を変える (`tool_parts.py`)

見せ方 (`ToolView`, `awm/core/tool_view.py`) は、環境のサーバーのコードに組み込む (`bake_view`)。外部の RL トレーナー (OpenEnv など) は `gen_envs.jsonl` の `full_code` を起動して MCP でつなぐだけで `task_meta` を読まないので、サーバー自身が見せ方どおりにツールを一覧する必要がある。
報酬は元の verifier。

- 見せ方はタスクごとに決め、タスクごとの環境のサーバーに組み込む。
- 組み込むのは `awm/core/view_server.py` のソースと見せ方の JSON。`awm` を import せず FastAPI と Python 3.10 だけで動くので、`awm` を入れていないトレーナーの環境でも動く。`if __name__ == "__main__":` の前に置き、関数名 `_install_tool_view` を 1 つだけ足す。
- サーバーを MCP で公開する層 (`awm/core/mcp_bridge`、OpenEnv が使う fastapi-mcp) は、どちらも `fastapi.openapi.utils.get_openapi` が作る OpenAPI 文書からツールを作る。組み込んだコードはこの関数を包み、文書に見せ方を当てはめる。説明文は `### Responses:` より前を差し替え、レスポンスの説明はサーバーが付け直す。
- 呼び出しは、追加引数を宣言のないクエリ引数として受けて無視し、偽ツールには専用のルートでエラーを返し、別の形式の値はミドルウェアで元の形式に戻す (形式が合わなければ期待する形式を添えて 422)。
- `miss_func` が追加発話まで隠すツールはターンで変わるので組み込めず、`task_meta` に残す (`awm agent` の `PresentedTools` が隠す)。見せ方の実装はサーバー側 (`view_server.py`) だけにあり、`PresentedTools` は隠すことしかしない。追加発話のターンも同じく、`task_meta` を読むクライアントでしか再現できない。

| パーツ | クラス | 内容 | LLM |
|---|---|---|---|
| `add_params` | `AddParams` | 使われない任意引数 (`request_id` など) を、ツールごとに 2〜3 個足す。呼び出し時に取り除く | 使わない |
| `distractor_tools` | `DistractorTools` | 他シナリオのツールを最大 3 個混ぜる。呼ばれたらエラーを返す | 使わない |
| `tool_doc_paraphrase` | `ToolDocParaphrase` | ツールと引数の説明文を言い換える。サーバーが付けるレスポンスの説明 (`### Responses:` 以降) は言い換えない | 使う |
| `param_enum` | `ParamEnum` | 取りうる値が少数に決まっている文字列の引数 (ステータス、種別、並び順など) を LLM が選び、値の一覧 (`enum`) を付ける。自由な文字列や ID は選ばない。参照軌跡で渡した値が一覧に入っていること、値が 2〜12 個であることを機械的に確かめる。選べる引数がなければ、作り直しの上限に達してそのタスクは捨てられる | 使う |
| `param_format` | `ParamFormat` | 参照軌跡で渡す引数のうち最大 2 個を、別の形式で見せる (日付 `YYYYMMDD` / `YYYY/MM/DD`、真偽値 `"yes"`/`"no"`、整数を数字の文字列)。呼び出し時に元の形式へ戻し、形式が合わなければ期待する形式を添えてエラーを返す (`VALUE_FORMATS`, `view_server.py`) | 使わない |

### タスク文を書き換える (`task_parts.py`)

LLM が書き換える。報酬は元の verifier。
参照軌跡の引数値のうちタスク文に書かれているものは、書き換え後も残っているかを機械的に確かめる (`keep_values` / `rewritten_task`, `draft.py`)。

| パーツ | クラス | 内容 |
|---|---|---|
| `task_ambiguous` | `TaskAmbiguous` | 要点をぼかした表現にする。日本語では「です・ます」調の丁寧な書き言葉、英語ではくだけた口調 |
| `task_no_tool_names` | `TaskNoToolNames` | 文中のツール名を消し、日常的なことばで書く |
| `task_distractor_names` | `TaskDistractorNames` | 正解では使わないツール名を文中に混ぜる |
| `task_value_format` | `TaskValueFormat` | 文中の値の書き方を変える (例: 「2026-05-20」→「2026 年 5 月 20 日」)。ID・コード・名前はそのまま。書き換えた値の元の表記が残らないこと、新しい表記が文中にあること、ほかの値が残っていることを確かめる。自己チェックには書き換えた値の対応を渡し、表記が値をちょうど表しているかを確かめさせる |

### ユーザーの追加発話を待たせる (`followup_parts.py`)

| パーツ | クラス | 内容 | LLM |
|---|---|---|---|
| `miss_func` | `MissFunc` | 参照軌跡の最後に呼ばれたツールを 1 ターン目だけ隠す。2 ターン目でユーザーが「〇〇が使えるようになった」と伝え、ツールが戻る | 使わない |
| `miss_param` | `MissParam` | タスク文から引数値を 1 つ抜く。2 ターン目でユーザーがその値を伝える。追加発話も LLM が作る | 使う |

報酬は `build_followup_verifier` (`draft.py`) が作る verifier で出す。判定は次の順に進む。

1. 1 ターン目の終わりに DB が初期状態から変わっていたら失敗。
2. 1 ターン目に禁止された呼び出し (`forbidden_calls`) をしていたら失敗。
3. どちらも問題なければ、元の verifier で最終状態を判定する。

## `Draft` のフィールド

パーツは結果を `Draft` (`draft.py`) のフィールドに書き、`finalize` がレシピの最後の `Draft` をデータに変える。
タスク文の書き換えでは、`rewritten_task` / `constraint_problems` が `removed_value`、`required_mentions`、`banned_mentions`、`value_rewrites` を確かめる。

| フィールド | 意味 | 入れるパーツ |
|---|---|---|
| `parts` | ここまでに当てたパーツ (`TaskMeta.augmentations` になる) | すべて (`evolve`) |
| `code` / `call_errors` | 書き直したサーバーのコード、そのサーバーで各参照呼び出しが失敗するか | 名前を変えるパーツ |
| `tools` / `reference_calls` | ツール一覧と参照軌跡。名前を変えるパーツのあとは新しい名前 | 名前を変えるパーツ |
| `view` | ツールの見せ方 (`ToolView`) | ツールの見せ方を変えるパーツ |
| `removed_value` | 消した値。書き換えた文に残してはいけない | `miss_param` |
| `required_mentions` | 書き換えた文に入れるべき文言 | `task_distractor_names`、名前を変えるパーツ |
| `banned_mentions` | 出してはいけないツール名 | `task_no_tool_names`、名前を変えるパーツ |
| `value_rewrites` | 表記を変えた値と、その表記。表記は書き換えた文に残す | `task_value_format` |
| `hidden_tool` / `followup` / `forbidden_calls` | 1 ターン目に隠すツール、2 ターン目の発話、1 ターン目に禁止する呼び出し | `miss_func` / `miss_param` |

## LLM の返答の作り直し (`refine.py`)

LLM を使うパーツは `LLMStep` (`draft.py`) を返す。結果の型は問わず、タスク文の書き換えなら `Draft`、環境の書き直しなら `RebuiltEnv` になる。
`refine` はそれをまとめて送り、使える返答が得られるまで同じ会話で作り直させる。`finish` が重いステップ (サーバーを起動する環境の書き直し) は、`workers` 個まで並行して確かめる。

1. **生成**: 返答を JSON として読み、パーツの `finish` で確かめる。
   - 使えない返答には、理由 (`Rejection`) を付けて作り直させる。例: 「`blue mug` が消えている」「`CreateOrder` は大文字・小文字を変えただけ」。
   - 返答が空だったときは、フィードバックを付けずに同じ依頼を送り直す。
2. **自己チェック** (`check_prompt` を持つステップだけ): 機械チェックを通った返答について、一貫しているかを LLM に答えさせる。
   - 環境の書き直しでは、コードの差分を渡す (前述)。
   - タスク文の書き換え (`task_*`、`miss_param`、名前を変えた環境の依頼文) では、ユーザーの発話が参照軌跡と一貫しているかを答えさせる (`dialogue_check_prompt`, `task_parts.py`)。渡す材料は次のとおり。
     - 1 ターン目の発話と追加発話
     - 期待される振る舞い (1 ターンでこなすのか、1 ターン目は聞き返すのか)
     - 1 ターン目に禁止された呼び出し
     - 参照のツール呼び出し (名前を変えた環境では新しい名前)

   返答は `{"consistent": bool, "problems": [...]}` の形で、`SelfCheckReply` で検証する。
   - 一貫していないと言われたら、`problems` をフィードバックとして作り直させる。
   - 自己チェックの返答が読めなければ、最大 `MAX_CHECK_RETRY` 回まで聞き直す。
3. **打ち切り**: 生成が `Config.max_generations` 回 (既定 3) に達しても採用できなければ、そのステップは捨てる。自己チェックの聞き直しは生成の回数に数えない。

プロンプトは `awm/prompts.py` の `AUG_*` に置く。
- 自然文を出力するプロンプト (タスク文の書き換え、追加発話、そのフィードバックと自己チェック) は、`AWM_SYN_LANG=ja` のとき `awm/prompts_ja.py` の日本語版に置き換わる。
- ツール定義とサーバーのコードは環境の一部なので、それらを扱うプロンプト (`AUG_TOOL_*`、`AUG_ENV_*`) は英語のままにする。

## 実行時の扱い

| 処理 | 場所 |
|---|---|
| ツールの見せ方を当てはめる | 環境のサーバー (`awm/core/view_server.py` を組み込んだもの) |
| `task_meta` を読み、追加発話まで隠すツールを隠し、そのターンで見せる | `awm/core/agent.py` (`PresentedTools`) |
| 追加発話のターンを送る | `awm/core/agent.py` |
| ターンごとの DB スナップショットとツール呼び出しを verifier に渡す | `awm/core/verify.py` (`run_verifier`) |

## 使い方

```bash
awm gen reference \
    --input_task outputs/gen_tasks.jsonl \
    --input_verifier outputs/gen_verifier.pure_code.jsonl \
    --output outputs/gen_reference.jsonl \
    --output_tools outputs/gen_tools.jsonl

awm gen aug \
    --input_task outputs/gen_tasks.jsonl \
    --input_verifier outputs/gen_verifier.pure_code.jsonl \
    --input_reference outputs/gen_reference.jsonl \
    --input_tools outputs/gen_tools.jsonl \
    --envs_path outputs/gen_envs.jsonl \
    --db_path outputs/gen_db.jsonl \
    --sample_path outputs/gen_sample.jsonl \
    --output_dir outputs_aug \
    --parts "tool_rename_random,task_ambiguous,tool_rename+task_ambiguous+miss_param" \
    --num_per_part 2 \
    --max_generations 3
```

Slurm では `scripts/run_gen_aug.sh` で vLLM サーバーごと両方の段階を動かす。出力先は `-o` で指定したディレクトリ名に `_aug` を付けたものになる (`-p` でレシピを指定、`-n 20` で `<dir>_aug_debug/` に試し実行)。

## テスト

`tests/augment/` と、次の 3 つ。

- `tests/test_augmented_data.py`: 既存の生成ステージとの取り決め (`original_task_rows` など)
- `tests/test_server_process.py`: サーバーの起動と停止 (`serve_scenario`)
- `tests/test_view_server.py`: 見せ方を組み込んだサーバーの一覧と呼び出し (サーバーはプロセス内で読み込む)

| ファイル | 対象 |
|---|---|
| `test_parts.py` | 各パーツ。前のパーツの結果を引き継ぐこと |
| `test_draft.py` | 機械チェックと verifier の組み立て |
| `test_recipe.py` | レシピの検証と適用、パーツをつないだ結果、見せ方の組み込みとその確かめ方 |
| `test_dataset.py` | 環境へのまとめ方と、シナリオファイル・verifier への追加 |
| `test_run.py` | 出力ディレクトリへの書き出し、入力が変わらないこと、`--allowed_scenarios` での引き継ぎ、つないだレシピの書き出し |
| `test_refine.py` | フィードバックと自己チェックの往復 |
| `test_env_parts.py` | 名前の決め方、コードの置き換え、依頼文の書き換え、確認用タスクの選び方。小さな FastAPI サーバーを実際に起動して、再生と機械チェックも確かめる |
