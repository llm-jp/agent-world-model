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
3. **パーツを適用する** (`generate_variants` / `apply_part`, `compose.py`)
   `--parts` で指定したパーツを、それぞれ元タスクに単独で当てる。パーツを組み合わせることはしない。
   - 適用できない元タスクもあるので、シナリオごとに `num_per_part × 2` 件を試し、成功したものから `num_per_part` 件を採用する。
   - 乱数は (seed, パーツ, シナリオ, タスク) から決まる (`part_rng`) ので、何度実行しても結果は同じ。
4. **LLM の返答を作り直させる** (`refine`, `refine.py`)
   LLM を使うパーツの返答は、使えるものになるまで作り直させる (後述)。
5. **データとして組み立てる** (`finalize`, `draft.py`)
   `Draft` から、タスク文、`TaskMeta` (ツールの見せ方と追加発話のターン)、verifier コードを作る。
6. **`.aug` ファイルに書き出す** (`merge_variants`, `compose.py`)
   入力ファイルには手を加えない。入力と同じ場所に、拡張子の前に `.aug` を付けたファイルを書き出す (`--output_suffix`、`output_path`)。

   | 入力 | 出力 |
   |---|---|
   | `gen_tasks.jsonl` | `gen_tasks.aug.jsonl` |
   | `gen_verifier.pure_code.jsonl` | `gen_verifier.pure_code.aug.jsonl` |
   | `gen_envs.jsonl` / `gen_db.jsonl` / `gen_sample.jsonl` | `gen_envs.aug.jsonl` / `gen_db.aug.jsonl` / `gen_sample.aug.jsonl` |

   どの出力も、元のデータに拡張データを足したデータセット全体になっている。RL はこれを 1 つのデータセットとして読む。
   - `gen_tasks.aug.jsonl`: 各シナリオの `tasks` の末尾に拡張タスクを足し、設定は並びを揃えた `task_meta` に入れる (元タスクの位置は `null`)。
   - verifier ファイル: 元の verifier のあとに、同じ `(scenario, task_idx)` で `"augmented": true` の行として足す。
   - 再実行すると、出力ファイルを作り直す。`--allowed_scenarios` を指定したときは、選ばなかったシナリオの拡張分を前回の出力から引き継ぐ。

## 既存の生成ステージとの関係

拡張データは `.aug` ファイルにだけ書くので、生成ステージを実行し直しても拡張データは変わらない。新しい内容に揃えるには、生成ステージのあとで `awm gen aug` を実行し直す。

入力ファイルに拡張データが含まれていても壊れないよう、次の取り決めにしている。判定の関数は `awm/tools.py` にまとめている。

| 関数 | 中身 | 使う場所 |
|---|---|---|
| `original_task_rows` | 作り直した環境の行を除き、拡張タスクを外した `gen_tasks.jsonl` の行 | `gen db` / `gen sample` / `gen spec` / `gen verifier` / `gen reference` の入力 |
| `with_rebuilt_env_rows` | 出力ファイルにあった作り直した環境の行を、書き直した結果に残す | `gen db` / `gen sample` / `gen env` の出力 |
| `is_rebuilt_env_row` | 行が作り直した環境のものか (`source_scenario` を持つか) | 上の 2 つと `awm gen aug` |

`awm gen aug` も、入力ファイルにある拡張タスクと作り直した環境の行は読み飛ばす。

- **作り直した環境の扱い**: コード・DB・タスク・verifier をまるごと写したものなので、元のシナリオを作り直してもそのまま動く。ただし更新はされない。
- **古い形式のデータ**: 前のバージョンの `task_meta` (`ToolView` の `rename_tools` を含むもの) は読めない。`--allowed_scenarios` で選ばなかったシナリオの前回の出力に古い形式が残っている場合は、警告を出して捨てる (`existing_variants`)。

## パーツ一覧

パーツ名は、ツールの見せ方・タスク文・追加発話を扱うものを `compose.py` の `PARTS` に、名前を変えるものを `env_parts.py` の `ENV_PARTS` に登録する。どちらも `--parts` で指定する。

### ツールの名前を変える (`env_parts.py`)

見せ方を差し替えるのではなく、**環境を作り直す**。サーバーのコードを新しい名前で書き直すので、エージェントに見えるもの (ツール名、説明文、エラー文) に古い名前が残らない。
作り直した環境は新しいシナリオ (`<シナリオ>__<パーツ>`、例: `Shop__tool_rename`) として追加する。報酬は元の verifier。

| パーツ | クラス | 名前の決め方 | LLM |
|---|---|---|---|
| `tool_rename` | `ToolRename` | 機能が伝わる別の名前を LLM が考える。大文字・小文字や区切りだけの変更、英数字と `_` 以外を含む名前、重なる名前は理由付きで送り返す (`rename_problems`) | 使う |
| `tool_rename_random` | `ToolRenameRandom` | 英小文字 8 文字のランダムな名前にする。ツールの区別は説明文だけが頼りになる | 使わない |

どのパーツも、名前を決めたあとはシナリオごとに次の流れで環境を作る (`build_env_variants`)。

1. **確認用のタスクを選ぶ** (`check_tasks`): 参照軌跡のある元タスクを選び、その呼び出しを元のサーバーで再生する (`replay`)。verifier で `complete` になったものを最大 `num_per_part` 件残す。各呼び出しが失敗したかどうかも記録する。
2. **コードを書き直す** (`code_step`)
   - `operation_id="旧名"` は機械的に新しい名前にする (`rename_operation_ids`)。
   - 残り (説明文、エラー文、引数の `alias` など) は、LLM に置き換えの一覧 (`{"edits": [{"old", "new"}]}`) で返させる (`apply_edits`)。コード全体を書かせないので、返答が短く済む。
3. **機械チェック**: どれかに当たれば、理由を付けて作り直させる。
   - 置き換え前の文字列が、コードにちょうど 1 回だけ出てくる。
   - 書き直したコードが構文として正しい。
   - コードの文字列リテラルに、古いツール名が残っていない。
   - サーバーが起動する (起動しなければ、サーバーの出力の末尾を返す)。
   - ツール一覧が、新しいツール名と元の引数名にちょうど一致し、説明文に古いツール名が残っていない (`listing_problems`)。
   - 確認用のタスクの呼び出しを新しい名前で再生すると、各呼び出しの成否が元のサーバーと同じになり、verifier で `complete` になる (`replay_problems`)。
4. **自己チェック**: コードの差分を LLM に見せ、名前以外を変えていないか、名前を指す表現 (表記ゆれを含む) がすべて直っているかを答えさせる。
5. **依頼文を書き換える** (`task_step`): 元の依頼文に古いツール名が出てくれば、LLM が新しい名前に書き換える。機械チェック (古い名前が残らない、新しい名前が入る、参照軌跡の値が残る) と自己チェック (`dialogue_check_prompt`) をかける。出てこなければそのまま使う。
6. **追加する** (`merge_env_variants`): 次の `.aug` ファイルにそのシナリオの行を足す。行には `source_scenario` と `augmentation` を付け、再実行のときに入れ替えられるようにする。
   - `gen_envs.aug.jsonl`: 書き直したコード
   - `gen_db.aug.jsonl` / `gen_sample.aug.jsonl`: 元のシナリオの行をそのまま写す (DB は変えないため)
   - `gen_tasks.aug.jsonl`: 書き換えた依頼文と `task_meta` (`source_scenario` に元のシナリオ)
   - verifier ファイル: 元タスクの verifier

再生用のサーバーとデータベースは `--work_dir` (既定は `gen_tasks.jsonl` と同じ場所の `aug_env_runs/`) に作る。再生は `--max_concurrency` 個まで並行して動かす。

### ツールの見せ方を変える (`tool_parts.py`)

MCP サーバーのコードには手を入れない。`ToolView` (`awm/core/tool_view.py`) がエージェントとサーバーの間に入ってツール一覧の見せ方を変え、呼び出しをサーバーが受け付ける形に戻す (`PresentedTools.resolve`)。
報酬は元の verifier。

| パーツ | クラス | 内容 | LLM |
|---|---|---|---|
| `add_params` | `AddParams` | 使われない任意引数 (`request_id` など) を足す。呼び出し時に取り除く | 使わない |
| `distractor_tools` | `DistractorTools` | 他シナリオのツールを最大 3 個混ぜる。呼ばれたらエラーを返す | 使わない |
| `tool_doc_paraphrase` | `ToolDocParaphrase` | ツールと引数の説明文を言い換える | 使う |

### タスク文を書き換える (`task_parts.py`)

LLM が書き換える。報酬は元の verifier。
参照軌跡の引数値のうちタスク文に書かれているものは、書き換え後も残っているかを機械的に確かめる (`keep_values` / `rewritten_task`, `draft.py`)。

| パーツ | クラス | 内容 |
|---|---|---|
| `task_paraphrase` | `TaskParaphrase` | 意味を変えずに言い換える |
| `task_ambiguous` | `TaskAmbiguous` | ぼかした砕けた表現にする |
| `task_no_tool_names` | `TaskNoToolNames` | 文中のツール名を消し、日常的なことばで書く |
| `task_distractor_names` | `TaskDistractorNames` | 正解では使わないツール名を文中に混ぜる |

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

パーツは結果を `Draft` (`draft.py`) のフィールドに書き、`finalize` がそれをデータに変える。
タスク文の書き換えでは、`rewritten_task` が次のフィールドを確かめる。

| フィールド | 意味 | 入れるパーツ |
|---|---|---|
| `view` | ツールの見せ方 (`ToolView`) | ツールの見せ方を変えるパーツ |
| `removed_value` | 消した値。書き換えた文に残してはいけない | `miss_param` |
| `required_mentions` | 書き換えた文に入れるべき文言 | `task_distractor_names` |
| `banned_mentions` | 出してはいけないツール名 | `task_no_tool_names` |
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
| `task_meta` を読み、ツールの見せ方を当てはめる | `awm/core/agent.py` (`PresentedTools`) |
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
    --parts "tool_rename_random,task_paraphrase,miss_param" \
    --num_per_part 2 \
    --max_generations 3
```

Slurm では `scripts/run_gen_aug.sh` で vLLM サーバーごと両方の段階を動かす (`-p` でパーツを指定、`-n 20` で `debug_aug/` に試し実行)。

## テスト

`tests/augment/` と、次の 2 つ。

- `tests/test_augmented_data.py`: 既存の生成ステージとの取り決め (`original_task_rows` など)
- `tests/test_server_process.py`: サーバーの起動と停止 (`serve_scenario`)

| ファイル | 対象 |
|---|---|
| `test_parts.py` | 各パーツ |
| `test_draft.py` | 機械チェックと verifier の組み立て |
| `test_compose.py` | パーツの適用とデータへの追記 |
| `test_run.py` | `.aug` ファイルへの書き出し、入力が変わらないこと、`--allowed_scenarios` での引き継ぎ |
| `test_refine.py` | フィードバックと自己チェックの往復 |
| `test_env_parts.py` | 名前の決め方、コードの置き換え、環境の追加。小さな FastAPI サーバーを実際に起動して、再生と機械チェックも確かめる |
