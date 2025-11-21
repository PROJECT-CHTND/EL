# nani_bot.py - 相手の考えを引き出す汎用Bot

import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import openai
import json
from datetime import datetime
from pathlib import Path
import sys
import yaml  # type: ignore

# --- Pipeline modules ---
from agent.pipeline.stage02_extract import extract_knowledge
from agent.pipeline.stage03_merge import merge_and_persist
from agent.pipeline.stage04_slots import propose_slots
from agent.pipeline.stage05_gap import analyze_gaps
from agent.pipeline.stage06_qgen import generate_questions
from agent.pipeline.stage07_qcheck import return_validated_questions
from agent.pipeline.runner import run_turn
from agent.models.question import Question
from agent.models.kg import KGPayload
from agent.slots import Slot, SlotRegistry
from agent.stores.sqlite_store import SqliteSessionRepository
from agent.slots.postmortem import build_postmortem_registry, fallback_question

import asyncio
import io
from typing import List, Dict, Any, Optional
from openai.types.chat import ChatCompletionMessageParam

# プロンプトテンプレートをインポート
from prompts import BILINGUAL_PROMPT_TEMPLATE, STYLE_GUIDES, RAG_EXTRACTION_PROMPT_TEMPLATE

# --- el-agent (v2 core) integration ---
# 検索・戦略・評価をフル活用するために el-agent を動的に取り込み
EL_AGENT_SRC = Path(__file__).parent / "el-agent" / "src"
if EL_AGENT_SRC.exists():
    sys.path.insert(0, str(EL_AGENT_SRC))

# 外部プランナー/記者プロンプト設定（デフォルト）
PLANNERS_FILE_DEFAULT = Path(__file__).parent / "agent" / "prompts" / "planners.yaml"
JOURNALIST_FILE_DEFAULT = Path(__file__).parent / "agent" / "prompts" / "journalist.yaml"

try:
    from el_agent.core.strategist import Strategist  # type: ignore
    from el_agent.core.knowledge_integrator import KnowledgeIntegrator  # type: ignore
    from el_agent.core.evaluator import Evaluator, update_belief  # type: ignore
    from el_agent.schemas import Hypothesis, Evidence  # type: ignore
    EL_AGENT_AVAILABLE = True
except Exception:
    EL_AGENT_AVAILABLE = False

# 環境変数読み込み
load_dotenv()

# OpenAI設定（テスト互換の軽量シム）
class _ShimCompletions:
    def __init__(self, parent: "_ShimClient"):
        self._parent = parent

    async def create(self, *args, **kwargs):
        # 実行時にだけ本物のクライアントへ委譲（テストでは monkeypatch で差し替え）
        if self._parent._real is None:
            self._parent._real = openai.AsyncOpenAI(api_key=self._parent._api_key)
        return await self._parent._real.chat.completions.create(*args, **kwargs)


class _ShimChat:
    def __init__(self, parent: "_ShimClient"):
        self.completions = _ShimCompletions(parent)


class _ShimClient:
    def __init__(self, api_key: str | None):
        self._api_key = api_key or ""
        self._real = None  # 遅延初期化
        self.chat = _ShimChat(self)  # 安定オブジェクト


client = _ShimClient(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# OpenAI呼び出しの安全ラッパ（temperature非対応モデルへのフォールバック）
async def safe_chat_completion(*, messages, temperature=None, response_format=None, **extra):
    kwargs = {"model": OPENAI_MODEL, "messages": messages, **extra}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as e:
        msg = str(e)
        # 一部モデル(gpt-5系など)は temperature 固定のため温度を外して再試行
        if ("Unsupported value" in msg or "unsupported_value" in msg) and "temperature" in msg:
            kwargs.pop("temperature", None)
            return await client.chat.completions.create(**kwargs)
        raise

# el-agent の必須化フラグ（デフォルト: 必須）
EL_AGENT_REQUIRED = os.getenv("EL_AGENT_REQUIRED", "1") == "1"
# 確からしさ表示トグル（デフォルト: 非表示）
SHOW_CONFIDENCE = os.getenv("SHOW_CONFIDENCE", "0") == "1"
# LLMを使ったゴール自動推定/質問リファイン/インタビュー合成
GOAL_CLASSIFIER = os.getenv("GOAL_CLASSIFIER", "1") == "1"
QUESTION_REFINER = os.getenv("QUESTION_REFINER", "1") == "1"
INTERVIEW_MODE = os.getenv("INTERVIEW_MODE", "1") == "1"

# 簡易的な言語検出
def detect_language(text: str) -> str:
    """簡易的な言語検出"""
    # 日本語文字の存在をチェック
    japanese_chars = any(
        '\u3040' <= char <= '\u309f' or  # ひらがな
        '\u30a0' <= char <= '\u30ff' or  # カタカナ
        '\u4e00' <= char <= '\u9faf'     # 漢字
        for char in text
    )
    return "Japanese" if japanese_chars else "English"

class ELBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        self.sessions = {}
        self.session_repo = SqliteSessionRepository()
        
    async def on_ready(self):
        print(f'🧠 {self.user} - EL has started!')
        try:
            await self.session_repo.init()
        except Exception:
            pass
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="to your thoughts"
            )
        )

bot = ELBot()


class AgentRuntime:
    """el-agent の Strategist / KnowledgeIntegrator / Evaluator を束ねるランタイム。
    利用不可の場合は None を保持し、フォールバック経路に委譲する。
    """

    def __init__(self):
        if EL_AGENT_AVAILABLE:
            self.strategist = Strategist()
            self.integrator = KnowledgeIntegrator()
            self.evaluator = Evaluator()
        else:
            self.strategist = None
            self.integrator = None
            self.evaluator = None

    @property
    def available(self) -> bool:
        return self.strategist is not None and self.integrator is not None and self.evaluator is not None


agent_runtime = AgentRuntime()
if EL_AGENT_REQUIRED and not agent_runtime.available:
    print("❌ el-agent が利用できません（EL_AGENT_REQUIRED=1）。el-agent/src の配置や依存を確認してください。")

class ThinkingSession:
    def __init__(self, user_id: str, topic: str, thread_id: int, language: str):
        self.user_id = user_id
        self.topic = topic
        self.thread_id = thread_id
        self.language = language
        self.created_at = datetime.now()
        self.messages: List[Dict[str, Any]] = []
        self.insights: List[Any] = []
        self.phase = "introduction"
        self.last_question: Optional[str] = None
        self.pending_slot: Optional[str] = None
        self.db_session_id: Optional[int] = None
        # el-agent 連携用の状態
        self.hypothesis: Optional["Hypothesis"] = None  # type: ignore[name-defined]
        self.belief: float = 0.5
        self.belief_ci: Optional[tuple[float, float]] = None
        self.last_action: Optional[str] = None
        self.last_supports: List[str] = []
        self.belief_updated: bool = False
        # ゴール指向プランニング
        self.goal_kind: str = infer_goal_kind(topic)
        self.goal_state: Dict[str, Any] = {"asked": set(), "filled": {}}
        self.slot_registry: SlotRegistry = SlotRegistry()
        self.slot_answers: Dict[str, List[str]] = {}
        self.configure_slots(self.goal_kind)

    def add_exchange(self, question: str, answer: str):
        self.messages.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer
        })

    def configure_slots(self, goal_kind: str) -> None:
        self.goal_kind = goal_kind
        if goal_kind == "postmortem":
            self.slot_registry = build_postmortem_registry()
        else:
            self.slot_registry = SlotRegistry()
        self.pending_slot = None
        self.slot_answers = {}

@bot.command(name='explore')
async def start_exploration(ctx, *, topic: Optional[str] = None):
    """思考の探求を開始"""
    if not topic:
        embed = discord.Embed(
            title="🌟 Welcome to the Exploration / 思考の探求へようこそ",
            description=(
                "What would you like to think about? Please provide a topic.\n"
                "何について考えてみましょうか？トピックを指定してください。\n\n"
                "**Examples / 例:**\n"
                "• `!explore Something that moved me recently`\n"
                "• `!explore 人生で大切にしていること`"
            ),
            color=discord.Color.blue()
        )
        await ctx.send(embed=embed)
        return
    
    user_id = str(ctx.author.id)
    
    # スレッド作成
    thread = await ctx.channel.create_thread(
        name=f"💭 {topic[:30]}... - {ctx.author.name}",
        type=discord.ChannelType.public_thread,
        auto_archive_duration=60
    )
    
    # セッション言語をトピックから検出
    session_language = detect_language(topic)
    
    # el-agent が必須かつ未利用可能ならエラー
    if EL_AGENT_REQUIRED and not agent_runtime.available:
        await ctx.send("❌ el-agent が利用できません。`el-agent/src` の配置と依存を確認してください（必要なら EL_AGENT_REQUIRED=0 でフォールバック可）。")
        return
    
    # セッション作成
    session = ThinkingSession(user_id, topic, thread.id, session_language)
    bot.sessions[user_id] = session

    # セッション永続化（SQLite）
    try:
        sid = await bot.session_repo.create_session(
            user_id=user_id,
            topic=topic,
            goal_kind=session.goal_kind,
            created_at_iso=session.created_at.isoformat(),
        )
        session.db_session_id = sid
    except Exception:
        session.db_session_id = None

    # el-agent 初期化（仮説初期値）
    if agent_runtime.available:
        try:
            session.hypothesis = Hypothesis(
                id=f"h-{user_id}",
                text=topic,
                belief=0.5,
                belief_ci=(0.25, 0.75),
                action_cost={"ask": 1.0, "search": 0.5},
                slots=["topic"],
            )  # type: ignore[name-defined]
        except Exception:
            session.hypothesis = None
    
    # 最初の定型メッセージは送らず、以降はスレッド内で会話

    # LLMでゴール種別を推定（任意）
    if GOAL_CLASSIFIER and _PLANNER_SPEC.get("goals"):
        try:
            kinds_csv = ",".join([str(g.get("kind")) for g in _PLANNER_SPEC.get("goals", [])])
            sys_prompt = (_JOURNALIST_SPEC.get("goal_classifier", {}) or {}).get("system_prompt", "")
            if sys_prompt:
                sys_prompt = sys_prompt.replace("{{ kinds_csv }}", kinds_csv)
                resp = await safe_chat_completion(
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": topic}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                _tmp = json.loads(content)
                picked = _tmp.get("kind")
                if isinstance(picked, str) and picked:
                    session.goal_kind = picked
        except Exception:
            pass

    # session.configure_slots(session.goal_kind)  # Removed redundant call

    # 最初の質問
    first_question = await generate_opening_question(topic, session.language, session.goal_kind)
    session.last_question = first_question  # 最初の質問を保存
    if session.goal_kind == "postmortem":
        session.pending_slot = "summary"

    q_embed = discord.Embed(
        description=first_question,
        color=discord.Color.blue()
    )
    author_name = "💭 First Question" if session.language == "English" else "💭 最初の問いかけ"
    q_embed.set_author(name=author_name)
    
    await thread.send(embed=q_embed)
    await thread.add_user(ctx.author)
    # 初回質問を永続化（assistantロール）
    if session.db_session_id is not None:
        try:
            await bot.session_repo.add_message(
                session_id=session.db_session_id,
                ts_iso=datetime.now().isoformat(),
                role="assistant",
                text=first_question,
            )
        except Exception:
            pass

@bot.command(name='reflect')
async def reflect_session(ctx):
    """セッションを振り返る"""
    user_id = str(ctx.author.id)
    
    session = bot.sessions.get(user_id)
    if not session:
        error_msg = "No active session found. / アクティブなセッションがありません。"
        await ctx.send(error_msg)
        return
    
    # 振り返りを生成
    reflection = await generate_reflection(session)
    
    embed = discord.Embed(
        title="🔍 Reflection" if session.language == "English" else "🔍 振り返り",
        description=f"A dialogue about \"**{session.topic}**\"" if session.language == "English" else f"「**{session.topic}**」についての対話",
        color=discord.Color.purple()
    )
    
    # 主な発見
    if reflection.get('key_insights'):
        insights_title = "💎 Key Discoveries" if session.language == "English" else "💎 主な発見"
        insights_text = "\n".join([f"• {i}" for i in reflection['key_insights'][:5]])
        embed.add_field(
            name=insights_title,
            value=insights_text,
            inline=False
        )
    
    # 深まった理解
    if reflection.get('deepened_understanding'):
        understanding_title = "🌊 Deepened Understanding" if session.language == "English" else "🌊 深まった理解"
        embed.add_field(
            name=understanding_title,
            value=reflection['deepened_understanding'],
            inline=False
        )
    
    # さらなる探求
    if reflection.get('future_questions'):
        future_title = "🚀 Potential for Further Exploration" if session.language == "English" else "🚀 さらなる探求の可能性"
        questions_text = "\n".join([f"• {q}" for q in reflection['future_questions'][:3]])
        embed.add_field(
            name=future_title,
            value=questions_text,
            inline=False
        )
    
    footer_text = "May this dialogue be meaningful to you." if session.language == "English" else "この対話があなたにとって意味のあるものでありますように"
    embed.set_footer(text=footer_text)
    
    await ctx.send(embed=embed)

@bot.command(name='finish')
async def finish_session(ctx):
    """セッションを終了し、ログとRAGデータを出力する"""
    user_id = str(ctx.author.id)
    
    session = bot.sessions.get(user_id)
    if not session:
        error_msg = "No active session found. / アクティブなセッションがありません。"
        await ctx.send(error_msg)
        return

    # 処理中メッセージを送信
    thinking_message = "Processing the dialogue and generating files... please wait." if session.language == "English" else "対話を処理し、ファイルを生成中です。少々お待ちください..."
    await ctx.send(thinking_message)

    # 対話の記録を生成 (Markdown)
    record = await create_dialogue_record(session)
    filename_md = f"dialogue_{session.topic.replace(' ', '_').replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    file_md = discord.File(io.BytesIO(record.encode('utf-8')), filename=filename_md)

    # RAG用のデータを生成 (JSONL)
    rag_data = await generate_rag_data(session)
    file_jsonl = None
    if rag_data:
        filename_jsonl = f"rag_data_{session.topic.replace(' ', '_').replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
        # 各JSONオブジェクトを改行で区切ったJSONL形式の文字列に変換
        jsonl_content = "\n".join(json.dumps(item, ensure_ascii=False) for item in rag_data)
        file_jsonl = discord.File(io.BytesIO(jsonl_content.encode('utf-8')), filename=filename_jsonl)

    # 結果を埋め込みで送信
    embed = discord.Embed(
        title="✨ Dialogue Complete" if session.language == "English" else "✨ 対話の完了",
        description="It was a wonderful time of exploration." if session.language == "English" else "素晴らしい探求の時間でした。",
        color=discord.Color.green()
    )
    
    files_to_send = [file_md]
    record_value_md = f"The dialogue record has been saved as `{filename_md}`." if session.language == "English" else f"対話の記録を `{filename_md}` として保存しました。"
    embed.add_field(name="📝 Dialogue Record (Markdown)", value=record_value_md, inline=False)
    
    if file_jsonl:
        record_value_jsonl = f"Data for RAG has been saved as `{file_jsonl.filename}`." if session.language == "English" else f"RAG用のデータを `{file_jsonl.filename}` として保存しました。"
        embed.add_field(name="🔍 RAG Data (JSONL)", value=record_value_jsonl, inline=False)
        files_to_send.append(file_jsonl)
    
    await ctx.send(embed=embed, files=files_to_send)
    
    # セッション削除
    del bot.sessions[user_id]
    
    # スレッドをアーカイブ
    thread = bot.get_channel(session.thread_id)
    if thread:
        await thread.edit(archived=True)

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    
    await bot.process_commands(message)
    
    # スレッド内のメッセージ処理
    if not isinstance(message.channel, discord.Thread):
        return
    
    user_id = str(message.author.id)
    session = bot.sessions.get(user_id)
    # 再起動復帰: メモリに無ければDBから復元
    if not session:
        try:
            rec = await bot.session_repo.get_session_by_thread(user_id=user_id, thread_id=message.channel.id)
            if rec:
                restored = ThinkingSession(user_id=user_id, topic=rec.topic, thread_id=message.channel.id, language=rec.language or detect_language(rec.topic))
                restored.db_session_id = rec.id
                # スロットをDBから再構築
                try:
                    rows = await bot.session_repo.get_slots(session_id=rec.id)
                    restored.configure_slots(rec.goal_kind or restored.goal_kind)
                    for row in rows:
                        try:
                            restored.slot_registry.add(Slot(
                                name=row.get("name"),
                                description=row.get("description") or "",
                                type=row.get("type"),
                                importance=float(row.get("importance") or 1.0),
                                filled_ratio=float(row.get("filled_ratio") or 0.0),
                                last_filled_ts=row.get("last_filled_ts"),
                                value=row.get("value"),
                                source_kind=row.get("source_kind"),
                            ))
                        except Exception:
                            pass
                    # 最終assistant質問を復元
                    try:
                        last_q = None
                        async for ts_iso, role, text in bot.session_repo.iter_messages(session_id=rec.id):
                            if role == "assistant" and text:
                                last_q = text
                        if last_q:
                            restored.last_question = last_q
                    except Exception:
                        pass
                except Exception:
                    pass
                bot.sessions[user_id] = restored
                session = restored
        except Exception:
            session = None
    
    if not session or message.channel.id != session.thread_id or not session.last_question:
        return
        
    # 今回のやり取りを履歴に追加
    session.add_exchange(session.last_question, message.content)

    if session.pending_slot:
        slot_name = session.pending_slot
        session.slot_registry.update(slot_name, value=message.content, source_kind="user")
        session.goal_state.setdefault("filled", {})[slot_name] = message.content
        session.slot_answers.setdefault(slot_name, []).append(message.content)
        session.pending_slot = None

    # el-agent が必須かつ未利用可能なら即エラーを返す
    if EL_AGENT_REQUIRED and not agent_runtime.available:
        await message.reply("❌ el-agent が利用できません。`el-agent/src` の配置と依存を確認してください（必要なら EL_AGENT_REQUIRED=0 でフォールバック可）。")
        return

    # 回答を分析（el-agent を優先的に利用。必須ならフォールバックしない）
    async with message.channel.typing():
        if agent_runtime.available:
            try:
                # 1) 仮説を用意
                if session.hypothesis is None:
                    session.hypothesis = Hypothesis(
                        id=f"h-{user_id}",
                        text=session.topic,
                        belief=session.belief,
                        belief_ci=session.belief_ci or (0.25, 0.75),
                        action_cost={"ask": 1.0, "search": 0.5},
                    )  # type: ignore[name-defined]
                # 最新回答で仮説テキストを更新（軽量）
                h = session.hypothesis.model_copy(update={"text": f"{session.topic}: {message.content}"})

                # 2) 戦略選択
                action = agent_runtime.strategist.pick_action(h)  # type: ignore[union-attr]
                session.last_action = action.action

                next_question_text: Optional[str] = action.question
                supports: List[str] = []

                # 3) 必要に応じて検索→エビデンス→信頼度更新
                session.belief_updated = False
                if action.action in ("search",):
                    try:
                        docs = agent_runtime.integrator.retrieve(message.content)  # type: ignore[union-attr]
                        sents = agent_runtime.integrator.sentence_extract(docs, h)  # type: ignore[union-attr]
                        ev = agent_runtime.integrator.to_evidence(sents, h)  # type: ignore[union-attr]
                        supports = sents[:3]
                        new_belief = agent_runtime.evaluator.score(h, [ev])  # type: ignore[union-attr]
                        session.belief = float(new_belief)
                        session.belief_ci = (max(0.0, session.belief - 0.2), min(1.0, session.belief + 0.2))
                        session.hypothesis = h.model_copy(update={"belief": session.belief, "belief_ci": session.belief_ci})
                        session.belief_updated = True
                    except Exception:
                        # 検索失敗時は何もしない
                        pass
                elif action.action in ("ask",):
                    # 検索しない場合でも、ユーザ応答がある限り小さな正の更新を反映
                    try:
                        delta = 0.35
                        updated_h = update_belief(h, delta)  # type: ignore[union-attr]
                        session.belief = float(updated_h.belief)
                        session.belief_ci = tuple(updated_h.belief_ci)  # type: ignore[assignment]
                        session.hypothesis = updated_h
                        session.belief_updated = True
                    except Exception:
                        pass

                # 4) ゴール指向プランナーを優先
                planner_q = plan_next_question(session, message.content)
                if planner_q:
                    next_question_text = planner_q

                # 5) 次の質問テキスト（まだ無ければフォールバック生成）
                if not next_question_text:
                    # 既存の軽量LLMフローにフォールバック
                    legacy = await analyze_and_respond_legacy(session)
                    dflt = legacy.get("next_questions", [])
                    next_question_text = (dflt[0].get("question") if dflt else None)  # type: ignore[index]

                if not next_question_text:
                    next_question_text = "もう少し詳しく教えていただけますか？"

                # 記者スタイルで洗練（任意）
                refined_question = next_question_text
                if QUESTION_REFINER:
                    try:
                        ref_sys = (_JOURNALIST_SPEC.get("question_refiner", {}) or {}).get("system_prompt", "")
                        if ref_sys:
                            ref_sys = ref_sys.replace("{{ goal_kind }}", session.goal_kind).replace("{{ language }}", session.language)
                            r = await safe_chat_completion(
                                messages=[
                                    {"role": "system", "content": ref_sys},
                                    {"role": "user", "content": next_question_text},
                                ],
                                temperature=0.3,
                            )
                            refined = (r.choices[0].message.content or "").strip()
                            if refined:
                                refined_question = refined
                    except Exception:
                        pass

                # インタビュー合成（共感的前置き＋核心質問）
                final_text = refined_question
                if INTERVIEW_MODE:
                    try:
                        emp_sys = (_JOURNALIST_SPEC.get("empathy_preface", {}) or {}).get("system_prompt", "")
                        turn_sys = (_JOURNALIST_SPEC.get("interviewer_turn", {}) or {}).get("system_prompt", "")
                        if emp_sys and turn_sys:
                            emp_sys = emp_sys.replace("{{ goal_kind }}", session.goal_kind).replace("{{ language }}", session.language)
                            turn_sys = turn_sys.replace("{{ goal_kind }}", session.goal_kind).replace("{{ language }}", session.language)
                            # 共感前置き生成
                            e = await safe_chat_completion(
                                messages=[
                                    {"role": "system", "content": emp_sys},
                                    {"role": "user", "content": message.content},
                                ],
                                temperature=0.4,
                            )
                            preface = (e.choices[0].message.content or "").strip()
                            # 結合最適化
                            t = await safe_chat_completion(
                                messages=[
                                    {"role": "system", "content": turn_sys},
                                    {"role": "user", "content": f"Preface: {preface}\nQuestion: {refined_question}"},
                                ],
                                temperature=0.2,
                            )
                            merged = (t.choices[0].message.content or "").strip()
                            if merged:
                                final_text = merged
                    except Exception:
                        pass

                # 6) 応答メッセージ（根拠と信頼度を併記）
                color = discord.Color.blue() if action.action == "ask" else discord.Color.green()
                embed = discord.Embed(description=final_text, color=color)
                header = "🔎 次の質問" if session.language == "Japanese" else "🔎 Next Question"
                embed.set_author(name=header)

                if supports:
                    refs_title = "参考になった記述" if session.language == "Japanese" else "Supporting snippets"
                    refs_val = "\n".join([f"• {s}" for s in supports])
                    embed.add_field(name=refs_title, value=refs_val[:1000], inline=False)

                if SHOW_CONFIDENCE and session.belief_updated:
                    conf_title = "現在の確からしさ" if session.language == "Japanese" else "Current confidence"
                    embed.add_field(name=conf_title, value=f"{session.belief:.2f}", inline=True)

                await message.reply(embed=embed)
                session.last_question = final_text
                session.last_supports = supports
                return
            except Exception as _:
                # 必須モードではエラーを返して終了（フォールバックしない）
                if EL_AGENT_REQUIRED:
                    await message.reply("❌ el-agent 実行時エラーが発生しました。ログを確認してください。")
                    return
                # 任意モードでは従来経路へフォールバック
                pass

        # --- 従来（v1）経路 ---
        response = await analyze_and_respond(session)
        questions = response.get('next_questions', [])
        status = response.get('status')
        if questions:
            selected_q_obj = questions[0]
            question_text = selected_q_obj.get('question', '')
            slot_name = selected_q_obj.get('slot_name')
            embed = discord.Embed(
                description=question_text,
                color=discord.Color.blue()
            )
            embed.set_author(name=("🔎 Next Question" if session.language == "English" else "🔎 次の質問"))
            await message.reply(embed=embed)
            session.last_question = question_text
            if slot_name:
                session.pending_slot = slot_name
        elif status == "complete":
            summary_items = response.get("slot_summary") or []
            if not summary_items:
                summary_items = [
                    {
                        "slot": slot.name,
                        "value": slot.value,
                        "filled": slot.filled,
                    }
                    for slot in session.slot_registry.all_slots()
                ]
            lines = []
            for item in summary_items:
                mark = "✅" if item.get("filled") else "⚠️"
                value = (item.get("value") or "").strip() or ("No detail provided" if session.language == "English" else "記録なし")
                lines.append(f"{mark} {item.get('slot')}: {value[:160]}")
            description = "\n".join(lines) or ("All critical slots are filled." if session.language == "English" else "主要スロットはすべて充足しました。")
            embed = discord.Embed(description=description, color=discord.Color.green())
            embed.set_author(name=("✅ Postmortem coverage" if session.language == "English" else "✅ ポストモーテムの準備完了"))
            await message.reply(embed=embed)
            session.last_question = None
            session.pending_slot = None
        elif status == "stalled":
            missing = response.get("missing_slots") or [slot.name for slot in session.slot_registry.unfilled_slots()]
            if session.language == "English":
                text = "I'm pausing for now. Remaining slots: " + ", ".join(missing)
            else:
                text = "一旦ここで止めます。未充足のスロット: " + "、".join(missing)
            await message.reply(text)
            session.last_question = None
            session.pending_slot = None
        else:
            fallback_q = "That's interesting. Could you elaborate on that a bit more?" if session.language == "English" else "興味深いですね。もう少し詳しく教えていただけますか？"
            await message.reply(fallback_q)
            session.last_question = fallback_q


# Opening Question generator using external prompt
async def generate_opening_question(topic: str, lang: str, goal_kind: str = "generic") -> str:
    """Generate the very first question for a given topic using an external prompt file.
    Falls back to the legacy rule-based function if the LLM fails.
    """

    if goal_kind == "postmortem":
        return fallback_question("summary", lang)
    try:
        prompt_path = Path(__file__).parent / "agent" / "prompts" / "opening.yaml"
        with prompt_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        system_prompt_tpl: str = data.get("system_opening_prompt", "")
        system_prompt = system_prompt_tpl.format(language=lang)

        response = await safe_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Topic: {topic}"},
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content or ""
        question = content.strip()
        if question and len(question) < 250:
            return question
        raise ValueError("Invalid opening question")
    except Exception as exc:  # noqa: BLE001
        # Fallback to deterministic rule-based question
        print(f"[OpeningQ] Fallback due to: {exc}")
        return get_opening_question(topic, lang, goal_kind)

# ヘルパー関数
def get_opening_question(topic: str, lang: str, goal_kind: str = "generic") -> str:
    """トピックに応じた知識を活用した最初の質問を生成"""

    if goal_kind == "postmortem":
        return fallback_question("summary", lang)
    topic_lower = topic.lower()

    if lang == "Japanese":
        # 技術カテゴリ
        if any(word in topic_lower for word in ['プログラミング', 'コード', '開発', 'アプリ', 'システム', '機械学習', 'ai']):
            return f"""「{topic}」についてお考えなのですね。技術的なテーマは常に進化しますが、\n今まさに直面している課題や興味深いポイントはどの辺りでしょうか？"""

        # 旅行・体験カテゴリ（行った／訪れた 等を含む）
        if any(word in topic_lower for word in ['行った', '行きました', '訪れ', '旅行', '旅', '観光']):
            return f"""「{topic}」の中で、\n最も印象に残っている『瞬間』や『シーン』は何でしたか？\nその時に感じたことをもう少し詳しく教えていただけますか？"""

        # 抽象 / その他
        return f"""「{topic}」というテーマ、興味深いですね。\nまず、その中であなたが一番大切にしたいポイントは何でしょうか？"""
    else: # English
        if any(word in topic_lower for word in ['programming', 'code', 'develop', 'app', 'system', 'machine learning', 'ai']):
            return f"Thinking about \"{topic}\", I see. Technical topics are deep and always offer new discoveries.\n\nWhat specific aspects of this topic are you currently interested in, or what specific challenges are you working on?"
        elif any(word in topic_lower for word in ['life', 'happiness', 'meaning', 'value', 'live', 'die', 'love']):
            return f"That's a deep theme, \"{topic}\". It's a universal question humanity has pondered throughout history, yet one with a unique answer for each individual.\n\nWhat prompted you to think about this theme, and could you share your current thoughts on it?"

        # 旅行・体験系
        if any(word in topic_lower for word in ['visited', 'travel', 'trip', 'vacation', 'went to']):
            return f"Thinking about your experience \"{topic}\", what was the single most memorable moment or scene?\nCould you describe what made it stand out for you?"

        return f"\"{topic}\" sounds interesting and multi-faceted.\nWhat aspect feels most important for you to explore first?"


# --- Goal-oriented planner (configurable via YAML) ---
def _load_planners() -> Dict[str, Any]:
    path_env = os.getenv("EL_PLANNERS_FILE")
    path = Path(path_env).expanduser() if path_env else PLANNERS_FILE_DEFAULT
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {"goals": []}
    except Exception:
        return {"goals": []}


_PLANNER_SPEC = _load_planners()


def _load_journalist() -> Dict[str, Any]:
    path_env = os.getenv("EL_JOURNALIST_PROMPTS")
    path = Path(path_env).expanduser() if path_env else JOURNALIST_FILE_DEFAULT
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except Exception:
        return {}


_JOURNALIST_SPEC = _load_journalist()


def infer_goal_kind(topic: str) -> str:
    t = topic.lower()
    for goal in _PLANNER_SPEC.get("goals", []):
        kws = (goal.get("match") or {}).get("any_keywords", [])
        if any(str(k).lower() in t for k in kws):
            return str(goal.get("kind", "generic"))
    return "generic"


def plan_next_question(session: ThinkingSession, last_user_msg: str) -> Optional[str]:
    kind = session.goal_kind
    asked = session.goal_state["asked"]
    # 将来的に last_user_msg を filled に反映可能

    # kind に合致するステップを取得
    steps = []
    for goal in _PLANNER_SPEC.get("goals", []):
        if str(goal.get("kind")) == kind:
            steps = goal.get("steps", [])
            break

    lang_key = "ja" if session.language == "Japanese" else "en"
    for step in steps:
        sid = str(step.get("id"))
        if sid not in asked:
            asked.add(sid)
            text = step.get(lang_key) or step.get("en") or step.get("ja") or ""

            # プレースホルダを安全に置換（未定義キーはそのまま残す）
            try:
                class _SafeDict(dict):
                    def __missing__(self, key):  # type: ignore[no-redef]
                        return "{" + key + "}"

                variables = _SafeDict(
                    topic=session.topic or "",
                    last_answer=last_user_msg or "",
                    goal_kind=kind or "",
                    language=session.language or "",
                )
                rendered = str(text).format_map(variables)
            except Exception:
                rendered = str(text)

            return rendered
    return None

async def analyze_and_respond(session: ThinkingSession) -> dict:
    """Dispatch analyze-and-respond flow based on session goal."""

    if session.goal_kind == "postmortem" and session.slot_registry.all_slots():
        return await _run_postmortem_turn(session)
    return await _analyze_and_respond_generic(session)


async def _run_postmortem_turn(session: ThinkingSession) -> dict:
    """Gap-driven loop for postmortem sessions (M1a scope)."""

    if not session.messages:
        raise ValueError("No previous messages recorded")

    answer_text: str = session.messages[-1]["answer"]

    try:
        kg = await extract_knowledge(answer_text, focus=session.topic)
    except Exception as exc:  # noqa: BLE001
        print(f"[Postmortem] extract_knowledge failed: {exc}")
        kg = KGPayload(entities=[], relations=[])

    try:
        merge_and_persist(kg)
    except Exception as merge_err:  # noqa: BLE001
        print(f"[Postmortem] merge_and_persist failed: {merge_err}")

    ranked_slots = analyze_gaps(session.slot_registry, kg)
    next_slot: Optional[Slot] = None
    for slot, priority in ranked_slots:
        if priority <= 0:
            continue
        next_slot = slot
        break

    if next_slot is None:
        if session.slot_registry.is_all_filled():
            summary = [
                {"slot": slot.name, "value": slot.value, "filled": slot.filled}
                for slot in session.slot_registry.all_slots()
            ]
            return {"next_questions": [], "status": "complete", "slot_summary": summary}

        missing = [slot.name for slot in session.slot_registry.unfilled_slots()]
        return {"next_questions": [], "status": "stalled", "missing_slots": missing}

    questions: List[Question] = []
    try:
        questions = await generate_questions([next_slot])
    except Exception as exc:  # noqa: BLE001
        print(f"[Postmortem] generate_questions failed: {exc}")

    validated: List[Question] = []
    if questions:
        try:
            validated = await return_validated_questions(questions)
        except Exception as exc:  # noqa: BLE001
            print(f"[Postmortem] return_validated_questions failed: {exc}")

    chosen: Question | None = None
    candidate_pool: List[Question] = validated if validated else questions
    fallback = fallback_question(next_slot.name, session.language)

    for candidate in candidate_pool:
        if candidate.slot_name and candidate.slot_name != next_slot.name:
            continue
        text = (candidate.text or "").strip()
        if not text:
            continue
        update = {}
        if candidate.slot_name is None:
            update["slot_name"] = next_slot.name
        if text != candidate.text:
            update["text"] = text
        chosen = candidate.model_copy(update=update) if update else candidate
        break

    if chosen is None:
        chosen = Question(slot_name=next_slot.name, text=fallback, specificity=1.0, tacit_power=1.0)

    session.pending_slot = next_slot.name

    return {
        "next_questions": [
            {
                "level": 2,
                "type": "clarifying",
                "question": chosen.text,
                "slot_name": next_slot.name,
            }
        ],
        "status": "continue",
    }


async def _analyze_and_respond_generic(session: ThinkingSession) -> dict:
    """Generate the next question(s) using the full pipeline; fallback to legacy method."""
    try:
        if not session.messages:
            raise ValueError("No previous messages recorded")
        answer_text: str = session.messages[-1]["answer"]

        validated = await run_turn(answer_text, topic_meta=session.topic)
        if not validated:
            raise ValueError("No validated questions")

        next_questions = [
            {
                "level": 2,
                "type": "clarifying",
                "question": q.text,
            }
            for q in validated
        ]
        return {"next_questions": next_questions, "session_progress": {}}
    except Exception as exc:  # noqa: BLE001
        print(f"[Pipeline] Falling back to legacy: {exc}")
        return await analyze_and_respond_legacy(session)

# --- Legacy analysis function kept for fallback ---
async def analyze_and_respond_legacy(session: ThinkingSession) -> dict:
    """回答を分析して次の質問を生成"""
    system_prompt = BILINGUAL_PROMPT_TEMPLATE.format(
        language=session.language,
        style_guide=STYLE_GUIDES.get(session.language, "")
    )
    
    # 対話履歴をuser/assistantロール形式で構築
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        # トピックを最初のユーザーメッセージとして設定
        {"role": "user", "content": f"Let's talk about: {session.topic}"}
    ]

    # 過去のやり取りを履歴に追加
    for exchange in session.messages[-5:]: # 直近5件に制限
        messages.append({"role": "assistant", "content": exchange['question']})
        messages.append({"role": "user", "content": exchange['answer']})
    
    try:
        response = await safe_chat_completion(
            messages=messages,
            temperature=0.8,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("API response content is empty.")
        return json.loads(content)
    except Exception as e:
        print(f"Error in analyze_and_respond: {e}")
        fallback_q = "I see. Could you please tell me a little more about that?" if session.language == "English" else "なるほど、それについてもう少し聞かせていただけますか？"
        return {
            "next_questions": [{"level": 2, "type": "clarifying", "question": fallback_q}]
        }

async def generate_reflection(session: ThinkingSession) -> dict:
    prompt_text = {
        "Japanese": f"以下の対話から、主要な洞察と発見を抽出してください：\n\nトピック: {session.topic}\n対話ログ：\n{session.messages}\n\n主な内容を分析し、以下のキーを持つJSON形式で、日本語で出力してください：\n{{\"key_insights\": [\"主要な気づきや洞察を3-5個\"], \"deepened_understanding\": \"深まった理解の要約\", \"future_questions\": [\"さらに探求できそうな問いを2-3個\"]}}",
        "English": f"From the following dialogue, extract the key insights and discoveries:\n\nTopic: {session.topic}\nLog:\n{session.messages}\n\nAnalyze the content and output in JSON format with the following keys, in English:\n{{\"key_insights\": [\"3-5 key insights\"], \"deepened_understanding\": \"A summary of deepened understanding\", \"future_questions\": [\"2-3 potential future questions\"]}}"
    }
    
    try:
        response = await safe_chat_completion(
            messages=[{"role": "system", "content": prompt_text[session.language]}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("API response content is empty.")
        return json.loads(content)
    except Exception as e:
        print(f"Error in generate_reflection: {e}")
        # ... (error handling) ...
        return {}

async def generate_rag_data(session: ThinkingSession) -> List[Dict[str, Any]]:
    """会話履歴をRAGに適したJSONL形式に変換する"""
    dialogue_text = ""
    for msg in session.messages:
        dialogue_text += f"Q: {msg['question']}\nA: {msg['answer']}\n\n"

    prompt = RAG_EXTRACTION_PROMPT_TEMPLATE.format(
        language=session.language,
        topic=session.topic,
        user_id=session.user_id,
        dialogue_text=dialogue_text
    )

    messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": prompt}]

    try:
        response = await safe_chat_completion(
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("API response for RAG data is empty.")
        
        result = json.loads(content)
        return result.get("knowledge_chunks", [])
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error processing RAG data: {e}")
        return []

async def create_dialogue_record(session: ThinkingSession) -> str:
    """対話全体を要約した Markdown を生成"""

    # 1) ChatGPT から reflection を取得
    reflection = await generate_reflection(session)

    # 2) ラベル設定
    if session.language == "English":
        record = f"# Dialogue Summary: {session.topic}\n\n"
        record += f"**Date**: {session.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
        title_insights = "## Key Insights"
        title_understanding = "## Deepened Understanding"
        title_future = "## Potential Future Questions"
    else:
        record = f"# 対話サマリー: {session.topic}\n\n"
        record += f"**日時**: {session.created_at.strftime('%Y年%m月%d日 %H:%M')}\n\n"
        title_insights = "## 主要な発見"
        title_understanding = "## 深まった理解"
        title_future = "## さらなる探求の問い"

    # 3) 各セクションを追加（reflection が None の場合のフォールバック込み）
    key_insights = (reflection or {}).get("key_insights", [])
    if not key_insights and session.messages:
        # フォールバックとして直近の発言から1行要約を生成
        last_answer = session.messages[-1]["answer"]
        key_insights = [last_answer[:100]]

    if key_insights:
        record += f"{title_insights}\n\n"
        for ins in key_insights:
            record += f"- {ins}\n"
        record += "\n"

    deepened = (reflection or {}).get("deepened_understanding")
    if deepened:
        record += f"{title_understanding}\n\n{deepened}\n\n"

    future_qs = (reflection or {}).get("future_questions", [])
    if future_qs:
        record += f"{title_future}\n\n"
        for fq in future_qs:
            record += f"- {fq}\n"
        record += "\n"

    return record

def get_question_color(level: int) -> discord.Color:
    # (変更なし)
    colors = {1: discord.Color.blue(), 2: discord.Color.green(), 3: discord.Color.gold(), 4: discord.Color.purple(), 5: discord.Color.red()}
    return colors.get(level, discord.Color.default())

def get_question_type_name(q_type: str, lang: str) -> str:
    """質問タイプの名前を返す"""
    names = {
        "Japanese": {'exploratory':'探索的な問いかけ','clarifying':'理解を深める質問','connecting':'知識をつなぐ質問','essential':'本質に迫る問い','creative':'創造性を刺激する質問'},
        "English": {'exploratory':'Exploratory Question','clarifying':'Clarifying Question','connecting':'Connecting Question','essential':'Essential Question','creative':'Creative Question'}
    }
    return names.get(lang, {}).get(q_type, 'Question' if lang == 'English' else '問いかけ')

def create_depth_indicator(depth: int, lang: str) -> str:
    """深さのビジュアル表現"""
    levels_ja = {20:"🌱 芽生え", 40:"🌿 成長", 60:"🌳 深化", 80:"🌲 成熟", 101:"🌟 洞察"}
    levels_en = {20:"🌱 Budding", 40:"🌿 Growing", 60:"🌳 Deepening", 80:"🌲 Maturing", 101:"🌟 Insight"}
    
    levels = levels_en if lang == "English" else levels_ja
    for limit, indicator in levels.items():
        if depth < limit:
            return indicator
    return levels[101]


# --- 追加のユーザーフレンドリーなコマンド ---
@bot.command(name='status')
async def session_status(ctx):
    user_id = str(ctx.author.id)
    session = bot.sessions.get(user_id)
    if not session:
        await ctx.send("アクティブなセッションがありません。/ No active session.")
        return
    embed = discord.Embed(
        title=("セッションの状態" if session.language == "Japanese" else "Session Status"),
        color=discord.Color.teal(),
    )
    embed.add_field(name=("トピック" if session.language == "Japanese" else "Topic"), value=session.topic, inline=False)
    if agent_runtime.available:
        action = session.last_action or "-"
        if SHOW_CONFIDENCE and session.belief_updated:
            belief = getattr(session, "belief", 0.5)
            embed.add_field(name=("確からしさ" if session.language == "Japanese" else "Confidence"), value=f"{belief:.2f}", inline=True)
        embed.add_field(name=("直近のアクション" if session.language == "Japanese" else "Last action"), value=action, inline=True)
    if session.last_supports:
        refs_title = "参考スニペット" if session.language == "Japanese" else "Supporting snippets"
        refs_val = "\n".join([f"• {s}" for s in session.last_supports[:3]])
        embed.add_field(name=refs_title, value=refs_val[:1000], inline=False)
    await ctx.send(embed=embed)


@bot.command(name='help_el')
async def help_el(ctx):
    txt = (
        "利用可能なコマンド:\n"
        "• !explore <トピック> — セッション開始\n"
        "• !reflect — 振り返り要約\n"
        "• !finish — セッション終了とファイル出力\n"
        "• !status — 状態表示（確からしさ・根拠）\n"
        "• !help_el — このヘルプ"
    )
    await ctx.send(txt)


@bot.command(name='end')
async def end_alias(ctx):
    await finish_session(ctx)

if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("❌ DISCORD_BOT_TOKEN is not set.")
    else:
        print("🧠 EL is starting...")
        bot.run(token) 