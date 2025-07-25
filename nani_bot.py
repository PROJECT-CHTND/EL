# nani_bot.py - 相手の考えを引き出す汎用Bot

import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import openai
import json
from datetime import datetime
import asyncio
import io
from typing import List, Dict, Any, Optional
from openai.types.chat import ChatCompletionMessageParam

# プロンプトテンプレートをインポート
from prompts import BILINGUAL_PROMPT_TEMPLATE, STYLE_GUIDES, RAG_EXTRACTION_PROMPT_TEMPLATE

# 環境変数読み込み
load_dotenv()

# OpenAI設定
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

class NaniBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        self.sessions = {}
        
    async def on_ready(self):
        print(f'🧠 {self.user} - Nani has started!')
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="to your thoughts"
            )
        )

bot = NaniBot()

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
        
    def add_exchange(self, question: str, answer: str):
        self.messages.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer
        })

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
    
    # セッション作成
    session = ThinkingSession(user_id, topic, thread.id, session_language)
    bot.sessions[user_id] = session
    
    # ウェルカムメッセージ
    welcome_title = "🌱 Let's Begin the Exploration" if session.language == "English" else "🌱 探求の開始"
    welcome_desc = f"Let's think together about \"**{topic}**\"." if session.language == "English" else f"「**{topic}**」について一緒に考えていきましょう。"
    
    embed = discord.Embed(
        title=welcome_title,
        description=welcome_desc,
        color=discord.Color.green()
    )

    about_title = "💡 About this session" if session.language == "English" else "💡 このセッションについて"
    about_value = (
        "I'm here to help you explore your thoughts.\n"
        "There's no need to rush for an answer.\n"
        "Please feel free to speak your mind."
    ) if session.language == "English" else (
        "私はあなたの考えを引き出すお手伝いをします。\n"
        "答えを急ぐ必要はありません。\n"
        "思いつくままに、自由にお話しください。"
    )
    embed.add_field(name=about_title, value=about_value, inline=False)
    
    footer_text = "When you're ready, please reply in the thread." if session.language == "English" else "準備ができたら、スレッド内でお返事ください"
    embed.set_footer(text=footer_text)
    
    await ctx.send(embed=embed)
    
    # 最初の質問
    first_question = get_opening_question(topic, session.language)
    session.last_question = first_question # 最初の質問を保存
    
    q_embed = discord.Embed(
        description=first_question,
        color=discord.Color.blue()
    )
    author_name = "💭 First Question" if session.language == "English" else "💭 最初の問いかけ"
    q_embed.set_author(name=author_name)
    
    await thread.send(embed=q_embed)
    await thread.add_user(ctx.author)

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
    
    if not session or message.channel.id != session.thread_id or not session.last_question:
        return
        
    # 今回のやり取りを履歴に追加
    session.add_exchange(session.last_question, message.content)
    
    # 回答を分析
    async with message.channel.typing():
        response = await analyze_and_respond(session) # answer引数を削除
        
        questions = response.get('next_questions', [])
        if questions:
            selected_q_obj = questions[0]
            
            for q_data in [selected_q_obj]:
                level = int(q_data.get('level', 1))
                q_type = q_data.get('type', 'exploratory')
                
                embed = discord.Embed(
                    description=q_data['question'],
                    color=get_question_color(level)
                )
                
                type_emoji = {'exploratory':'🔍','clarifying':'💡','connecting':'🔗','essential':'💎','creative':'✨'}
                
                embed.set_author(
                    name=f"{type_emoji.get(q_type, '💭')} {get_question_type_name(q_type, session.language)}"
                )
                
                depth = int(response.get('session_progress', {}).get('depth_reached', 0))
                if depth > 0:
                    progress_text = f"{'Depth of Exploration' if session.language == 'English' else '探求の深さ'}: {create_depth_indicator(depth, session.language)}"
                    embed.set_footer(text=progress_text)
                
                await message.reply(embed=embed)
                session.last_question = q_data['question'] # 次の質問を保存
        else:
             # フォールバック: 質問が生成できなかった場合
            fallback_q = "That's interesting. Could you elaborate on that a bit more?" if session.language == "English" else "興味深いですね。もう少し詳しく教えていただけますか？"
            await message.reply(fallback_q)
            session.last_question = fallback_q


# ヘルパー関数
def get_opening_question(topic: str, lang: str) -> str:
    """トピックに応じた知識を活用した最初の質問を生成"""
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

async def analyze_and_respond(session: ThinkingSession) -> dict:
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
        response = await client.chat.completions.create(
            model="gpt-4o",
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
        response = await client.chat.completions.create(
            model="gpt-4o",
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
        response = await client.chat.completions.create(
            model="gpt-4o",
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

if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("❌ DISCORD_BOT_TOKEN is not set.")
    else:
        print("🧠 Nani is starting...")
        bot.run(token) 