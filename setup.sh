#!/bin/bash
# Python仮想環境作成
python3 -m venv venv

# 必要なパッケージを仮想環境にインストール
./venv/bin/pip install discord.py openai python-dotenv

# .envファイルのテンプレート作成
cat > .env << EOL
DISCORD_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
EOL

# .gitignore作成
cat > .gitignore << EOL
.env
venv/
__pycache__/
*.pyc
.DS_Store
EOL

echo "✅ セットアップ完了！"
echo "📝 .envファイルにトークンを設定してください"
echo "👉 仮想環境を有効にするには 'source venv/bin/activate' を実行してください"