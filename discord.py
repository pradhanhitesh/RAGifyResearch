__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from openai import OpenAI
from RAGify.ingest import IngestData
from RAGify.agents import SafetyAgent, QueryAgent, RAGAgent

import discord
from discord.ext import commands

# Add your discord TOKEN
TOKEN='your_discord_token'

# Setup clients for LMS and ChromaDB
lms_client = OpenAI(base_url="http://localhost:5555/v1", api_key="lm-studio")
chroma_client = chromadb.PersistentClient(path='chroma_db')

# Initialize IngestData
ingest = IngestData(lms_client, chroma_client)
database = ingest._load_db("your_db_name")

# Enable necessary intents
intents = discord.Intents.default()
intents.message_content = True

# Initialize bot with intents
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.command(name='ask')
async def ask(ctx, *, prompt: str):
    try:
        async with ctx.typing():  # Show typing animation
            # Generate response from LLM
            query = QueryAgent(lms_client).enhanced_query(prompt, 'rewrite')
            rag = RAGAgent(query, database, lms_client)
            response = rag._generation()

            # Chunking the response to below 2000 characters
            split_chunks = rag._smart_chunk(response.split("\n"))
            for i, chunk in enumerate(split_chunks):
                await ctx.send(chunk)

    except Exception as e:
        await ctx.send(f"An error occurred: {e}")

# Run the bot
bot.run(TOKEN)