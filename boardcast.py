import os
import json
import requests
import feedparser
from groq import Groq
from openai import OpenAI
from datetime import datetime

# ==========================================
# 1. 初始化与配置 (云端环境不再需要 dotenv)
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if not GROQ_API_KEY or not DEEPSEEK_API_KEY:
    raise ValueError("未找到 API Key！请检查 GitHub Secrets 配置。")

groq_client = Groq(api_key=GROQ_API_KEY)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# 你的播客 RSS 订阅源字典 (可以随意增减)
RSS_FEEDS = {
    "GS The Markets": "https://feeds.megaphone.fm/GLD9322922848",
    "FT News Briefing": "https://feeds.acast.com/public/shows/ftnewsbriefing",
    "GS The Exchanges":"https://feeds.megaphone.fm/GLD9218176758"
}

# 历史记录文件路径
HISTORY_FILE = "podcast_history.json"

# ==========================================
# 2. 状态管理函数 (记忆功能)
# ==========================================
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# ==========================================
# 3. 核心功能函数
# ==========================================
def get_latest_audio(rss_url, save_path="temp_audio.mp3"):
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        return None, None, None
        
    latest_episode = feed.entries[0]
    title = latest_episode.title
    episode_id = latest_episode.get('id', latest_episode.get('link'))
    
    audio_url = next((link.href for link in latest_episode.links if link.rel == 'enclosure' and 'audio' in link.type), None)
            
    if not audio_url:
        return title, None, episode_id
        
    print(f"  ⬇️ 正在下载: {title}")
    response = requests.get(audio_url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    return title, save_path, episode_id

def transcribe_audio(audio_path):
    print("  🎙️ 正在调用 Groq 进行转录...")
    with open(audio_path, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
          file=(audio_path, file.read()),
          model="whisper-large-v3",
          response_format="text",
          language="en" 
        )
    return transcription

def analyze_transcript(transcript, podcast_name, title):
    print("  🧠 正在调用 DeepSeek 进行 A 股映射分析...")
    
    prompt = f"""
      请阅读以下这集名为《{title}》（来源：{podcast_name}）的英文播客逐字稿，完成以下两部分深度的结构化分析。输出请全部使用中文，排版清晰：
    
    ## 第一部分：原汁原味的播客核心精炼 (Podcast Highlights)
    1. **执行摘要 (Executive Summary)**：用200字左右精准总结这集播客最核心的叙事、逻辑与结论。
    2. **核心观点 (Key Takeaways)**：提炼出3-5个最具洞察力的核心观点，要求保留客观事实，不要主观发散。
    3. **关键数据与专有名词**：列出提到的关键经济数据、专业金融术语或重要机构，并附带简短解释。

    ## 第二部分：A股市场映射与策略推演 (A-Share Market Impact Analysis)
    基于上述播客中提到的海外宏观变量（如政策预期、美债、大宗商品、地缘等），运用宏观对冲与量化思维，推演其对中国A股市场的潜在影响（尽量在150字以内，简洁明了）：
    1. **大盘情绪与流动性**：外围变化将如何通过汇率、利差、外资偏好或全球风险偏好，传导至A股的整体流动性和估值中枢？
    2. **行业与板块映射 (Sector Impact)**：明确指出哪些A股行业板块可能因此**受益（利多）**，哪些可能**受损（利空）**，并一两句话简述传导逻辑（例如：输入型通胀利多上游资源股，美债收益率走高压制科技成长股等）。
    3. **量化监控建议**：为了验证上述推演逻辑，建议在国内市场或跨市场交易中，紧盯哪几个具体的量化宏观指标或高频数据？（如果没有关联紧密的，可以不必有）
    
    以下是逐字稿原文：
    {transcript}
    """
    
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个精通全球宏观经济、资本市场博弈以及中国A股市场策略的首席量化研究员。你的分析极具实战价值，逻辑严密，用词专业。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content

# ==========================================
# 4. 运行主程序与生成 Markdown
# ==========================================
if __name__ == "__main__":
    today_str = datetime.now().strftime("%Y-%m-%d")
    md_filename = f"{today_str}_海外宏观与A股映射简报.md"
    TEMP_AUDIO_FILE = "temp_episode.mp3"
    
    history = load_history()
    md_content = f"# 🎙️ 每日海外宏观与A股映射简报\n**生成日期：** {today_str}\n\n---\n\n"
    has_new_updates = False
    
    print(f"🚀 开始执行自动化工作流...\n")
    
    for podcast_name, rss_url in RSS_FEEDS.items():
        print(f"► 正在检查播客源: {podcast_name}")
        try:
            ep_title, audio_file, episode_id = get_latest_audio(rss_url, TEMP_AUDIO_FILE)
            
            if history.get(podcast_name) == episode_id:
                print(f"  ⏭️ 发现重复：【{ep_title}】之前已处理，跳过。\n")
                continue
            
            if not audio_file:
                print(f"  ⚠️ 未找到音频文件，跳过。\n")
                continue
                
            has_new_updates = True
            text_content = transcribe_audio(audio_file)
            analysis_result = analyze_transcript(text_content, podcast_name, ep_title)
            
            md_content += f"# 【{podcast_name}】{ep_title}\n\n"
            md_content += f"{analysis_result}\n\n---\n\n"
            print("  ✅ 处理完成！\n")
            
            history[podcast_name] = episode_id
            
        except Exception as e:
            print(f"  ❌ 处理 {podcast_name} 时出错: {e}\n")
            
    if os.path.exists(TEMP_AUDIO_FILE):
        os.remove(TEMP_AUDIO_FILE)
        
    save_history(history)
        
    if has_new_updates:
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"🎉 任务执行完毕！简报已生成: {md_filename}")
    else:
        print(f"😴 任务执行完毕。今天所有播客未更新，无需生成简报。")
