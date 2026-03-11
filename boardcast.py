import os
import json
import requests
import feedparser
from groq import Groq
from openai import OpenAI
from datetime import datetime

# ==========================================
# 1. 初始化与配置
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if not GROQ_API_KEY or not DEEPSEEK_API_KEY:
    raise ValueError("未找到 API Key！请检查 GitHub Secrets 配置。")

groq_client = Groq(api_key=GROQ_API_KEY)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

RSS_FEEDS = {
    "GS The Markets": "https://feeds.megaphone.fm/GLD9322922848",
    "WSJ What's News": "https://video-api.wsj.com/podcast/rss/wsj/whats-news",
    "Bloomberg Daybreak": "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/d15648c4-cee0-4e99-a5ca-af80014d921d/65ce2954-d8a8-4c66-9f42-af80014d923e/podcast.rss",
    "Marketplace Morning": "https://feeds.publicradio.org/public_feeds/marketplace-morning-report/itunes/rss",
    "NPR Up First":"https://feeds.npr.org/510318/podcast.xml"
}

HISTORY_FILE = "podcast_history.json"
# 新增：定义保存 Markdown 文件的专属文件夹
REPORTS_DIR = "Daily_Reports"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# ==========================================
# 2. 核心功能函数 (下载与转录)
# ==========================================
def get_latest_audio(rss_url, save_path="temp_audio.mp3"):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }
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
    response = requests.get(audio_url, stream=True, headers=headers) 
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    return title, save_path, episode_id

def transcribe_audio(audio_path):
    print("  🎙️ 正在调用 Groq 进行极速转录...")
    with open(audio_path, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
          file=(audio_path, file.read()),
          model="whisper-large-v3",
          response_format="text",
          language="en" 
        )
    return transcription

# ==========================================
# 3. 核心功能函数 (集中融合分析 Synthesis)
# ==========================================
def synthesize_transcripts(all_transcripts_dict, date_str):
    print("  🧠 正在调用 DeepSeek 进行全局信息融合与分析...")
    
    # 将字典中的所有逐字稿拼接成一个巨大的文本块，并标明来源
    combined_text = ""
    for source, data in all_transcripts_dict.items():
        combined_text += f"\n\n### 来源：【{source}】 (节目：{data['title']})\n"
        combined_text += f"{data['transcript']}\n"
    
    prompt = f"""
    你是精通全球宏观经济与资本市场博弈的首席策略研究员。请阅读以下由今日多档顶级海外财经播客汇总而成的逐字稿合集。
    你需要跨越单一节目的局限，进行全局的“信息融合分析 (Synthesis Analysis)”。

    请严格按照以下结构输出 Markdown 格式的报告（全部使用中文，排版专业干练）：

    # 一、今日全球核心主线新闻
    从所有文本中提取 4-5 个最具市场影响力的头条宏观/政经事件。必须严格采用以下格式描述：
    * **[事件主题关键词]**：先陈述客观事实或核心数据。随后补充深度的观点分析。（来源：播客A / 播客B）

    # 二、各频道内容速览
    用无序列表的方式，列出今天抓取的所有节目，并用 20 字左右的极简语言概括该期节目的核心内容。格式如下：
    * **【播客名称】《节目名称》时间(格式20260310)。**：核心点1；核心点2。

    # 三、A股及大类资产映射推演
    基于上述所有信息的“共振”，运用宏观对冲与量化思维，推演其对中国A股市场及大宗商品的潜在影响(100字以内）：
    1. **流动性与大盘情绪**：外围变量对股票市场、大宗商品市场的整体影响。
    2. **行业板块推演**：哪些A股细分板块或者行业可能迎来利多或利空。
    
    以下是今日各档播客的逐字稿合集原文：
    {combined_text}
    """
    
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一位机构顶级的宏观与量化策略研究员，擅长从繁杂的信息中提炼交易线索。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# ==========================================
# 4. 运行主程序 (重构版逻辑)
# ==========================================
if __name__ == "__main__":
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 确保保存报告的文件夹存在，没有就新建一个
    os.makedirs(REPORTS_DIR, exist_ok=True)
    md_filename = os.path.join(REPORTS_DIR, f"{today_str}_海外宏观与A股映射晨报.md")
    TEMP_AUDIO_FILE = "temp_episode.mp3"
    
    history = load_history()
    
    # 用于临时存储所有抓取到的新播客及其转录文本
    # 格式: {"播客名称": {"title": "节目名字", "transcript": "长段逐字稿"}}
    new_episodes_data = {} 
    
    print(f"🚀 开始执行自动化晨报系统...\n")
    
    # 阶段一：批量下载与转录
    for podcast_name, rss_url in RSS_FEEDS.items():
        print(f"► 正在检查源: {podcast_name}")
        try:
            ep_title, audio_file, episode_id = get_latest_audio(rss_url, TEMP_AUDIO_FILE)
            
            if history.get(podcast_name) == episode_id:
                print(f"  ⏭️ 无更新或已处理，跳过。\n")
                continue
            
            if not audio_file:
                print(f"  ⚠️ 未找到音频，跳过。\n")
                continue
                
            text_content = transcribe_audio(audio_file)
            
            # 把成功转录的文本塞进字典里攒着
            new_episodes_data[podcast_name] = {
                "title": ep_title,
                "transcript": text_content,
                "id": episode_id # 暂时存着，最后一起更新账本
            }
            print("  ✅ 文本提取成功，进入缓存区。\n")
            
        except Exception as e:
            print(f"  ❌ 处理 {podcast_name} 时出错: {e}\n")
            
    if os.path.exists(TEMP_AUDIO_FILE):
        os.remove(TEMP_AUDIO_FILE)
        
    # 阶段二：集中处理并生成报告
    if new_episodes_data:
        print(f"🔄 阶段一完成。共获取 {len(new_episodes_data)} 档新节目文本，开始融合分析...")
        
        # 将所有缓存的文本一次性发给大模型
        final_report = synthesize_transcripts(new_episodes_data, today_str)
        
        # 加上大标题，写入文件
        full_md_content = f"# 🌅 {today_str} 海外宏观与A股映射晨报\n\n{final_report}"
        
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(full_md_content)
            
        # 更新历史账本
        for podcast_name, data in new_episodes_data.items():
            history[podcast_name] = data['id']
        save_history(history)
        
        print(f"\n🎉 晨报生成完毕！已存入专属文件夹: {md_filename}")
    else:
        print(f"\n😴 今天所有播客均未更新，无需生成晨报。")
