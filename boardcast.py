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
    "CNBC Closing Bell": "https://rss.art19.com/cnbc-closing-bell", # 新增的美股复盘
    "FT News Briefing": "https://sphinx.acast.com/p/acast/s/ftnewsbriefing"
}

HISTORY_FILE = "podcast_history.json"
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
# 2. 核心下载与转录
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
    print("  🎙️ 正在调用 Groq 进行转录...")
    with open(audio_path, "rb") as file:
        return groq_client.audio.transcriptions.create(
          file=(audio_path, file.read()),
          model="whisper-large-v3",
          response_format="text",
          language="en" 
        )

# ==========================================
# 3. 大模型分析 (区分首次生成与增量更新)
# ==========================================
def generate_or_update_report(new_transcripts_dict, existing_md_content, date_str):
    print("  🧠 正在调用 DeepSeek 进行融合分析...")
    
    # 将新抓取到的逐字稿拼接
    new_text = ""
    for source, data in new_transcripts_dict.items():
        new_text += f"\n\n### 来源：【{source}】 (节目：{data['title']})\n{data['transcript']}\n"
    
    # 动态构建 Prompt
    if existing_md_content:
        print("  🔄 检测到今日已有报告，启动【增量更新】模式...")
        prompt = f"""
        你是精通全球宏观经济与资本市场博弈的首席策略研究员。
        
        以下是你今天早些时候已经生成的《海外宏观与A股映射简报》：
        ```markdown
        {existing_md_content}
        ```
        
        现在市场有了最新的播客更新，以下是新节目的逐字稿：
        ```text
        {new_text}
        ```
        
        **你的任务是进行“增量更新”，输出一份完整的、全新的 Markdown 报告。必须严格遵守以下原则：**
        1. **只做加法**：绝对不能删减或大幅改写原有报告中已经存在的信息。
        2. **标签高亮**：请将新播客提取出的【核心新闻】、【播客速览】无缝追加到相应板块中，并**必须**在新增文字的最前面加上显眼的 `[🌙 晚间新增]` 标签。
        3. **推演升级**：在最后的【A股及大类资产映射推演】部分，结合新老信息，补充 1-2 条新的跨市场交易逻辑，同样打上 `[🌙 晚间新增]` 标签。
        """
    else:
        print("  📝 今日首次运行，启动【全局生成】模式...")
        prompt = f"""
        你是精通全球宏观经济与资本市场博弈的首席策略研究员。请阅读以下今日海外财经播客的逐字稿合集。

        请严格按照以下结构输出 Markdown 格式的报告（全部使用中文，排版专业干练）：

        # 一、今日全球核心主线新闻
        提取 4-5 个最具市场影响力的头条宏观/政经事件。格式：
        * **[事件主题关键词]**：客观事实或核心数据 + 深度观点分析。（来源：播客A/播客B） 根据实际情况在来源中增加注释。

        # 二、各频道内容速览
        无序列表，20字极简概括各期节目核心内容。格式：
        * **【播客名称】《节目名称》**：核心点1；核心点2。

        # 三、A股及大类资产映射推演
        基于上述信息的“共振”，推演其对A股市场及大宗商品的潜在影响（100字以内）：
        1. **流动性与大盘情绪**：外围变量对A股、大宗商品市场整体风险偏好的传导。
        2. **行业板块推演**：明确指出A股细分板块的利多/利空逻辑。
        
        以下是逐字稿原文：
        {new_text}
        """
        
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一位机构顶级的宏观与量化策略研究员。必须严格输出完整的 Markdown 格式文本。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# ==========================================
# 4. 主流程控制
# ==========================================
if __name__ == "__main__":
    today_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    md_filename = os.path.join(REPORTS_DIR, f"{today_str}_海外宏观与A股映射简报.md")
    TEMP_AUDIO_FILE = "temp_episode.mp3"
    
    history = load_history()
    new_episodes_data = {} 
    
    print(f"🚀 开始执行自动化投研系统...\n")
    
    for podcast_name, rss_url in RSS_FEEDS.items():
        print(f"► 正在检查: {podcast_name}")
        try:
            ep_title, audio_file, episode_id = get_latest_audio(rss_url, TEMP_AUDIO_FILE)
            
            if history.get(podcast_name) == episode_id:
                print(f"  ⏭️ 已处理过，跳过。\n")
                continue
            
            if not audio_file:
                print(f"  ⚠️ 无音频，跳过。\n")
                continue
                
            text_content = transcribe_audio(audio_file)
            new_episodes_data[podcast_name] = {"title": ep_title, "transcript": text_content, "id": episode_id}
            print("  ✅ 提取成功，存入缓存区。\n")
            
        except Exception as e:
            print(f"  ❌ 出错: {e}\n")
            
    if os.path.exists(TEMP_AUDIO_FILE):
        os.remove(TEMP_AUDIO_FILE)
        
    if new_episodes_data:
        # 检查今天是否已经存在报告
        existing_md_content = ""
        if os.path.exists(md_filename):
            with open(md_filename, 'r', encoding='utf-8') as f:
                existing_md_content = f.read()
                
        # 生成或更新报告
        final_report = generate_or_update_report(new_episodes_data, existing_md_content, today_str)
        
        # 覆盖写入更新后的完整内容
        if not existing_md_content:
            final_report = f"# 🌅 {today_str} 海外宏观与A股映射简报\n\n{final_report}"
            
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(final_report)
            
        # 更新账本
        for podcast_name, data in new_episodes_data.items():
            history[podcast_name] = data['id']
        save_history(history)
        
        print(f"\n🎉 简报更新完毕！已存入: {md_filename}")
    else:
        print(f"\n😴 无新内容，无需更新。")
