import os
import re
import json
import requests
import feedparser
from groq import Groq
from openai import OpenAI
from datetime import datetime
from pydub import AudioSegment

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
    "CNBC Closing Bell": "https://feeds.simplecast.com/Nh1wIaXT",
    "FT News Briefing": "https://feeds.acast.com/public/shows/73fe3ede-5c5c-4850-96a8-30db8dbae8bf",
    "NPR Up First":"https://feeds.npr.org/510318/podcast.xml"
}

HISTORY_FILE = "podcast_history.json"
REPORTS_DIR = "Daily_Reports"
TRANSCRIPTS_DIR = "Raw_Transcripts" # 新增：存放原始逐字稿的文件夹

def load_history():
    if os.path.exists(HISTORY_FILE):
        if os.path.getsize(HISTORY_FILE) == 0:
            return {}
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("  ⚠️ 警告: history.json 格式错误，已重置为空账本。")
            return {}
    return {}

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def sanitize_filename(filename):
    """清洗文件名，去除不能用于文件命名的特殊符号"""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

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
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    if file_size_mb <= 24.5:
        print(f"音频大小({file_size_mb:.1f}MB)，正在调用 Groq 进行极速转录...")
        with open(audio_path, "rb") as file:
            return groq_client.audio.transcriptions.create(
              file=(audio_path, file.read()),
              model="whisper-large-v3",
              response_format="text",
              language="en" 
            )
    # 如果文件超过 24MB，启动切片机制
    print(f"音频大小({file_size_mb:.1f}MB)！正在将音频分段...")
    audio = AudioSegment.from_file(audio_path)
    
    # 设定每段长度为 15 分钟 (15 * 60 * 1000 毫秒)，确保每段绝对不会超过 25MB
    chunk_length_ms = 15 * 60 * 1000 
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    
    full_transcript = ""
    for idx, chunk in enumerate(chunks):
        print(f"    - 正在转录第 {idx+1}/{len(chunks)} 段...")
        chunk_path = f"temp_chunk_{idx}.mp3"
        # 导出临时分段文件
        chunk.export(chunk_path, format="mp3")
        
        # 逐段发送给 Groq
        with open(chunk_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
              file=(chunk_path, file.read()),
              model="whisper-large-v3",
              response_format="text",
              language="en" 
            )
            # 将每一段的文本拼接到最终的长文本中
            full_transcript += transcription + " "
            
        # 阅后即焚，清理临时切片
        os.remove(chunk_path)
        
    return full_transcript        

# ==========================================
# 3. 大模型分析 (区分首次生成与增量更新)
# ==========================================
def generate_or_update_report(new_transcripts_dict, existing_md_content, today_str, today_str_compact):
    print("  🧠 正在调用 DeepSeek 进行多源信息融合与对冲推演...")
    
    new_text = ""
    for source, data in new_transcripts_dict.items():
        new_text += f"\n\n### 来源：【{source}】 (节目：{data['title']})\n{data['transcript']}\n"
    
    if existing_md_content:
        print("  🔄 检测到今日已有晨报，启动【增量更新】模式...")
        prompt = f"""
        你是精通全球宏观经济与资本市场博弈的首席策略研究员。
        
        以下是你今天早些时候已经生成的《海外宏观与A股映射晨报》：
        ```markdown
        {existing_md_content}
        ```
        
        现在市场有了最新的晚间播客更新，以下是新节目的逐字稿：
        ```text
        {new_text}
        ```
        
        **你的任务是进行高精度的“增量融合更新”，输出一份完整的、全新的 Markdown 报告。必须严格遵守以下原则：**
        
        1. **信息留存**：绝对保留早间报告中的所有事实和数据，不可随意删减。
        2. **主线更新 (最重要)**：在【一、关注要点】中，如果新播客对早上的某个主题提供了新的证据、相反的观点或更深度的监管博弈视角，请在原有的“共识与分歧”下方，另起一行补充对比分析，并在此段开头打上 `[🌙 晚间视角追加]` 标签。如果新播客带来了一个全新的重大主题，请作为一个全新的【主题关键词】追加到第一部分末尾。
        3. **深度提炼追加**：在【二、各播客内容摘要】中，按原定严谨格式，新增晚间播客的深度梳理，并在播客名称旁打上 `[🌙 晚间更新]` 标签。
        4. **推演迭代**：在【三、市场分析】中，结合晚间新获取的量化数据或逻辑，补充 1-2 条新的市场观点，并在段首打上 `[🌙 晚间推演更新]`。如果没有新的增量信息，不必强行补充。
        5.请直接输出正文内容，开头和结尾绝对不要使用 ```markdown 这种代码块标记
        """
    else:
        print("  📝 今日首次运行，启动【全局生成】模式...")
        prompt = f"""
        请阅读以下今日海外财经播客的逐字稿合集。你的任务是进行深度整理和分析，最终输出一份专业、硬核的宏观策略晨报。

        请严格按照以下结构输出 Markdown 格式的报告（排版需结构化、层次分明）：

        # 一、关注要点
        严格根据逐字稿（不要增加额外信息），提取 3-4 个今日最需要关注的市场要点（包括但不限于经济数据、地缘冲突、中美政治局势、金融监管动态等）。针对每个主题，严格采用以下结构进行深度拆解：
        * **【主题关键词】**
          * **客观事实/量化数据**：列出逐字稿中支撑该主题的核心客观数据或事件进展。（请在括号内明确标注涉及这一主题的播客）。
          * **共识与分歧**：综合对比不同播客的观点，根据逐字稿在不同播客之间判断共识和分歧，不要增加额外信息，避免幻觉。如果有共识，提炼核心观点和逻辑；如果存在观点分歧，必须明确指出哪家机构/播客持有什么不同看法（例如：关于降息预期，播客A强调通胀粘性，而播客B更看重劳动力市场降温）。（请在括号内明确标注观点来源）。

        # 二、各播客内容摘要
        摒弃空泛的总结。请对今日抓取的每一个播客进行独立梳理，采用无序列表，提取播客主要内容，这个部分要严格客观，严格基于转录提供的逐字稿来进行总结、分类并归纳。
        * **【播客名称】《节目名称》**（日期，{today_str_compact}）
          * **关键词**：[列出3-5个核心关键词]
          * **关键细节**：如提及的关键经济指标、专有名词、微观博弈现象或英美市场监管动态，逐一列明。
          * **独特视角**：独特的行业观察或非主流观点。

        # 三、市场分析
        基于上述海外信息的共振与分歧，运用宏观对冲思维，推演其对中国 A 股及大宗商品市场的潜在影响（请展开详细的传导逻辑，字数在 300-400 字左右）：
        1. **流动性与估值中枢**：外围变量（如美元、美债、全球风险偏好）对 A 股及大宗商品整体影响和传导路径。
        2. **行业板块映射**：明确指出 A 股哪些细分板块或大宗商品品种具备做多/做空逻辑，并说明原因。

        以下是逐字稿原文：
        {new_text}
        """
        
    response = deepseek_client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": "你是一位顶级的宏观量化对冲基金经理兼首席策略师。你精通经济金融理论和投资管理实践，能从纷繁的市场噪音中寻找多空资金的博弈逻辑，并且对政治局势、政策变化、行业发展等政策面、产业面、政治面和宏观面的因素都具有极高的敏感度。你的分析极其严谨，绝不生成毫无根据的废话。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    final_report = response.choices[0].message.content
    # --- 新增：强力清洗大模型偶尔生成的 ```markdown 外壳 ---
    final_report = final_report.strip()
    if final_report.startswith("```markdown"):
        final_report = final_report[11:].strip()
    elif final_report.startswith("```"):
        final_report = final_report[3:].strip()
        
    if final_report.endswith("```"):
        final_report = final_report[:-3].strip()
        
    return final_report

# ==========================================
# 4. 主流程控制
# ==========================================
if __name__ == "__main__":
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_str_compact = datetime.now().strftime("%Y%m%d")
    
    # 确保保存报告和原始逐字稿的文件夹存在
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    
    md_filename = os.path.join(REPORTS_DIR, f"{today_str}_海外宏观策略晨报.md")
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
            print("  ✅ 提取成功，存入内存缓存区。")
            
            # --- 新增：保存原始逐字稿到单独文件夹 ---
            safe_title = sanitize_filename(ep_title)
            transcript_filename = f"{podcast_name}-{safe_title}-{today_str_compact}.md"
            transcript_filepath = os.path.join(TRANSCRIPTS_DIR, transcript_filename)
            
            with open(transcript_filepath, 'w', encoding='utf-8') as f:
                f.write(f"# 【{podcast_name}】{ep_title}\n")
                f.write(f"**抓取日期：** {today_str}\n\n---\n\n")
                f.write(text_content)
            print(f"  💾 原始逐字稿已归档: {transcript_filename}\n")
            
        except Exception as e:
            print(f"  ❌ 出错: {e}\n")
            
    if os.path.exists(TEMP_AUDIO_FILE):
        os.remove(TEMP_AUDIO_FILE)
        
    if new_episodes_data:
        existing_md_content = ""
        if os.path.exists(md_filename):
            with open(md_filename, 'r', encoding='utf-8') as f:
                existing_md_content = f.read()
                
        final_report = generate_or_update_report(new_episodes_data, existing_md_content, today_str, today_str_compact)
        
        if not existing_md_content:
            final_report = f"# 🌅 {today_str} 海外宏观与市场博弈晨报\n\n{final_report}"
            
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(final_report)
            
        for podcast_name, data in new_episodes_data.items():
            history[podcast_name] = data['id']
        save_history(history)
        
        print(f"\n🎉 简报更新完毕！已存入: {md_filename}")
    else:
        print(f"\n😴 无新内容，无需更新。")
