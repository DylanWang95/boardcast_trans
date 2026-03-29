import os
import re

# ⚠️ 注意这里：改成了相对路径，因为要在 GitHub 云端根目录下运行
data_folder = './Daily_Reports'
output_file = '市场趋势总览.md'

pattern = re.compile(r'(?:^|\n)#*\s*三、\s*市场分析[^\n]*\n(.*?)(?=\n#|\n四、|\Z)', re.DOTALL)

print("开始在云端提取市场分析内容...")

all_md_files = []
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith('.md'):
            all_md_files.append(os.path.join(root, file))

all_md_files.sort(key=lambda x: os.path.basename(x), reverse=True)

with open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.write("# 📈 市场趋势总览（按天自动汇编）\n\n")
    
    for filepath in all_md_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            match = pattern.search(content)
            if match:
                analysis_text = match.group(1).strip() 
                filename = os.path.basename(filepath)
                date_title = filename.replace('_海外宏观策略晨报.md', '').replace('.md', '')
                
                outfile.write(f"## {date_title}\n\n")
                outfile.write(analysis_text + "\n\n")
                outfile.write("---\n\n")

print("✅ 云端提取完成！")
