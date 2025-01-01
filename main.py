# 标准库
import json
import calendar
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# 数据处理
import numpy as np
import pandas as pd
import tiktoken
import pytz
from tqdm import tqdm

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# NLP相关
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


def process_raw_data(input_path: Path) -> list:
    """第一阶段：处理原始数据，生成基础信息"""
    # 初始化tokenizer
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    # 读取原始数据
    raw_data = json.loads(input_path.read_text(encoding='utf-8'))
    processed_conversations = []
    
    # 使用tqdm创建进度条
    for conv in tqdm(raw_data, desc="处理对话数据"):
        # 基础信息
        conv_info = {
            'uuid': conv['uuid'],
            'name': conv['name'],
            'input_tokens': 0,
            'output_tokens': 0,
            'turns': len(conv['chat_messages']),  # 添加对话轮次
            'human_turns': 0,  # 人类发言次数
            'assistant_turns': 0,  # AI发言次数
        }
        
        # 转换时间到北京时间
        utc_time = datetime.fromisoformat(conv['created_at'].replace('Z', '+00:00'))
        beijing_tz = pytz.timezone('Asia/Shanghai')
        beijing_time = utc_time.astimezone(beijing_tz)
        conv_info['created_at'] = beijing_time.isoformat()
        
        # 计算tokens和轮次
        for msg in conv['chat_messages']:
            tokens = len(enc.encode(msg['text'], allowed_special={'<|endoftext|>'}))
            if msg['sender'] == 'human':
                conv_info['input_tokens'] += tokens
                conv_info['human_turns'] += 1
            else:
                conv_info['output_tokens'] += tokens
                conv_info['assistant_turns'] += 1
        
        processed_conversations.append(conv_info)
    
    return processed_conversations

def analyze_conversations(processed_data: list) -> dict:
    """第二阶段：统计分析处理后的数据"""
    stats = {
        'total_conversations': len(processed_data),
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'total_turns': 0,
        'total_human_turns': 0,
        'total_assistant_turns': 0,
        'monthly_counts': defaultdict(int),
        'weekly_counts': defaultdict(int),
        'daily_counts': defaultdict(int),
        'weekday_counts': defaultdict(int),
        'latest_conversation': None
    }
    
    latest_time = datetime.min.replace(tzinfo=pytz.UTC)
    
    for conv in processed_data:
        # 累加tokens和轮次
        stats['total_input_tokens'] += conv['input_tokens']
        stats['total_output_tokens'] += conv['output_tokens']
        stats['total_turns'] += conv['turns']
        stats['total_human_turns'] += conv['human_turns']
        stats['total_assistant_turns'] += conv['assistant_turns']
        
        # 转换时间字符串回datetime对象
        created_time = datetime.fromisoformat(conv['created_at'])
        
        # 更新各时间维度的计数
        stats['monthly_counts'][created_time.strftime('%Y-%m')] += 1
        stats['weekly_counts'][created_time.strftime('%Y-W%W')] += 1
        stats['daily_counts'][created_time.strftime('%Y-%m-%d')] += 1
        stats['weekday_counts'][calendar.day_name[created_time.weekday()]] += 1
        
        # 更新最晚对话
        if created_time > latest_time:
            latest_time = created_time
            stats['latest_conversation'] = conv
    
    # 计算平均值
    stats['avg_turns_per_conversation'] = stats['total_turns'] / stats['total_conversations']
    stats['avg_human_turns_per_conversation'] = stats['total_human_turns'] / stats['total_conversations']
    stats['avg_assistant_turns_per_conversation'] = stats['total_assistant_turns'] / stats['total_conversations']
    
    # 找出最忙的日子
    stats['busiest_day'] = max(stats['daily_counts'].items(), key=lambda x: x[1])
    stats['busiest_weekday'] = max(stats['weekday_counts'].items(), key=lambda x: x[1])
    
    return stats

def print_analysis(stats: dict) -> None:
    """打印分析结果"""
    print("\n=== Claude 对话年度总结 ===")
    print("\n基础统计:")
    print(f"总对话数: {stats['total_conversations']:,}")
    print(f"总对话轮次: {stats['total_turns']:,}")
    print(f"- 人类发言次数: {stats['total_human_turns']:,}")
    print(f"- AI发言次数: {stats['total_assistant_turns']:,}")
    print(f"平均每次对话轮次: {stats['avg_turns_per_conversation']:.2f}")
    print(f"- 平均人类发言次数: {stats['avg_human_turns_per_conversation']:.2f}")
    print(f"- 平均AI发言次数: {stats['avg_assistant_turns_per_conversation']:.2f}")
    print(f"\nToken统计:")
    print(f"总输入 tokens: {stats['total_input_tokens']:,}")
    print(f"总输出 tokens: {stats['total_output_tokens']:,}")
    print(f"总 tokens: {stats['total_input_tokens'] + stats['total_output_tokens']:,}")
    
    print("\n时间分布:")
    print(f"最忙的一天: {stats['busiest_day'][0]}, 共 {stats['busiest_day'][1]} 次对话")
    print(f"一周中最忙的日子: {stats['busiest_weekday'][0]}, 平均每周 {stats['busiest_weekday'][1]} 次对话")


def filter_data_by_year(processed_data: list, year: int = 2024) -> list:
    """
    按年份筛选数据并补充完整的时间范围
    
    Args:
        processed_data: 处理后的对话数据列表
        year: 要筛选的年份
    Returns:
        list: 过滤后的数据列表
    """
    # 转换时间并筛选指定年份的数据
    filtered_data = []
    for conv in processed_data:
        date = datetime.fromisoformat(conv['created_at'])
        if date.year == year:
            filtered_data.append(conv)
    
    return filtered_data

def create_contribution_wall(processed_data: list, year: int = 2024, save_path: str = 'data/contribution_wall.png') -> None:
    """
    创建类似 GitHub 贡献墙的可视化
    
    Args:
        processed_data: 处理后的对话数据列表
        year: 要统计的年份
        save_path: 保存图片的路径
    """
    # 首先过滤数据
    # filtered_data = filter_data_by_year(processed_data, year)
    # 提取日期并创建DataFrame
    dates = [datetime.fromisoformat(conv['created_at']).date() for conv in filtered_data]
    df = pd.DataFrame({'date': pd.to_datetime(dates)})
    
    # 计算每天的对话次数
    daily_counts = df['date'].value_counts().reset_index()
    daily_counts.columns = ['date', 'count']
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    # 创建完整年份的日期范围
    start_date = pd.Timestamp(f"{year}-01-01")
    end_date = pd.Timestamp(f"{year}-12-31")
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 创建完整的日期DataFrame并合并数据
    full_df = pd.DataFrame({'date': date_range})
    full_df = full_df.merge(daily_counts, on='date', how='left')
    full_df['count'] = full_df['count'].fillna(0)
    
    # 添加年、周和工作日信息
    full_df['year'] = full_df['date'].dt.isocalendar().year
    full_df['week'] = full_df['date'].dt.isocalendar().week
    full_df['weekday'] = full_df['date'].dt.weekday
    
    # 创建年周组合的唯一标识
    full_df['year_week'] = full_df['year'].astype(str) + '-' + full_df['week'].astype(str).str.zfill(2)
    
    # 准备热力图数据
    pivot_data = full_df.pivot_table(
        index='weekday',
        columns='year_week',
        values='count',
        aggfunc='sum'
    )
    
    # 设置绘图样式
    plt.style.use('default')
    plt.figure(figsize=(20, 4))
    
    # 创建自定义颜色映射
    colors = ['#ebedf0', '#9be9a8', '#40c463', '#30a14e', '#216e39']
    max_count = full_df['count'].max()
    breaks = [0, 1, max_count * 0.25, max_count * 0.5, max_count * 0.75, max_count]
    
    # 绘制热力图
    ax = sns.heatmap(pivot_data, 
                     cmap=sns.color_palette(colors),
                     square=True,
                     linewidths=1,
                     linecolor='white',
                     cbar=False,
                     norm=plt.Normalize(0, max_count))
    
    # 设置y轴标签（周几）
    plt.yticks(np.arange(7) + 0.5, 
               [calendar.day_abbr[i] for i in range(7)], 
               rotation=0)
    
    # 添加月份标签
    months = pd.date_range(start=start_date, end=end_date, freq='M')
    month_positions = []
    month_labels = []
    
    for date in months:
        # 计算这个月开始时在数据中的位置
        month_start = pd.Timestamp(f"{date.year}-{date.month:02d}-01")
        position = full_df[full_df['date'] == month_start].index[0] // 7
        month_positions.append(position)
        month_labels.append(date.strftime('%b'))
    
    plt.xticks(month_positions, month_labels, rotation=0)
    
    # 设置标题和样式
    plt.title('Annual Conversation Distribution', pad=20, fontsize=14)
    
    # 添加图例
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color) 
                      for color in colors]
    legend_labels = [f'{int(breaks[i])}-{int(breaks[i+1])}' 
                    for i in range(len(breaks)-1)]
    ax.legend(legend_elements, legend_labels, 
             title='Daily Conversations',
             loc='center left',
             bbox_to_anchor=(1, 0.5))
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 输出统计信息
    total_days = len(full_df)
    active_days = len(full_df[full_df['count'] > 0])
    max_streak = get_max_streak(full_df)
    current_streak = get_current_streak(full_df)
    
    print(f"\n活跃度统计:")
    print(f"总天数: {total_days}")
    print(f"有对话的天数: {active_days}")
    print(f"活跃率: {(active_days/total_days)*100:.1f}%")
    print(f"最长连续对话天数: {max_streak}")
    print(f"当前连续对话天数: {current_streak}")
    
    # 输出每月统计
    monthly_stats = full_df.set_index('date').resample('M')['count'].agg(['sum', 'mean', 'max'])
    monthly_stats.index = monthly_stats.index.strftime('%Y-%m')
    print("\n每月统计:")
    for month, stats in monthly_stats.iterrows():
        print(f"{month}: 总对话 {int(stats['sum'])}次, 平均每天 {stats['mean']:.1f}次, 最高 {int(stats['max'])}次")

def get_max_streak(df: pd.DataFrame) -> int:
    """计算最长连续对话天数"""
    streak = 0
    max_streak = 0
    for count in df['count']:
        if count > 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

def get_current_streak(df: pd.DataFrame) -> int:
    """计算当前连续对话天数"""
    streak = 0
    for count in df['count'][::-1]:  # 从最近的日期开始
        if count > 0:
            streak += 1
        else:
            break
    return streak

def is_chinese(text):
    """判断字符串是否包含中文"""
    return bool(re.search('[\u4e00-\u9fff]', text))

def process_text(text):
    """处理文本，返回分词结果"""
    # 移除URL
    text = re.sub(r'http\S+|www.\S+', '', text)
    # 移除特殊字符，保留中英文字母和空格
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    
    # 初始化结果列表
    words = []
    
    # 如果文本包含中文，使用jieba分词
    if is_chinese(text):
        words.extend(jieba.cut(text))
    else:
        # 使用nltk进行英文分词
        words.extend(word_tokenize(text.lower()))
    
    return words

def generate_wordcloud(processed_data: list, year: int = 2024, save_path: str = 'data/wordcloud.png') -> None:
    """
    从对话主题生成词云，支持中英文混合
    
    Args:
        processed_data: 处理后的对话数据列表
        year: 要分析的年份
        save_path: 保存图片的路径
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
    
    # 过滤指定年份的数据
    filtered_data = [
        conv for conv in processed_data 
        if datetime.fromisoformat(conv['created_at']).year == year
    ]
    
    # 提取所有对话主题
    titles = [conv['name'] for conv in filtered_data]
    
    # 停用词集合（中英文）
    chinese_stop_words = {'的', '了', '和', '是', '在', '我', '有', '让', '与', '你', '用', '要', 
                         '把', '从', '这', '那', '都', '为', '及', '或', '被', '给', '上', '下'}
    english_stop_words = set(stopwords.words('english'))
    stop_words = chinese_stop_words.union(english_stop_words)
    
    # 处理所有标题并收集词频
    word_freq = Counter()
    for title in titles:
        words = process_text(title)
        # 过滤停用词和空字符串
        filtered_words = [w for w in words if w.strip() and w not in stop_words and len(w) > 1]
        word_freq.update(filtered_words)
    
    # 创建词云
    wordcloud = WordCloud(
        font_path='/System/Library/Fonts/PingFang.ttc',  # macOS 的中文字体路径
        width=1200,
        height=800,
        background_color='white',
        max_words=100,
        max_font_size=200,
        random_state=42,
        collocations=False  # 避免重复词组
    )
    
    # 生成词云
    wordcloud.generate_from_frequencies(word_freq)
    
    # 显示词云
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Conversation Topics in {year}', pad=20, fontsize=16)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 输出最常见的话题
    print("\n最常见的话题词：")
    for word, count in word_freq.most_common(10):
        print(f"{word}: {count}次")

    # 分别统计中英文词频
    chinese_words = {word: count for word, count in word_freq.items() if is_chinese(word)}
    english_words = {word: count for word, count in word_freq.items() if not is_chinese(word)}
    
    print("\n中文话题词 TOP 5:")
    for word, count in Counter(chinese_words).most_common(5):
        print(f"{word}: {count}次")
        
    print("\n英文话题词 TOP 5:")
    for word, count in Counter(english_words).most_common(5):
        print(f"{word}: {count}次")

def get_stats_json(stats: dict) -> dict:
    return {
        "conversation_stats": {
            "total": stats['total_conversations'],
            "turns": stats['total_turns'],
            "human_turns": stats['total_human_turns'],
            "ai_turns": stats['total_assistant_turns'],
            "avg_turns": round(stats['avg_turns_per_conversation'], 2),
            "avg_human": round(stats['avg_human_turns_per_conversation'], 2),
            "avg_ai": round(stats['avg_assistant_turns_per_conversation'], 2)
        },
        "token_stats": {
            "input": stats['total_input_tokens'],
            "output": stats['total_output_tokens'],
            "total": stats['total_input_tokens'] + stats['total_output_tokens']
        },
        "time_stats": {
            "busiest_day": {
                "date": stats['busiest_day'][0],
                "count": stats['busiest_day'][1]
            },
            "busiest_weekday": {
                "day": stats['busiest_weekday'][0],
                "avg_count": stats['busiest_weekday'][1]
            }
        }
    }

def main():
    # 设置输入输出路径
    input_path = Path('data/data-2024-12-31-10-52-42-conversations.json')
    processed_path = Path('data/processed_conversations.json')
    
    # 检查原始数据文件是否存在
    if not input_path.exists():
        print(f"错误：找不到输入文件 {input_path}")
        return
    
    # 检查是否存在处理好的数据
    if processed_path.exists():
        print("找到已处理的数据，直接读取...")
        processed_data = json.loads(processed_path.read_text(encoding='utf-8'))
    else:
        print("未找到处理好的数据，开始处理原始数据...")
        # 第一阶段：处理原始数据
        processed_data = process_raw_data(input_path)
        
        # 保存处理后的数据
        processed_path.write_text(
            json.dumps(processed_data, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        print(f"处理完成，结果已保存至 {processed_path}")
    
    analysis_year = 2024
    processed_data = filter_data_by_year(processed_data, analysis_year)
    # 第二阶段：统计分析
    print("开始统计分析...")
    stats = analyze_conversations(processed_data)
    
    # 打印结果
    print_analysis(stats)
    # 保存结果
    with open("public/data/stats.json", "w", encoding="utf-8") as f:
        json.dump(get_stats_json(stats), f, ensure_ascii=False, indent=4)
    
    # 第三阶段：生成贡献墙
    print("\n开始生成贡献墙...")
    create_contribution_wall(processed_data)

    # 生成词云
    print(f"\n开始生成{analysis_year}年词云...")
    generate_wordcloud(processed_data, year=analysis_year)


if __name__ == '__main__':
    main()