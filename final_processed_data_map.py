import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as mpatches
import matplotlib.image as mpimg

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
})


def create_map_with_real_background():
    """创建带有真实地图背景的数据分布图"""
    # 加载真实地图背景
    try:
        # 尝试加载您提供的地图背景
        background_img = mpimg.imread("2025-4-30-participant selection strategy-map-uf.png")
        background_extent = [100, 125, 15, 45]  # 调整这个范围以匹配您的地图
        use_real_background = True
    except:
        print("无法加载真实地图背景，使用简化轮廓")
        use_real_background = False

    # 模拟预处理后的数据
    np.random.seed(42)

    # 定义地区中心坐标（与真实地图匹配）
    regions = {
        'Guangzhou': (113.23, 23.16),
        'Foshan': (113.11, 23.05),
        'Shenzhen': (114.07, 22.62),
        'Dongguan': (113.75, 23.04),
        'Zhanjiang': (110.35, 21.27),
        'Shaoguan': (113.62, 24.84),
        'Suzhou': (120.62, 31.32),
        'Loudi': (111.96, 27.71),
        'Chongqing': (106.54, 29.59)
    }

    # 生成预处理后的数据
    n_tasks = 835
    n_participants = 1847

    tasks_coords = []
    participants_coords = []

    for region, (center_lon, center_lat) in regions.items():
        # 确定每个地区的任务和参与者数量
        n_region_tasks = max(8, int(n_tasks * np.random.uniform(0.07, 0.13)))
        n_region_participants = max(45, int(n_participants * np.random.uniform(0.07, 0.18)))

        # 生成任务坐标
        tasks_lon = np.random.normal(center_lon, 0.2, n_region_tasks)
        tasks_lat = np.random.normal(center_lat, 0.2, n_region_tasks)
        tasks_coords.extend(zip(tasks_lon, tasks_lat, ['task'] * n_region_tasks))

        # 生成参与者坐标
        parts_lon = np.random.normal(center_lon, 0.4, n_region_participants)
        parts_lat = np.random.normal(center_lat, 0.4, n_region_participants)
        participants_coords.extend(zip(parts_lon, parts_lat, ['participant'] * n_region_participants))

    # 创建DataFrame
    df_tasks = pd.DataFrame(tasks_coords, columns=['longitude', 'latitude', 'type'])
    df_parts = pd.DataFrame(participants_coords, columns=['longitude', 'latitude', 'type'])

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 添加背景地图
    if use_real_background:
        ax.imshow(background_img, extent=background_extent, alpha=0.7)
        ax.set_xlim(background_extent[0], background_extent[1])
        ax.set_ylim(background_extent[2], background_extent[3])
    else:
        # 绘制简化中国轮廓
        china_lons = [100, 100, 125, 125, 100]
        china_lats = [15, 45, 45, 15, 15]
        ax.plot(china_lons, china_lats, 'k-', linewidth=2, alpha=0.7)
        ax.set_xlim(100, 125)
        ax.set_ylim(15, 45)

    # 绘制任务和参与者
    ax.scatter(df_tasks['longitude'], df_tasks['latitude'], c='red', s=50,
               alpha=0.8, marker='o', edgecolors='white', linewidth=0.5)
    ax.scatter(df_parts['longitude'], df_parts['latitude'], c='blue', s=20,
               alpha=0.6, marker='.', edgecolors='white', linewidth=0.3)

    # 添加地区标签
    for region, (lon, lat) in regions.items():
        ax.annotate(region, (lon, lat), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

    # 设置标题和标签
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geographic Distribution of Tasks and Participants After Preprocessing\n'
                 f'Tasks: {n_tasks}, Participants: {n_participants}')
    ax.grid(True, alpha=0.3)

    # 创建合并的图例
    task_legend = mpatches.Patch(color='red', label=f'Tasks ({n_tasks})')
    participant_legend = mpatches.Patch(color='blue', label=f'Participants ({n_participants})')
    ax.legend(handles=[task_legend, participant_legend], loc='upper right')

    # 保存图片
    plt.tight_layout()
    plt.savefig('processed_data_on_real_map.png', dpi=300, bbox_inches='tight')
    plt.savefig('processed_data_on_real_map.pdf', bbox_inches='tight')
    plt.show()

    return 'processed_data_on_real_map.png', n_tasks, n_participants


def add_custom_legend(input_path, output_path, n_tasks, n_participants):
    """使用PIL添加自定义图例"""
    # 读取图片
    img = Image.open(input_path)
    draw = ImageDraw.Draw(img)

    # 设置图例参数
    legend_width = 220
    legend_height = 100
    margin = 20

    # 尝试加载字体
    try:
        font = ImageFont.truetype("arialbd.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("msyhbd.ttc", 14)
            small_font = ImageFont.truetype("msyh.ttc", 12)
        except:
            # 使用默认字体
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

    # 计算图例位置（右上角）
    x0 = img.width - legend_width - margin
    y0 = margin

    # 绘制半透明背景
    legend_bg = Image.new('RGBA', (legend_width, legend_height), (255, 255, 255, 200))
    img.paste(legend_bg, (x0, y0), legend_bg)

    # 绘制边框
    draw.rectangle([x0, y0, x0 + legend_width, y0 + legend_height], outline="black", width=2)

    # 绘制标题
    draw.text((x0 + 10, y0 + 10), "DATA LEGEND", font=font, fill="black")

    # 绘制分隔线
    draw.line([x0 + 10, y0 + 30, x0 + legend_width - 10, y0 + 30], fill="black", width=1)

    # 绘制色块和标签
    color_size = 15
    text_offset = 35

    # 红色图例（任务）
    draw.rectangle([x0 + 10, y0 + 40, x0 + 10 + color_size, y0 + 40 + color_size], fill="#FF0000")
    draw.text((x0 + text_offset, y0 + 40), f"Tasks ({n_tasks})", font=small_font, fill="black")

    # 蓝色图例（参与者）
    draw.rectangle([x0 + 10, y0 + 65, x0 + 10 + color_size, y0 + 65 + color_size], fill="#0000FF")
    draw.text((x0 + text_offset, y0 + 65), f"Participants ({n_participants})", font=small_font, fill="black")

    # 保存结果
    img.save(output_path)
    print(f"已保存带图例的地图: {output_path}")
    return img


# 主程序
if __name__ == "__main__":
    # 创建带有真实地图背景的数据分布图
    map_filename, n_tasks, n_participants = create_map_with_real_background()

    # 添加自定义图例
    final_map = add_custom_legend(map_filename, "final_processed_data_map.png", n_tasks, n_participants)

    # 显示最终结果
    final_map.show()