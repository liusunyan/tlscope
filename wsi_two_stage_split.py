import openslide
from PIL import Image
import os
import numpy as np
import tqdm
import argparse
import cv2

def is_background(img, entropy_threshold=4.0):
    """
    检测图像是否为背景图像
    
    参数:
    - img: PIL Image或numpy array
    - entropy_threshold: 熵值阈值，低于此值认为是背景 (默认4.0)
    
    返回:
    - bool: True表示是背景，False表示不是背景
    """
    # 如果是PIL Image，转换为numpy array
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # 确保是BGR格式（OpenCV格式）
    if len(img.shape) == 2:  # 已经是灰度图
        gray = img
    elif img.shape[2] == 3:  # RGB
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.shape[2] == 4:  # RGBA
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    else:
        gray = img
    
    # 应用高斯模糊以减少噪声
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # 归一化
    
    # 计算熵
    entropy = 0.0
    for i in range(256):
        p = hist[i][0]
        if p > 0:
            entropy -= p * np.log2(p)
    
    # 判断是否为背景
    return entropy < entropy_threshold


def two_stage_split(slide_path, large_output_dir, small_output_dir, 
                    large_tile_size=5000, small_tile_size=1000, 
                    scaled_large_size=1000, level=0):
    """
    两阶段切分WSI图像：
    1. 切成5000*5000的大图，缩放到1000*1000，文件名加上行列号
    2. 将每张5000*5000大图切分成25张1000*1000的小图，保存到对应文件夹
    
    参数:
    - slide_path: WSI图像路径
    - large_output_dir: 大图输出文件夹
    - small_output_dir: 小图输出文件夹
    - large_tile_size: 大图切分尺寸 (默认5000)
    - small_tile_size: 小图切分尺寸 (默认1000)
    - scaled_large_size: 大图缩放后的尺寸 (默认1000)
    - level: OpenSlide读取的层级 (默认0，最高分辨率)
    """
    
    # 打开WSI文件
    slide = openslide.OpenSlide(slide_path)
    
    # 创建输出目录
    os.makedirs(large_output_dir, exist_ok=True)
    os.makedirs(small_output_dir, exist_ok=True)
    
    # 获取指定级别下的图像大小
    level_width, level_height = slide.level_dimensions[level]
    print(f"原始图像尺寸 (Level {level}): {level_width} x {level_height}")
    
    # 计算能裁剪出多少个大图
    cols = level_width // large_tile_size
    rows = level_height // large_tile_size
    print(f"大图网格: {rows} 行 x {cols} 列")
    
    # 获取图像文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(slide_path))[0]
    
    
    # 遍历每个大图位置
    for row in tqdm.tqdm(range(rows), desc="处理大图"):
        for col in range(cols):
            # 计算大图在原始图像中的位置
            x = col * large_tile_size
            y = row * large_tile_size
            
            # 读取5000*5000的区域
            large_region = slide.read_region((x, y), level, (large_tile_size, large_tile_size))
            large_region = large_region.convert("RGB")
            
            # 检查是否为背景图像
            if is_background(large_region):
                # print(f"检测到背景图像，跳过: row={row}, col={col}")
                continue
            
            # 1. 保存缩放后的大图 (5000*5000 -> 1000*1000)
            scaled_large_region = large_region.resize((scaled_large_size, scaled_large_size), Image.LANCZOS)
            large_tile_filename = f"{base_name}_{row}_{col}.png"
            large_tile_path = os.path.join(large_output_dir, large_tile_filename)
            scaled_large_region.save(large_tile_path)
            
            # 2. 将5000*5000大图切分成25张1000*1000的小图
            small_rows = large_tile_size // small_tile_size  # 5
            small_cols = large_tile_size // small_tile_size  # 5
            
            for small_row in range(small_rows):
                for small_col in range(small_cols):
                    # 从大图中裁剪1000*1000的小图
                    left = small_col * small_tile_size
                    top = small_row * small_tile_size
                    right = left + small_tile_size
                    bottom = top + small_tile_size
                    
                    small_region = large_region.crop((left, top, right, bottom))
                    
                    # 保存小图，文件名包含大图的行列号和小图的行列号
                    small_tile_filename = f"{base_name}_{row}_{col}_subtile_r{small_row}_c{small_col}.png"
                    small_tile_path = os.path.join(small_output_dir, small_tile_filename)
                    small_region.save(small_tile_path)
    
    slide.close()
    print(f"\n处理完成！")
    print(f"大图 (缩放到{scaled_large_size}x{scaled_large_size}) 已保存到: {large_output_dir}")
    print(f"小图 ({small_tile_size}x{small_tile_size}) 已保存到: {small_tiles_subdir}")


def process_directory(input_dir, large_output_dir, small_output_dir, 
                      extensions=('.mrxs', '.svs', '.vsi', '.ndpi', '.tif', '.tiff'),
                      **kwargs):
    """
    批量处理文件夹中的所有WSI图像
    
    参数:
    - input_dir: 输入文件夹路径
    - large_output_dir: 大图输出文件夹
    - small_output_dir: 小图输出文件夹
    - extensions: 支持的文件扩展名元组
    - **kwargs: 传递给two_stage_split的其他参数
    """
    
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    # 获取所有WSI文件
    wsi_files = [f for f in os.listdir(input_dir) 
                 if any(f.lower().endswith(ext) for ext in extensions)]
    
    if not wsi_files:
        print(f"在 {input_dir} 中未找到WSI图像文件")
        return
    
    print(f"找到 {len(wsi_files)} 个WSI文件")
    
    # 处理每个文件
    for wsi_file in wsi_files:
        slide_path = os.path.join(input_dir, wsi_file)
        print(f"\n{'='*60}")
        print(f"正在处理: {wsi_file}")
        print(f"{'='*60}")
        
        try:
            two_stage_split(slide_path, large_output_dir, small_output_dir, **kwargs)
        except Exception as e:
            print(f"处理 {wsi_file} 时出错: {str(e)}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WSI图像两阶段切分工具')
    parser.add_argument('--input', type=str, required=True, 
                        help='输入WSI文件路径或文件夹路径')
    parser.add_argument('--large-output', type=str, default='large_tiles_output',
                        help='大图输出文件夹 (默认: large_tiles_output)')
    parser.add_argument('--small-output', type=str, default='small_tiles_output',
                        help='小图输出文件夹 (默认: small_tiles_output)')
    parser.add_argument('--large-size', type=int, default=5000,
                        help='大图切分尺寸 (默认: 5000)')
    parser.add_argument('--small-size', type=int, default=1000,
                        help='小图切分尺寸 (默认: 1000)')
    parser.add_argument('--scaled-size', type=int, default=1000,
                        help='大图缩放后的尺寸 (默认: 1000)')
    parser.add_argument('--level', type=int, default=0,
                        help='OpenSlide读取层级 (默认: 0)')
    
    args = parser.parse_args()
    
    # 判断输入是文件还是文件夹
    if os.path.isfile(args.input):
        # 处理单个文件
        two_stage_split(
            slide_path=args.input,
            large_output_dir=args.large_output,
            small_output_dir=args.small_output,
            large_tile_size=args.large_size,
            small_tile_size=args.small_size,
            scaled_large_size=args.scaled_size,
            level=args.level
        )
    elif os.path.isdir(args.input):
        # 处理文件夹
        process_directory(
            input_dir=args.input,
            large_output_dir=args.large_output,
            small_output_dir=args.small_output,
            large_tile_size=args.large_size,
            small_tile_size=args.small_size,
            scaled_large_size=args.scaled_size,
            level=args.level
        )
    else:
        print(f"错误: {args.input} 不是有效的文件或文件夹")
