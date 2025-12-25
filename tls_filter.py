#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TLS过滤程序
对TLS分割结果进行过滤，输出可视化图片和统计数据
"""

import os
import json
import pickle
import numpy as np
from PIL import Image
from skimage import measure
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2


class TLSFilter:
    def __init__(self, large_tiles, tls_dir, cell_dir, area_threshold=60000, type2_ratio_threshold=0.7, cell_count_threshold=700):
        """
        初始化过滤器
        
        Args:
            base_dir: 基础数据目录路径
            area_threshold: 面积阈值，默认60000
            type2_ratio_threshold: Type-2细胞比例阈值，默认0.7
            cell_count_threshold: 细胞数量阈值，默认700
        """
        self.images_dir = Path(large_tiles)
        self.predictions_dir = Path(tls_dir)
        self.cells_dir = Path(cell_dir)
        
        # 过滤参数
        self.area_threshold = area_threshold
        self.type2_ratio_threshold = type2_ratio_threshold
        self.cell_count_threshold = cell_count_threshold
        
        # 验证目录存在
        self._validate_directories()
        
    def _validate_directories(self):
        """验证所有必需的目录存在"""
        dirs = [self.images_dir, self.predictions_dir, self.cells_dir]
        for d in dirs:
            if not d.exists():
                raise FileNotFoundError(f"目录不存在: {d}")
    
    def load_original_image(self, image_name):
        """
        加载原始图片
        
        Args:
            image_name: 图片名称
            
        Returns:
            RGB图片数组
        """
        img_path = self.images_dir / image_name
        if not img_path.exists():
            return None
        return np.array(Image.open(img_path))
    
    def load_prediction_mask(self, image_name):
        """
        加载预测结果mask
        
        Args:
            image_name: 图片名称
            
        Returns:
            二值mask，TLS区域为1
        """
        pred_path = self.predictions_dir / f"{image_name}.pkl"
        if not pred_path.exists():
            return None
            
        with open(pred_path, 'rb') as f:
            pred_mask = pickle.load(f)
        
        return pred_mask.astype(np.uint8)
    
    def load_cell_data(self, image_name):
        """
        加载细胞分割数据
        
        Args:
            image_name: 图片名称
            
        Returns:
            包含所有25个小图的细胞数据列表
        """
        base_name = image_name.replace('.png', '')
        all_cells = []
        
        # 加载25个小图的细胞数据 (r0_c0 到 r4_c4)
        for r in range(5):
            for c in range(5):
                json_path = self.cells_dir / f"{base_name}_subtile_r{r}_c{c}.json"
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if 'nuc' in data:
                            # 为每个细胞添加位置偏移量
                            offset_x = c * 200
                            offset_y = r * 200
                            
                            for cell_id, cell_info in data['nuc'].items():
                                cell_copy = cell_info.copy()
                                # 调整centroid坐标
                                if 'centroid' in cell_copy:
                                    cell_copy['centroid'] = [
                                        cell_copy['centroid'][1] / 5 + offset_y,  # y坐标
                                        cell_copy['centroid'][0] / 5 + offset_x   # x坐标
                                    ]
                                all_cells.append(cell_copy)
        
        return all_cells
    
    def extract_tls_regions(self, mask):
        """
        从mask中提取独立的TLS区域
        
        Args:
            mask: 二值mask
            
        Returns:
            标记后的区域图像，每个连通区域有唯一的标签
        """
        labeled_mask = measure.label(mask, connectivity=2)
        return labeled_mask
    
    def compute_region_properties(self, pred_mask, cells):
        """
        计算每个TLS区域的属性（面积、细胞数量、Type-2比例）
        
        Args:
            pred_mask: 预测的TLS mask
            cells: 细胞数据列表
            
        Returns:
            区域属性字典列表
        """
        labeled_regions = self.extract_tls_regions(pred_mask)
        region_properties = []
        
        # 遍历每个TLS区域
        for region_id in range(1, labeled_regions.max() + 1):
            region_mask = (labeled_regions == region_id)
            
            # 计算面积
            area = region_mask.sum() * 2
            
            # 获取该区域内的细胞
            cells_in_region = []
            for cell in cells:
                if 'centroid' in cell:
                    y, x = cell['centroid']
                    y, x = int(y), int(x)
                    if 0 <= y < region_mask.shape[0] and 0 <= x < region_mask.shape[1]:
                        if region_mask[y, x]:
                            cells_in_region.append(cell)
            
            # 计算细胞数量
            cell_count = len(cells_in_region)
            
            # 计算Type-2细胞比例
            type2_count = sum(1 for c in cells_in_region if c.get('type') == 2)
            type0_count = sum(1 for c in cells_in_region if c.get('type') == 0)
            type4_count = sum(1 for c in cells_in_region if c.get('type') == 4)
            denominator = cell_count - type0_count - type4_count
            type2_ratio = type2_count / denominator if denominator > 0 else 0
            
            region_properties.append({
                'region_id': region_id,
                'mask': region_mask,
                'area': area,
                'cell_count': cell_count,
                'type2_count': type2_count,
                'type2_ratio': type2_ratio
            })
        
        return region_properties
    
    def filter_tls_regions(self, region_properties):
        """
        根据阈值过滤TLS区域
        
        Args:
            region_properties: 区域属性列表
            
        Returns:
            过滤后的区域属性列表
        """
        filtered_regions = []
        
        for props in region_properties:
            if (props['area'] >= self.area_threshold and 
                props['cell_count'] >= self.cell_count_threshold and 
                props['type2_ratio'] >= self.type2_ratio_threshold):
                filtered_regions.append(props)
        
        return filtered_regions
    
    def create_filtered_mask(self, image_shape, filtered_regions):
        """
        创建过滤后的mask
        
        Args:
            image_shape: 图片尺寸
            filtered_regions: 过滤后的区域属性列表
            
        Returns:
            过滤后的二值mask
        """
        filtered_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for props in filtered_regions:
            filtered_mask[props['mask']] = 1
        
        return filtered_mask
    
    def visualize_filtered_results(self, original_image, pred_mask, filtered_regions, output_path):
        """
        可视化过滤结果
        
        Args:
            original_image: 原始图片
            pred_mask: 原始预测mask
            filtered_regions: 过滤后的区域列表
            output_path: 输出路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原始图片
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # 原始预测mask
        axes[1].imshow(original_image)
        pred_overlay = np.zeros_like(original_image)
        pred_overlay[pred_mask > 0] = [255, 0, 0]  # 红色
        axes[1].imshow(pred_overlay, alpha=0.4)
        num_pred = self.extract_tls_regions(pred_mask).max()
        axes[1].set_title(f'Original Prediction ({num_pred} regions)', fontsize=14)
        axes[1].axis('off')
        
        # 过滤后的结果
        axes[2].imshow(original_image)
        filtered_overlay = np.zeros_like(original_image)
        for props in filtered_regions:
            filtered_overlay[props['mask']] = [0, 255, 0]  # 绿色
        axes[2].imshow(filtered_overlay, alpha=0.4)
        axes[2].set_title(f'Filtered Result ({len(filtered_regions)} regions)', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_overlay_visualization(self, original_image, filtered_regions, output_path):
        """
        创建叠加可视化图片，在原图上叠加过滤后的TLS区域
        
        Args:
            original_image: 原始图片
            filtered_regions: 过滤后的区域列表
            output_path: 输出路径
        """
        # 创建副本以免修改原图
        overlay_image = original_image.copy()
        
        # 创建mask覆盖层
        overlay = np.zeros_like(original_image)
        for props in filtered_regions:
            overlay[props['mask']] = [0, 255, 0]  # 绿色
        
        # 叠加
        result = cv2.addWeighted(overlay_image, 0.7, overlay, 0.3, 0)
        
        # 在每个区域上标注信息
        for i, props in enumerate(filtered_regions, 1):
            # 计算区域中心
            y_coords, x_coords = np.where(props['mask'])
            if len(y_coords) > 0:
                center_y = int(y_coords.mean())
                center_x = int(x_coords.mean())
                
                # 添加文字标注
                text = f"#{i}"
                cv2.putText(result, text, (center_x, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 保存
        Image.fromarray(result).save(output_path)
    
    def process_all_images(self, output_dir='./filtered_results'):
        """
        处理所有图片，输出过滤结果和统计数据
        
        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建子目录
        vis_dir = output_dir / 'visualizations'
        overlay_dir = output_dir / 'overlays'
        vis_dir.mkdir(exist_ok=True)
        overlay_dir.mkdir(exist_ok=True)
        
        # 获取所有图片
        image_files = sorted([f.name for f in self.images_dir.glob('*.png')])
        print(f"找到 {len(image_files)} 张图片")
        
        # 统计数据
        all_stats = []
        summary_stats = []
        
        print(f"\n开始处理图片...")
        print(f"过滤参数:")
        print(f"  - 面积阈值: {self.area_threshold}")
        print(f"  - Type-2比例阈值: {self.type2_ratio_threshold}")
        print(f"  - 细胞数量阈值: {self.cell_count_threshold}")
        print()
        
        for img_name in tqdm(image_files, desc="处理图片"):
            # 加载数据
            original_image = self.load_original_image(img_name)
            pred_mask = self.load_prediction_mask(img_name)
            cells = self.load_cell_data(img_name)
            
            if original_image is None or pred_mask is None:
                continue
            
            # 计算区域属性
            region_properties = self.compute_region_properties(pred_mask, cells)
            
            # 过滤区域
            filtered_regions = self.filter_tls_regions(region_properties)
            
            # 保存每个区域的详细统计
            for props in filtered_regions:
                all_stats.append({
                    'image_name': img_name,
                    'region_id': props['region_id'],
                    'area': props['area'],
                    'cell_count': props['cell_count'],
                    'type2_count': props['type2_count'],
                    'type2_ratio': props['type2_ratio']
                })
            
            # 保存图片级别的统计
            num_original = len(region_properties)
            num_filtered = len(filtered_regions)
            summary_stats.append({
                'image_name': img_name,
                'original_regions': num_original,
                'filtered_regions': num_filtered,
                'removed_regions': num_original - num_filtered
            })
            
            # 生成可视化
            if num_original > 0:  # 只对有预测结果的图片生成可视化
                vis_path = vis_dir / f"{img_name.replace('.png', '_comparison.png')}"
                self.visualize_filtered_results(original_image, pred_mask, 
                                               filtered_regions, vis_path)
                
                overlay_path = overlay_dir / f"{img_name.replace('.png', '_overlay.png')}"
                self.create_overlay_visualization(original_image, filtered_regions, 
                                                 overlay_path)
        
        # 保存详细统计数据
        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_df.to_csv(output_dir / 'detailed_statistics.csv', index=False)
            print(f"\n详细统计数据已保存到: {output_dir / 'detailed_statistics.csv'}")
            
            # 打印统计摘要
            print(f"\n过滤后的TLS区域统计:")
            print(f"  - 总区域数: {len(all_stats)}")
            print(f"  - 平均面积: {stats_df['area'].mean():.2f}")
            print(f"  - 平均细胞数: {stats_df['cell_count'].mean():.2f}")
            print(f"  - 平均Type-2比例: {stats_df['type2_ratio'].mean():.4f}")
        else:
            print(f"\n警告: 没有找到符合过滤条件的TLS区域")
        
        # 保存图片级别统计
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
        print(f"\n图片级别统计已保存到: {output_dir / 'summary_statistics.csv'}")
        
        # 打印图片统计
        print(f"\n图片处理统计:")
        print(f"  - 处理的图片数: {len(summary_stats)}")
        print(f"  - 原始预测总区域数: {summary_df['original_regions'].sum()}")
        print(f"  - 过滤后总区域数: {summary_df['filtered_regions'].sum()}")
        print(f"  - 移除的区域数: {summary_df['removed_regions'].sum()}")
        print(f"  - 平均每张图过滤后区域数: {summary_df['filtered_regions'].mean():.2f}")
        
        # 保存参数信息
        with open(output_dir / 'filter_parameters.txt', 'w', encoding='utf-8') as f:
            f.write("TLS过滤参数\n")
            f.write("="*50 + "\n")
            f.write(f"面积阈值: {self.area_threshold}\n")
            f.write(f"Type-2比例阈值: {self.type2_ratio_threshold}\n")
            f.write(f"细胞数量阈值: {self.cell_count_threshold}\n")
        
        print(f"\n所有结果已保存到: {output_dir}")
        print(f"  - 可视化对比图: {vis_dir}")
        print(f"  - 叠加图: {overlay_dir}")


def main():
    
    # 创建过滤器（可以自定义参数）
    filter = TLSFilter(
        large_tiles="large_tiles_output",
        tls_dir="tls_pkl",
        cell_dir="small_tiles_output_infer/json",
        area_threshold=60000,      # 面积阈值
        type2_ratio_threshold=0.7,  # Type-2比例阈值
        cell_count_threshold=700    # 细胞数量阈值
    )
    
    # 处理所有图片
    print("开始TLS过滤...")
    filter.process_all_images(output_dir='./filtered_results')
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()
