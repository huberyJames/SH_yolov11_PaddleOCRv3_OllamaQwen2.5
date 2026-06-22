"""
修复版：大幅扩展台面区域，确保覆盖整个桌面/台面
"""

import cv2
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ==================== 配置区 ====================
VIDEO_PATH = r"E:\sh\video\3parts_Test2.mp4"
OUTPUT_DIR = r"E:\sh\output\yolo_3parts"

SEG_MODEL_PATH = r"E:\sh\models\yolo\yolo11n-seg.pt"
DET_MODEL_PATH = r"E:\sh\models\yolo\yolo11n.pt"

CONF_THRESHOLD = 0.25
IMG_SIZE = 640

REGIONS = ["桌子上", "椅子上", "地上"]

# 字体
FONT_PATHS = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]
FONT_PATH = None
for fp in FONT_PATHS:
    if os.path.exists(fp):
        FONT_PATH = fp
        break
# ==============================================


def put_chinese_text(img, text, position, font_size=20, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size) if FONT_PATH else ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_chinese_label(img, text, position, font_size=16, bg_color=(0, 255, 0), text_color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size) if FONT_PATH else ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x, y = position
    draw.rectangle([x, y - text_h - 4, x + text_w + 8, y + 4], fill=bg_color)
    draw.text((x + 4, y - text_h), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def is_point_in_mask(x, y, mask):
    if mask is None:
        return False
    h, w = mask.shape
    if 0 <= x < w and 0 <= y < h:
        return mask[y, x] > 0
    return False


def create_table_mask_from_scene(height, width, det_results):
    """
    从场景中的物体推断整个台面区域
    关键改进：大幅扩展，覆盖整个水平面
    """
    # 收集台面上所有物体的位置
    surface_objects = []
    
    for box in det_results.boxes:
        cls_id = int(box.cls[0])
        class_name = det_results.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # 这些物体通常放在台面上
        if class_name in ["laptop", "mouse", "keyboard", "book", "cup", "bottle", 
                          "cell_phone", "remote", "sink", "toilet", "hair_drier"]:
            surface_objects.append({
                "bbox": [x1, y1, x2, y2],
                "y_bottom": y2,  # 物体底部y坐标
                "class_name": class_name
            })
    
    if len(surface_objects) == 0:
        return None
    
    # 找到所有物体的最低底部（台面高度）
    # 台面通常是这些物体所在的水平面
    min_y = min(obj["bbox"][1] for obj in surface_objects)  # 最高物体的顶部
    max_y = max(obj["bbox"][3] for obj in surface_objects)    # 最低物体的底部
    
    # 台面高度 = 物体底部平均值（或最低物体的底部）
    table_y = int(np.median([obj["y_bottom"] for obj in surface_objects]))
    
    # 找到物体的左右范围
    min_x = min(obj["bbox"][0] for obj in surface_objects)
    max_x = max(obj["bbox"][2] for obj in surface_objects)
    
    # 大幅扩展台面区域
    # 关键：台面应该覆盖整个画面下方的大部分区域
    table_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 台面上边界：最高物体的顶部上方一点
    table_top = max(0, min_y - 100)
    # 台面下边界：画面底部
    table_bottom = height
    # 台面左右：整个画面宽度（或物体范围大幅扩展）
    table_left = 0
    table_right = width
    
    # 如果物体集中在某个区域，以该区域为中心扩展
    if max_x - min_x < width * 0.6:  # 物体集中在画面的一部分
        center_x = (min_x + max_x) // 2
        half_width = max(max_x - min_x, width // 2) // 2 + 100
        table_left = max(0, center_x - half_width)
        table_right = min(width, center_x + half_width)
    
    table_mask[table_top:table_bottom, table_left:table_right] = 1
    
    # 排除上方区域（墙上/背景）
    table_mask[:height//3, :] = 0
    
    return table_mask


def create_chair_mask(height, width, det_results):
    """创建椅子区域（如果有椅子的话）"""
    chair_mask = np.zeros((height, width), dtype=np.uint8)
    
    for box in det_results.boxes:
        cls_id = int(box.cls[0])
        class_name = det_results.names[cls_id]
        if class_name == "chair":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 扩展椅子区域
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w_half = (x2 - x1) // 2 + 50
            h_half = (y2 - y1) // 2 + 50
            x1_new = max(0, cx - w_half)
            x2_new = min(width, cx + w_half)
            y1_new = max(0, cy - h_half)
            y2_new = min(height, cy + h_half)
            chair_mask[y1_new:y2_new, x1_new:x2_new] = 1
    
    if np.sum(chair_mask) == 0:
        return None
    return chair_mask


def detect_regions_and_objects(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"加载分割模型: {SEG_MODEL_PATH}")
    seg_model = YOLO(SEG_MODEL_PATH)
    print(f"加载检测模型: {DET_MODEL_PATH}")
    det_model = YOLO(DET_MODEL_PATH)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n视频: {width}x{height}, {fps}fps, {total_frames}帧")
    print("=" * 60)
    
    output_video_path = os.path.join(output_dir, "two_step_region_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    region_colors_bgr = {"桌子上": (0, 100, 255), "椅子上": (0, 255, 100), "地上": (100, 100, 255)}
    region_colors_rgb = {"桌子上": (255, 100, 0), "椅子上": (100, 255, 0), "地上": (255, 100, 100)}
    
    results_list = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        # ========== Step 1: 分割区域 ==========
        seg_results = seg_model(frame, verbose=False, conf=CONF_THRESHOLD, imgsz=IMG_SIZE)[0]
        
        region_masks = {"桌子上": None, "椅子上": None, "地上": None}
        
        # 尝试从分割模型获取
        if seg_results.masks is not None and seg_results.boxes is not None:
            for box, mask in zip(seg_results.boxes, seg_results.masks.data):
                cls_id = int(box.cls[0])
                class_name = seg_model.names[cls_id]
                
                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
                mask_np = (mask_np > 0.5).astype(np.uint8)
                mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
                
                if class_name in ["dining_table", "sink"]:
                    region_masks["桌子上"] = mask_resized
                elif class_name == "chair":
                    region_masks["椅子上"] = mask_resized
        
        # ========== Step 2: 检测物体 ==========
        det_results = det_model(frame, verbose=False, conf=CONF_THRESHOLD, imgsz=IMG_SIZE)[0]
        
        # ========== Step 2.5: 如果没检测到桌子，从场景推断 ==========
        table_from_seg = region_masks["桌子上"] is not None and np.sum(region_masks["桌子上"]) > 0
        
        if not table_from_seg:
            # 关键改进：用新函数大幅扩展台面
            inferred_table = create_table_mask_from_scene(height, width, det_results)
            if inferred_table is not None and np.sum(inferred_table) > 0:
                region_masks["桌子上"] = inferred_table
                print(f"  帧{frame_count}: 从场景推断台面 (覆盖 {np.sum(inferred_table)} 像素)")
        
        # 椅子
        if region_masks["椅子上"] is None or np.sum(region_masks["椅子上"]) == 0:
            chair_mask = create_chair_mask(height, width, det_results)
            if chair_mask is not None:
                region_masks["椅子上"] = chair_mask
        
        # 地板 = 整个画面 - 桌子 - 椅子，但只在下方
        floor_mask = np.zeros((height, width), dtype=np.uint8)
        floor_mask[height//2:, :] = 1  # 下半部分默认是地板
        if region_masks["桌子上"] is not None:
            floor_mask[region_masks["桌子上"] > 0] = 0
        if region_masks["椅子上"] is not None:
            floor_mask[region_masks["椅子上"] > 0] = 0
        region_masks["地上"] = floor_mask
        
        # ========== Step 3: 判断物体区域 ==========
        region_targets = {region: [] for region in REGIONS}
        all_objects = []
        
        for box in det_results.boxes:
            cls_id = int(box.cls[0])
            class_name = det_model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            obj_info = {
                "class_name": class_name,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y]
            }
            all_objects.append(obj_info)
            
            # 判断区域
            assigned_region = "地上"
            
            # 优先检查是否在桌子上
            if is_point_in_mask(center_x, center_y, region_masks["桌子上"]):
                assigned_region = "桌子上"
            # 然后检查椅子
            elif is_point_in_mask(center_x, center_y, region_masks["椅子上"]):
                assigned_region = "椅子上"
            
            obj_info["assigned_region"] = assigned_region
            region_targets[assigned_region].append(obj_info)
        
        # ========== 绘制可视化 ==========
        
        # 绘制区域掩码
        for region, mask in region_masks.items():
            if mask is not None and np.sum(mask) > 0:
                color_bgr = region_colors_bgr.get(region, (128, 128, 128))
                color_rgb = region_colors_rgb.get(region, (128, 128, 128))
                
                # 半透明填充
                overlay = frame.copy()
                overlay[mask > 0] = color_bgr
                cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
                
                # 轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color_bgr, 2)
                
                # 区域标签
                if contours:
                    M = cv2.moments(contours[0])
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        frame = put_chinese_text(frame, f"【{region}】", (cx - 50, cy - 15),
                                                font_size=22, color=color_rgb)
        
        # 绘制物体
        for obj in all_objects:
            x1, y1, x2, y2 = obj["bbox"]
            region = obj["assigned_region"]
            color_bgr = region_colors_bgr.get(region, (128, 128, 128))
            color_rgb = region_colors_rgb.get(region, (128, 128, 128))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            
            label = f"{obj['class_name']} {obj['confidence']:.2f} [{region}]"
            frame = draw_chinese_label(frame, label, (x1, y1 - 5),
                                      font_size=15, bg_color=color_rgb)
            
            # 中心点
            cv2.circle(frame, tuple(obj["center"]), 4, (0, 255, 255), -1)
        
        out.write(frame)
        
        # 保存结果
        frame_result = {
            "frame_id": frame_count,
            "timestamp": round(timestamp, 3),
            "has_table": region_masks["桌子上"] is not None and np.sum(region_masks["桌子上"]) > 0,
            "table_from_seg": table_from_seg,
            "has_chair": region_masks["椅子上"] is not None and np.sum(region_masks["椅子上"]) > 0,
            "total_objects": len(all_objects)
        }
        for region in REGIONS:
            frame_result[f"{region}_count"] = len(region_targets[region])
            frame_result[f"{region}_details"] = region_targets[region]
        
        results_list.append(frame_result)
        
        if frame_count % 50 == 0:
            print(f"处理: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
    
    cap.release()
    out.release()
    
    print(f"\n完成！共处理 {frame_count} 帧")
    
    csv_path = os.path.join(output_dir, "two_step_region_results.csv")
    generate_csv(results_list, csv_path)
    
    stats = calculate_stats(results_list)
    stats_path = os.path.join(output_dir, "two_step_region_stats.txt")
    save_stats(stats, stats_path)
    
    print(f"\n输出文件:")
    print(f"视频: {output_video_path}")
    print(f"CSV: {csv_path}")
    print(f"统计: {stats_path}")
    
    return results_list, stats


def generate_csv(results_list, csv_path):
    rows = []
    for frame in results_list:
        base = {
            "frame_id": frame["frame_id"],
            "timestamp_s": frame["timestamp"],
            "has_table": frame["has_table"],
            "table_from_seg": frame["table_from_seg"],
            "has_chair": frame["has_chair"],
            "total_objects": frame["total_objects"]
        }
        for region in REGIONS:
            base[f"{region}_count"] = frame[f"{region}_count"]
        
        has_target = False
        for region in REGIONS:
            for i, obj in enumerate(frame[f"{region}_details"]):
                row = {
                    **base, "region": region, "obj_index": i + 1,
                    "class_name": obj["class_name"], "confidence": obj["confidence"],
                    "center_x": obj["center"][0], "center_y": obj["center"][1],
                    "bbox_x1": obj["bbox"][0], "bbox_y1": obj["bbox"][1],
                    "bbox_x2": obj["bbox"][2], "bbox_y2": obj["bbox"][3]
                }
                rows.append(row)
                has_target = True
        
        if not has_target:
            row = {**base, "region": "none", "obj_index": 0, "class_name": "none",
                   "confidence": 0, "center_x": None, "center_y": None,
                   "bbox_x1": None, "bbox_y1": None, "bbox_x2": None, "bbox_y2": None}
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV已保存: {csv_path}")


def calculate_stats(results_list):
    stats = {
        "total_frames": len(results_list),
        "frames_with_objects": 0,
        "frames_with_table_seg": 0,
        "frames_with_table_inferred": 0,
        "frames_with_chair": 0,
        "region_stats": {}
    }
    for region in REGIONS:
        stats["region_stats"][region] = {
            "frames_with_target": 0, "total_objects": 0,
            "class_distribution": {}, "avg_objects_per_frame": 0
        }
    
    for frame in results_list:
        has_any = False
        if frame["has_table"]:
            if frame["table_from_seg"]:
                stats["frames_with_table_seg"] += 1
            else:
                stats["frames_with_table_inferred"] += 1
        if frame["has_chair"]:
            stats["frames_with_chair"] += 1
        
        for region in REGIONS:
            details = frame[f"{region}_details"]
            if details:
                has_any = True
                r = stats["region_stats"][region]
                r["frames_with_target"] += 1
                r["total_objects"] += len(details)
                for obj in details:
                    cls = obj["class_name"]
                    r["class_distribution"][cls] = r["class_distribution"].get(cls, 0) + 1
        if has_any:
            stats["frames_with_objects"] += 1
    
    for region in REGIONS:
        r = stats["region_stats"][region]
        if stats["total_frames"] > 0:
            r["avg_objects_per_frame"] = round(r["total_objects"] / stats["total_frames"], 2)
            r["detection_rate"] = round(r["frames_with_target"] / stats["total_frames"] * 100, 2)
    
    stats["overall_detection_rate"] = round(stats["frames_with_objects"] / stats["total_frames"] * 100, 2)
    return stats


def save_stats(stats, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("两步法区域检测统计报告\n")
        f.write("(实例分割+场景推断）\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"总帧数: {stats['total_frames']}\n")
        f.write(f"有目标帧数: {stats['frames_with_objects']}\n")
        f.write(f"整体检测率: {stats['overall_detection_rate']}%\n\n")
        
        f.write(f"分割检测到桌子的帧数: {stats['frames_with_table_seg']}\n")
        f.write(f"推断出台面的帧数: {stats['frames_with_table_inferred']}\n")
        f.write(f"检测到椅子的帧数: {stats['frames_with_chair']}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("【各区域物体统计】\n")
        f.write("-" * 60 + "\n\n")
        
        for region, r in stats["region_stats"].items():
            f.write(f"{region}\n")
            f.write(f"检测到物体的帧数: {r['frames_with_target']}\n")
            f.write(f"检测率: {r['detection_rate']}%\n")
            f.write(f"总物体数: {r['total_objects']}\n")
            f.write(f"平均每帧物体: {r['avg_objects_per_frame']}\n")
            if r["class_distribution"]:
                f.write(f"物体类别分布:\n")
                for cls, count in sorted(r["class_distribution"].items(), key=lambda x: -x[1]):
                    f.write(f"    - {cls}: {count} 次\n")
            f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("【推断逻辑说明】\n")
        f.write("-" * 60 + "\n")
        f.write("1. 收集所有台面上的物体（laptop/mouse/cup/bottle等）\n")
        f.write("2. 计算这些物体的位置范围\n")
        f.write("3. 大幅扩展区域覆盖整个台面（左右全宽，上下覆盖物体范围）\n")
        f.write("4. 排除画面上方1/3（避免把墙上物体误判）\n")


if __name__ == "__main__":
    detect_regions_and_objects(VIDEO_PATH, OUTPUT_DIR)