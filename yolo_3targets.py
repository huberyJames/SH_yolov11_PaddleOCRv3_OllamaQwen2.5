import cv2
import pandas as pd
import os
from ultralytics import YOLO

BASE_MODEL_DIR = r"E:\\sh\\models"
YOLO_MODEL_DIR = os.path.join(BASE_MODEL_DIR, "yolo")    

TARGET_CLASSES = ["person", "car", "truck"]
VIDEO_PATH = r"E:\sh\video\3targets_Test1.mp4"
OUTPUT_DIR = r"E:\\sh\\output\\yolo_3targets"       



def detect_three_classes(video_path, output_dir, target_classes):
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在加载 YOLOv11n 模型...")

    model_det_path = os.path.join(YOLO_MODEL_DIR, "yolo11n.pt")
    model = YOLO(model_det_path)
    
    # 检查类别
    valid_classes = set(model.names.values())
    for cls in target_classes:
        if cls not in valid_classes:
            print(f"'{cls}' 不在类别中，将自动跳过")
        else:
            print(f"'{cls}' 有效")
    

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n视频: {width}x{height}, {fps}fps, {total_frames}帧")
    print("=" * 50)
    

    output_video_path = os.path.join(output_dir, "detected_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print(f"将保存检测视频到: {output_video_path}")
    # ======================================
    
    # 颜色配置（每类一个颜色 BGR格式）
    colors = {
        "person": (0, 255, 0),      # 绿色
        "car": (0, 0, 255),          # 红色
        "truck": (0, 255, 255)  # 黄色
    }
    
    results_list = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        # 初始化
        frame_targets = {cls: [] for cls in target_classes}
        
        # YOLO检测
        detections = model(frame, verbose=False)[0]
        
        # 遍历检测结果
        for box in detections.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            if class_name in target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                frame_targets[class_name].append({
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2]
                })
                
                # ========== 在画面上画框和标签 ==========
                color = colors.get(class_name, (255, 255, 255))
                
                # 画矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 画标签背景
                label = f"{class_name} {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                
                # 写文字
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                # ======================================
        
        # 写入输出视频
        out.write(frame)
        
        # 构建结果
        frame_result = {
            "frame_id": frame_count,
            "timestamp": round(timestamp, 3),
            "total_targets": sum(len(v) for v in frame_targets.values())
        }
        for cls in target_classes:
            frame_result[f"{cls}_count"] = len(frame_targets[cls])
            frame_result[f"{cls}_details"] = frame_targets[cls]
        
        results_list.append(frame_result)
        
        if frame_count % 30 == 0:
            print(f"处理: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
    
    # 释放资源
    cap.release()
    out.release()
    print(f"\n完成！共处理 {frame_count} 帧")
    
    # 生成CSV
    csv_path = os.path.join(output_dir, "detection_results.csv")
    generate_csv(results_list, csv_path, target_classes)
    
    # 统计
    stats = calculate_stats(results_list, target_classes)
    stats_path = os.path.join(output_dir, "accuracy_stats.txt")
    save_stats(stats, stats_path)
    
    print(f"\n输出文件:")
    print(f"检测视频: {output_video_path}")
    print(f"CSV结果: {csv_path}")
    print(f"统计报告: {stats_path}")
    
    return results_list, stats


def generate_csv(results_list, csv_path, target_classes):
    rows = []
    for frame in results_list:
        base = {"frame_id": frame["frame_id"], "timestamp_s": frame["timestamp"]}
        for cls in target_classes:
            base[f"{cls}_count"] = frame[f"{cls}_count"]
        
        has_target = False
        for cls in target_classes:
            for i, obj in enumerate(frame[f"{cls}_details"]):
                row = {
                    **base,
                    "target_class": cls,
                    "obj_index": i + 1,
                    "confidence": obj["confidence"],
                    "bbox_x1": obj["bbox"][0],
                    "bbox_y1": obj["bbox"][1],
                    "bbox_x2": obj["bbox"][2],
                    "bbox_y2": obj["bbox"][3]
                }
                rows.append(row)
                has_target = True
        
        if not has_target:
            row = {**base, "target_class": "none", "obj_index": 0,
                   "confidence": 0, "bbox_x1": None, "bbox_y1": None,
                   "bbox_x2": None, "bbox_y2": None}
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV已保存")


def calculate_stats(results_list, target_classes):
    stats = {
        "total_frames": len(results_list),
        "frames_with_target": 0,
        "class_stats": {}
    }
    for cls in target_classes:
        stats["class_stats"][cls] = {"frames_detected": 0, "total_instances": 0, "confidences": []}
    
    for frame in results_list:
        has_any = False
        for cls in target_classes:
            details = frame[f"{cls}_details"]
            if details:
                has_any = True
                s = stats["class_stats"][cls]
                s["frames_detected"] += 1
                s["total_instances"] += len(details)
                for obj in details:
                    s["confidences"].append(obj["confidence"])
        if has_any:
            stats["frames_with_target"] += 1
    
    for cls in target_classes:
        s = stats["class_stats"][cls]
        confs = s["confidences"]
        if confs:
            s["avg_confidence"] = round(sum(confs)/len(confs), 4)
            s["max_confidence"] = round(max(confs), 4)
            s["high_conf_ratio"] = round(sum(1 for c in confs if c > 0.8)/len(confs)*100, 2)
            s["detection_rate"] = round(s["frames_detected"]/stats["total_frames"]*100, 2)
        else:
            for k in ["avg_confidence", "max_confidence", "high_conf_ratio", "detection_rate"]:
                s[k] = 0
    
    stats["overall_detection_rate"] = round(stats["frames_with_target"]/stats["total_frames"]*100, 2)
    return stats


def save_stats(stats, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("YOLOv11n 三类目标检测统计报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总帧数: {stats['total_frames']}\n")
        f.write(f"有目标帧数: {stats['frames_with_target']}\n")
        f.write(f"整体检测率: {stats['overall_detection_rate']}%\n\n")
        for cls, s in stats["class_stats"].items():
            f.write(f"{cls}\n")
            f.write(f"检测帧数: {s['frames_detected']}\n")
            f.write(f"检测率: {s['detection_rate']}%\n")
            f.write(f"总实例: {s['total_instances']}\n")
            f.write(f"平均置信度: {s.get('avg_confidence', 0)}\n")
            f.write(f"最高置信度: {s.get('max_confidence', 0)}\n")
            f.write(f"高置信度占比: {s.get('high_conf_ratio', 0)}%\n\n")


if __name__ == "__main__":
    detect_three_classes(VIDEO_PATH, OUTPUT_DIR, TARGET_CLASSES)