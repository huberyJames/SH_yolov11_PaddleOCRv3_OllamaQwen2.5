import cv2
import numpy as np
import os
import json
from datetime import timedelta
from collections import deque
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# =====================================================
# 模型路径配置（统一使用 BASE_MODEL_DIR）
# =====================================================
BASE_MODEL_DIR = r"C:\\Users\\Administrator\\Desktop\\sh\\models"
YOLO_MODEL_DIR = os.path.join(BASE_MODEL_DIR, "yolo")      # YOLO 模型存放路径
PADDLEOCR_MODEL_DIR = os.path.join(BASE_MODEL_DIR, "paddleocr")  # PaddleOCR 模型存放路径

# =====================================================
# 配置参数
# =====================================================
CONFIG = {
    # 模型配置
    "model_path": os.path.join(YOLO_MODEL_DIR, "yolo11n-pose.pt"),  # YOLO11n-Pose 预训练权重
    "conf_threshold": 0.5,               # 检测置信度阈值
    "iou_threshold": 0.45,               # NMS IoU阈值

    # 弯腰检测参数
    "bend_angle_threshold": 30,          # 躯干角度大于此值判定为弯腰（侧面）
    "bend_ratio_threshold": 1.5,         # 躯干/肩宽比例小于此值判定为弯腰（正面）
    "min_bend_duration": 0.5,            # 最小弯腰持续时间（秒），过滤瞬时误判
    "smooth_window": 5,                  # 平滑窗口大小

    # 可视化参数
    "show_skeleton": True,               # 是否显示骨骼连线
    "show_keypoints": True,              # 是否显示关键点
    "font_scale": 0.7,                   # 字体大小
    "line_thickness": 2,                 # 线条粗细

    # 输出配置
    "output_dir": r"C:\\Users\\Administrator\\Desktop\\sh\\output\\bend",  # 输出目录
    "save_video": True,                  # 是否保存视频
    "save_report": True,                 # 是否保存报告
    "video_fps": 30,                     # 输出视频帧率
    "video_codec": "mp4v",               # 视频编码器
}

# =====================================================
# COCO 人体关键点定义（YOLO11-Pose 使用）
# =====================================================
KEYPOINT_NAMES = [
    "nose",           # 0  鼻子
    "left_eye",       # 1  左眼
    "right_eye",      # 2  右眼
    "left_ear",       # 3  左耳
    "right_ear",      # 4  右耳
    "left_shoulder",  # 5  左肩
    "right_shoulder", # 6  右肩
    "left_elbow",     # 7  左手肘
    "right_elbow",    # 8  右手肘
    "left_wrist",     # 9  左手腕
    "right_wrist",    # 10 右手腕
    "left_hip",       # 11 左髋部
    "right_hip",      # 12 右髋部
    "left_knee",      # 13 左膝盖
    "right_knee",     # 14 右膝盖
    "left_ankle",     # 15 左脚踝
    "right_ankle",    # 16 右脚踝
]

# 骨骼连接定义（用于可视化）
SKELETON_CONNECTIONS = [
    [5, 7], [7, 9],     # 左臂
    [6, 8], [8, 10],    # 右臂
    [5, 6],             # 肩膀
    [5, 11], [6, 12],   # 躯干两侧
    [11, 12],           # 髋部
    [11, 13], [13, 15], # 左腿
    [12, 14], [14, 16], # 右腿
    [0, 1], [0, 2],     # 鼻子到眼睛
    [1, 3], [2, 4],     # 眼睛到耳朵
    [0, 5], [0, 6],     # 鼻子到肩膀
]

# 关键点颜色映射
KEYPOINT_COLORS = {
    "head": (0, 255, 255),      # 青色 - 头部
    "body": (100, 255, 100),    # 绿色 - 躯干
    "arm": (255, 100, 100),     # 红色 - 手臂
    "leg": (255, 255, 0),       # 黄色 - 腿部
}


def get_keypoint_color(idx):
    """根据关键点索引返回对应颜色"""
    if idx <= 4:
        return KEYPOINT_COLORS["head"]
    elif idx in [5, 6, 11, 12]:
        return KEYPOINT_COLORS["body"]
    elif idx in [7, 8, 9, 10]:
        return KEYPOINT_COLORS["arm"]
    else:
        return KEYPOINT_COLORS["leg"]


def load_chinese_font(size=20):
    """
    加载中文字体
    尝试多个常见字体路径，返回可用的字体对象
    """
    font_paths = [
        r"C:\\Windows\\Fonts\\msyh.ttc",      # 微软雅黑
        r"C:\\Windows\\Fonts\\simhei.ttf",    # 黑体
        r"C:\\Windows\\Fonts\\simsun.ttc",    # 宋体
        r"C:\\Windows\\Fonts\\msyhbd.ttc",    # 微软雅黑粗体
        r"C:\\Windows\\Fonts\\simkai.ttf",    # 楷体
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue

    print("[WARNING] 未找到中文字体，使用默认字体（中文可能显示为方框）")
    return ImageFont.load_default()


class BendDetector:
    """
    弯腰动作检测器
    基于人体关键点几何特征进行弯腰动作识别
    支持正面和侧面姿态
    """

    def __init__(self, config=None):
        """初始化检测器"""
        self.config = config or CONFIG
        self.model = None
        self.chinese_font = None
        self.chinese_font_small = None
        self._init_model()
        self._init_font()

        # 状态跟踪
        self.bend_states = deque(maxlen=self.config["smooth_window"])
        self.is_bending = False
        self.bend_start_time = None
        self.bend_events = []  # 记录弯腰事件

    def _init_model(self):
        """加载 YOLO11-Pose 模型"""
        model_path = self.config["model_path"]
        print(f"[INFO] 正在加载模型: {model_path}")

        if not os.path.exists(model_path):
            print(f"[WARNING] 模型文件不存在: {model_path}")
            print(f"[INFO] 尝试从默认路径加载...")
            try:
                self.model = YOLO("yolo11n-pose.pt")
            except Exception as e:
                print(f"[ERROR] 自动下载失败: {e}")
                print(f"[INFO] 请确保模型文件已放置在: {YOLO_MODEL_DIR}")
                raise
        else:
            try:
                self.model = YOLO(model_path)
                print("[INFO] 模型加载成功!")
            except Exception as e:
                print(f"[ERROR] 模型加载失败: {e}")
                raise

    def _init_font(self):
        """加载中文字体"""
        print("[INFO] 正在加载中文字体...")
        self.chinese_font = load_chinese_font(size=20)
        self.chinese_font_small = load_chinese_font(size=16)
        print("[INFO] 字体加载完成")

    def draw_chinese_text(self, frame, text, position, font_size=20, color=(255, 255, 255),
                         bg_color=None, bg_padding=5):
        """
        使用 Pillow 在 OpenCV 图像上绘制中文文字
        """
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        if font_size <= 16:
            font = self.chinese_font_small
        else:
            font = self.chinese_font

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x, y = position

        if bg_color is not None:
            bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
            draw.rectangle(
                [(x - bg_padding, y - bg_padding),
                 (x + text_w + bg_padding, y + text_h + bg_padding)],
                fill=bg_rgb
            )

        text_rgb = (color[2], color[1], color[0])
        draw.text((x, y), text, font=font, fill=text_rgb)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def calculate_torso_metrics(self, keypoints):
        """
        计算躯干相关指标
        返回：躯干角度、躯干/肩宽比例、置信度
        """
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]

        conf_shoulder = min(keypoints[5][2], keypoints[6][2])
        conf_hip = min(keypoints[11][2], keypoints[12][2])
        confidence = min(conf_shoulder, conf_hip)

        if confidence < 0.3:
            return None, None, 0

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        # 躯干向量
        torso_vector = shoulder_center - hip_center

        # 指标1：躯干与垂直方向的夹角（侧面弯腰有效）
        vertical = np.array([0, -1])
        torso_angle = np.degrees(np.arccos(
            np.clip(np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) + 1e-6), -1, 1)
        ))

        # 指标2：躯干长度 / 肩宽 比例（正面弯腰有效）
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        torso_length = np.linalg.norm(torso_vector)
        ratio = torso_length / shoulder_width if shoulder_width > 0 else 999

        return torso_angle, ratio, confidence

    def is_bending_pose(self, keypoints):
        """
        判断是否为弯腰动作
        综合使用躯干角度（侧面）和躯干/肩宽比例（正面）
        """
        torso_angle, ratio, confidence = self.calculate_torso_metrics(keypoints)

        if torso_angle is None:
            return False, {'confidence': confidence}

        # 弯腰判定：角度大（侧面） OR 比例小（正面投影缩短）
        is_bending = (torso_angle > self.config["bend_angle_threshold"]) or (ratio < self.config["bend_ratio_threshold"])

        scores = {
            'torso_angle': round(torso_angle, 1),
            'torso_ratio': round(ratio, 2),
            'confidence': round(confidence, 2),
            'is_bending': is_bending
        }

        return is_bending, scores

    def smooth_detection(self, current_bend):
        self.bend_states.append(1 if current_bend else 0)

        if len(self.bend_states) < 3:
            return current_bend

        avg_state = sum(self.bend_states) / len(self.bend_states)
        return avg_state > 0.5

    def draw_skeleton(self, frame, keypoints, color=(0, 255, 0)):
        """
        在图像上绘制人体骨骼
        """
        h, w = frame.shape[:2]

        for connection in SKELETON_CONNECTIONS:
            pt1_idx, pt2_idx = connection
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]

            if pt1[2] > 0.3 and pt2[2] > 0.3:
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                cv2.line(frame, (x1, y1), (x2, y2), color, self.config["line_thickness"])

        for i, kp in enumerate(keypoints):
            if kp[2] > 0.3:
                x, y = int(kp[0]), int(kp[1])
                kp_color = get_keypoint_color(i)
                cv2.circle(frame, (x, y), 4, kp_color, -1)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)

    def draw_bend_info(self, frame, bbox, is_bending, scores, person_id=0):
        x1, y1, x2, y2 = map(int, bbox)

        box_color = (0, 0, 255) if is_bending else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        if is_bending:
            label = "弯腰"
            label_color = (0, 0, 255)
            bg_color = (255, 255, 255)
        else:
            label = "正常"
            label_color = (0, 255, 0)
            bg_color = (255, 255, 255)

        frame[:] = self.draw_chinese_text(
            frame, label, (x1 + 5, y1 - 30),
            font_size=20, color=label_color, bg_color=bg_color, bg_padding=3
        )

        info_lines = []
        if scores.get('torso_angle') is not None:
            info_lines.append(f"躯干角度: {scores['torso_angle']:.1f}\u00b0")
        if scores.get('torso_ratio') is not None:
            info_lines.append(f"躯干比例: {scores['torso_ratio']:.2f}")

        for i, line in enumerate(info_lines):
            y_pos = y2 + 20 + i * 25
            frame[:] = self.draw_chinese_text(
                frame, line, (x1 + 5, y_pos),
                font_size=16, color=(255, 255, 255)
            )

    def draw_global_info(self, frame, frame_count, total_frames, bend_frames):
        info_text = f"帧: {frame_count}/{total_frames} | 弯腰帧: {bend_frames}"
        frame[:] = self.draw_chinese_text(
            frame, info_text, (10, 10),
            font_size=18, color=(255, 255, 255), bg_color=(0, 0, 0), bg_padding=5
        )

    def process_video(self, video_path, output_path=None):
        print(f"[INFO] 开始处理视频: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"[INFO] 视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧, 时长{duration:.2f}s")

        os.makedirs(self.config["output_dir"], exist_ok=True)

        if output_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.config["output_dir"], f"{video_name}_bend_detected.mp4")

        if self.config["save_video"]:
            fourcc = cv2.VideoWriter_fourcc(*self.config["video_codec"])
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        bend_frames = 0
        current_bend_event = None

        print("[INFO] 正在处理帧...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps

            results = self.model(frame, conf=self.config["conf_threshold"],
                                iou=self.config["iou_threshold"], verbose=False)

            frame_bend_detected = False

            for result in results:
                if result.keypoints is None:
                    continue

                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []

                for person_idx, (kpts, bbox) in enumerate(zip(keypoints_data, boxes)):
                    is_bending, scores = self.is_bending_pose(kpts)

                    smoothed_bend = self.smooth_detection(is_bending)

                    if smoothed_bend:
                        frame_bend_detected = True
                        bend_frames += 1

                    if self.config["show_skeleton"]:
                        skeleton_color = (0, 0, 255) if smoothed_bend else (0, 255, 0)
                        self.draw_skeleton(frame, kpts, skeleton_color)

                    self.draw_bend_info(frame, bbox, smoothed_bend, scores, person_idx)

            if frame_bend_detected:
                if current_bend_event is None:
                    current_bend_event = {
                        "start_frame": frame_count,
                        "start_time": current_time,
                        "end_frame": frame_count,
                        "end_time": current_time,
                    }
                else:
                    current_bend_event["end_frame"] = frame_count
                    current_bend_event["end_time"] = current_time
            else:
                if current_bend_event is not None:
                    duration = current_bend_event["end_time"] - current_bend_event["start_time"]
                    if duration >= self.config["min_bend_duration"]:
                        self.bend_events.append(current_bend_event)
                    current_bend_event = None

            self.draw_global_info(frame, frame_count, total_frames, bend_frames)

            if self.config["save_video"]:
                out.write(frame)

            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"  进度: {progress:.1f}% ({frame_count}/{total_frames})")

        if current_bend_event is not None:
            duration = current_bend_event["end_time"] - current_bend_event["start_time"]
            if duration >= self.config["min_bend_duration"]:
                self.bend_events.append(current_bend_event)

        cap.release()
        if self.config["save_video"]:
            out.release()

        print(f"[INFO] 视频处理完成!")

        results = self._generate_report(video_path, frame_count, fps, bend_frames)

        return results

    def _generate_report(self, video_path, total_frames, fps, bend_frames):
        """
        生成弯腰动作识别报告
        """
        duration = total_frames / fps if fps > 0 else 0
        bend_duration = bend_frames / fps if fps > 0 else 0

        report = {
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": round(duration, 2),
            "bend_frames": bend_frames,
            "bend_duration_seconds": round(bend_duration, 2),
            "bend_percentage": round((bend_frames / total_frames) * 100, 2) if total_frames > 0 else 0,
            "bend_events": [],
            "bend_timestamps": []
        }

        for i, event in enumerate(self.bend_events, 1):
            start_time = timedelta(seconds=int(event["start_time"]))
            end_time = timedelta(seconds=int(event["end_time"]))
            event_duration = event["end_time"] - event["start_time"]

            event_info = {
                "event_id": i,
                "start_frame": event["start_frame"],
                "end_frame": event["end_frame"],
                "start_time": str(start_time),
                "end_time": str(end_time),
                "duration_seconds": round(event_duration, 2)
            }
            report["bend_events"].append(event_info)
            report["bend_timestamps"].append({
                "start": event["start_time"],
                "end": event["end_time"]
            })

        if self.config["save_report"]:
            report_path = os.path.join(self.config["output_dir"], "bend_detection_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 报告已保存: {report_path}")

            txt_path = os.path.join(self.config["output_dir"], "bend_detection_report.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("    YOLOv11 人体弯腰动作识别报告\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"视频文件: {video_path}\n")
                f.write(f"总帧数: {total_frames}\n")
                f.write(f"帧率: {fps:.2f} fps\n")
                f.write(f"视频时长: {duration:.2f} 秒\n\n")
                f.write(f"弯腰帧数: {bend_frames}\n")
                f.write(f"弯腰时长: {bend_duration:.2f} 秒\n")
                f.write(f"弯腰占比: {report['bend_percentage']:.2f}%\n\n")
                f.write(f"弯腰事件数: {len(self.bend_events)}\n")
                f.write("-" * 60 + "\n")

                for event in report["bend_events"]:
                    f.write(f"\n事件 #{event['event_id']}:\n")
                    f.write(f"起始时间: {event['start_time']} (帧 {event['start_frame']})\n")
                    f.write(f"结束时间: {event['end_time']} (帧 {event['end_frame']})\n")
                    f.write(f"持续时间: {event['duration_seconds']:.2f} 秒\n")

                f.write("\n" + "=" * 60 + "\n")
            print(f"[INFO] 文本报告已保存: {txt_path}")

        return report


def main():
    """
    主函数 - 弯腰动作识别演示
    """
    print("=" * 60)
    print("YOLOv11 人体弯腰动作自动识别系统")
    print("=" * 60)
    print(f"基础模型路径: {BASE_MODEL_DIR}")
    print(f"YOLO模型路径: {YOLO_MODEL_DIR}")
    print(f"PaddleOCR模型路径: {PADDLEOCR_MODEL_DIR}")
    print("=" * 60)

    detector = BendDetector()

    video_path = r"C:\\Users\\Administrator\\Desktop\\sh\\video\\bend2.mp4"

    if not os.path.exists(video_path):
        print(f"[WARNING] 视频文件不存在: {video_path}")
        print("[INFO] 请将视频文件放置到指定路径，或修改 video_path 变量")
        print("[INFO] 支持的格式: .mp4, .avi, .mov, .mkv 等")
        print("\n示例路径:")
        print(f"  视频输入: C:\\Users\\Administrator\\Desktop\\sh\\video\\bend2.mp4")
        print(f"  模型目录: {YOLO_MODEL_DIR}")
        print(f"  输出目录: {CONFIG['output_dir']}")
        return

    results = detector.process_video(video_path)

    print("\n" + "=" * 60)
    print("  识别结果摘要")
    print("=" * 60)
    print(f"视频时长: {results['duration_seconds']:.2f} 秒")
    print(f"弯腰时长: {results['bend_duration_seconds']:.2f} 秒")
    print(f"弯腰占比: {results['bend_percentage']:.2f}%")
    print(f"弯腰事件: {len(results['bend_events'])} 次")

    if results['bend_events']:
        print("\n弯腰时间点:")
        for event in results['bend_events']:
            print(f"  #{event['event_id']}: {event['start_time']} ~ {event['end_time']} "
                  f"(持续 {event['duration_seconds']:.2f}s)")

    print("\n" + "=" * 60)
    print("  处理完成！")
    print(f"  输出目录: {CONFIG['output_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
