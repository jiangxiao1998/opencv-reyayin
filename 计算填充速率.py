import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

def edge_detection_sobel(input_path, output_path):
    vc = cv2.VideoCapture(input_path)

    if not vc.isOpened():
        print("Error: Couldn't open video file.")
        exit()

    fps = int(vc.get(cv2.CAP_PROP_FPS))
    frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc="Sobel edge detection", unit="frame"):
        ret, frame = vc.read()

        if not ret:
            break

        # Sobel边缘检测
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=5)
        sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_edges_normalized = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX)
        sobel_edges_colored = cv2.cvtColor(np.uint8(sobel_edges_normalized), cv2.COLOR_GRAY2BGR)
        out.write(sobel_edges_colored)

    vc.release()
    out.release()



def calculate_contour_area_and_fill_rate(edges_video_path, skip_frames=10, mutation_threshold=1.3, window_size=21, polyorder=2):
    vc = cv2.VideoCapture(edges_video_path)

    if not vc.isOpened():
        print("Error: Couldn't open video file.")
        exit()

    fps = int(vc.get(cv2.CAP_PROP_FPS))
    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    contour_areas = []
    initial_contour_area = 0  # 初始轮廓面积
    prev_fill_rate = 0

    for frame_num in tqdm(range(0, total_frames, skip_frames), desc="Calculating contour areas", unit="frame"):
        # 将帧定位到指定间隔的位置
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = vc.read()

        if not ret:
            break

        # 寻找轮廓
        contours, _ = cv2.findContours(frame[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 计算轮廓面积
        contour_area = sum(cv2.contourArea(contour) for contour in contours)

        # 计算填充率
        if initial_contour_area > 0:
            fill_rate = (initial_contour_area - contour_area) / initial_contour_area
        else:
            fill_rate = 0

        # 判断是否为突变值
        if frame_num > 0 and fill_rate > prev_fill_rate * mutation_threshold:
            # 将突变值替换为上一个数值或进行其他处理
            fill_rate = prev_fill_rate

        contour_areas.append(contour_area)
        prev_fill_rate = fill_rate

        # 初始化初始轮廓面积
        if frame_num == 0:
            initial_contour_area = contour_area if contour_area > 0 else 1

    vc.release()

    # 对填充率进行滤波
    # 计算每个轮廓区域的填充率
    fill_rates = [(initial_contour_area - area)*100 / initial_contour_area for area in contour_areas]
        
    # 对填充率进行均值化，每skip_frames帧取一次均值
    fill_rates_averaged = [np.mean(fill_rates[i:i+skip_frames]) for i in range(0, len(fill_rates), skip_frames)]

    # 使用Savitzky-Golay滤波器对均值化后的填充率进行平滑处理
#     fill_rates_smoothed = savgol_filter(fill_rates_averaged, window_size, polyorder)
    fill_rates_smoothed = fill_rates_averaged
    return fps, contour_areas, fill_rates_smoothed



def calculate_fill_rates_speed(fill_rates):
    fill_rates_speed = 0.0  # 初始速度为0
    print(len(fill_rates))
    for i in range(1, len(fill_rates)):
        if i >= 14:
            if fill_rates[i]==fill_rates[i-1] or (fill_rates[i] - fill_rates[i-1])/(fill_rates[i]*1.0) <0.0001:
                fill_rates_speed = abs(fill_rates[i] / (i*3.04))
                break
    fill_rates_speed = round(fill_rates_speed,3)
    return fill_rates_speed

def plot_fill_rates_speed_curve(video_files, all_fill_rates, detection_method, output_folder):
    data_points = []
    for i, fill_rates in enumerate(all_fill_rates):
        video_files_name = video_files[i].split(".")[0]
        fill_rates_speed = calculate_fill_rates_speed(fill_rates)
        data_points.append((video_files_name,fill_rates_speed))
        # plt.plot(video_files_name,fill_rates_speed, label=f"{video_files_name} - {detection_method}")
    
    xpoint,ypoint = zip(*data_points)
#     plt.plot(xpoint,ypoint, label=f"{xpoint} - {detection_method}")
    
    # 绘制点
    plt.scatter(xpoint, ypoint, color='red')
    
    # 在点的附近标上数值
    for i, (x, y) in enumerate(data_points):
        plt.text(x,y,f'{y}', ha='right', va='bottom', fontsize=8)
    
    # 连线
    plt.plot(xpoint, ypoint, linestyle='-', color='blue')
    plt.xlabel("Experimental group")
    plt.ylabel("Fill Rate Speed %/s")
    plt.title(f"Fill Rate Speed Curve - {detection_method}")
    plt.legend([])
    plt.grid(True)

    curve_save_path = os.path.join(output_folder, f"fill_rate_speed_curve_{detection_method.lower()}.png")
    plt.savefig(curve_save_path)
    plt.close()
    
def process_videos(input_folder, output_folder, start_x, start_y, end_x, end_y, threshold_value, skip_frames, mutation_threshold, window_size, polyorder):
    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    # 用于保存每个视频的结果
    all_fill_rates_sobel = []
    min_fill_rates_length = float('inf')

    for video_file in video_files:
        original_filename = os.path.splitext(video_file)[0]
        input_path = os.path.join(input_folder, video_file)
        
        gaussian_filtered_video_path = os.path.join(output_folder, f"{original_filename}_gaussian_filtered_video.mp4")

        # Sobel边缘检测
        sobel_edges_video_path = os.path.join(output_folder, f"{original_filename}_sobel_edges_video.mp4")
#         edge_detection_sobel(gaussian_filtered_video_path, sobel_edges_video_path)
        fps_sobel, _, fill_rates_sobel = calculate_contour_area_and_fill_rate(sobel_edges_video_path, skip_frames, mutation_threshold, window_size, polyorder)
        all_fill_rates_sobel.append(fill_rates_sobel)

        # 保存当前视频的填充率最小长度
        min_fill_rates_length = min(min_fill_rates_length, len(fill_rates_sobel))
        
    # 截取相同长度的填充率
    all_fill_rates_sobel = [fill_rates[:min_fill_rates_length] for fill_rates in all_fill_rates_sobel]

    # 绘制曲线图 
    plot_fill_rates_speed_curve(video_files, all_fill_rates_sobel, "Sobel", output_folder)


def main():
    input_folder = "input_folder"  # 修改为包含视频文件的文件夹路径
    output_folder = "output_folder"  # 修改为输出结果的文件夹路径

    start_x, start_y = 1600, 1500  # 修改为裁剪的起始位置
    end_x, end_y = 2300, 2000  # 修改为裁剪的结束位置
    threshold_value = 90  # 二值化阈值
    skip_frames = 10  # 平均帧数
    mutation_threshold = 1.3  # 突变值阈值
    window_size = 21
    polyorder = 2

    process_videos(input_folder, output_folder, start_x, start_y, end_x, end_y, threshold_value, skip_frames, mutation_threshold, window_size, polyorder)


if __name__ == "__main__":
    main()