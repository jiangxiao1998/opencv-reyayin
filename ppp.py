import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

def crop_video(input_path, output_path, start_x, start_y, end_x, end_y):
    vc = cv2.VideoCapture(input_path)

    if not vc.isOpened():
        print("Error: Couldn't open video file.")
        exit()

    fps = int(vc.get(cv2.CAP_PROP_FPS))
    frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (end_x - start_x, end_y - start_y))

    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc="Cropping frames", unit="frame"):
        ret, frame = vc.read()

        if not ret:
            break

        # 裁剪
        cropped_frame = frame[start_y:end_y, start_x:end_x]

        out.write(cropped_frame)

    vc.release()
    out.release()

def smooth_video(input_path, output_path):
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

    for _ in tqdm(range(total_frames), desc="Smoothing frames", unit="frame"):
        ret, frame = vc.read()

        if not ret:
            break

        # 高斯滤波去抖
        smoothed_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        out.write(smoothed_frame)

    vc.release()
    out.release()

def threshold_video(input_path, output_path, threshold_value):
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

    for _ in tqdm(range(total_frames), desc="Thresholding frames", unit="frame"):
        ret, frame = vc.read()

        if not ret:
            break

        # 二值化
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresholded_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)
        thresholded_colored_frame = cv2.cvtColor(thresholded_frame, cv2.COLOR_GRAY2BGR)
        out.write(thresholded_colored_frame)

    vc.release()
    out.release()

def gaussian_filter_video(input_path, output_path):
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

    for _ in tqdm(range(total_frames), desc="Applying Gaussian filter", unit="frame"):
        ret, frame = vc.read()

        if not ret:
            break

        # 高斯滤波
        smoothed_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        out.write(smoothed_frame)

    vc.release()
    out.release()

def edge_detection_canny(input_path, output_path, low_threshold, high_threshold):
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

    for _ in tqdm(range(total_frames), desc="Canny edge detection", unit="frame"):
        ret, frame = vc.read()

        if not ret:
            break

        # Canny边缘检测
        edges = cv2.Canny(frame, low_threshold, high_threshold)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out.write(edges_colored)

    vc.release()
    out.release()

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

def edge_detection_laplacian(input_path, output_path):
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

    for _ in tqdm(range(total_frames), desc="Laplacian edge detection", unit="frame"):
        ret, frame = vc.read()

        if not ret:
            break

        # Laplacian边缘检测
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_edges = cv2.Laplacian(gray_frame, cv2.CV_64F)
        laplacian_edges = np.abs(laplacian_edges)
        laplacian_edges_normalized = cv2.normalize(laplacian_edges, None, 0, 255, cv2.NORM_MINMAX)
        laplacian_edges_colored = cv2.cvtColor(np.uint8(laplacian_edges_normalized), cv2.COLOR_GRAY2BGR)
        out.write(laplacian_edges_colored)

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
    fill_rates_smoothed = savgol_filter(fill_rates_averaged, window_size, polyorder)

    return fps, contour_areas, fill_rates_smoothed


def plot_combined_fill_rate_curve(video_files, fps, all_fill_rates, detection_method, output_folder):
    time_seconds = np.arange(len(all_fill_rates[0]))*3.04  # 将帧数转换为秒  / fps
    # 绘制每个视频的填充率曲线
    for i, fill_rates in enumerate(all_fill_rates):
        video_files_name = video_files[i].split(".")[0]
        plt.plot(time_seconds, fill_rates, label=f"{video_files_name} - {detection_method}")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Fill Rate %")
    plt.title(f"Combined Fill Rate Curve - {detection_method}")
    plt.legend()
    plt.grid(True)

    curve_save_path = os.path.join(output_folder, f"combined_fill_rate_curve_{detection_method.lower()}.png")
    plt.savefig(curve_save_path)
    plt.close()
    
def process_videos(input_folder, output_folder, start_x, start_y, end_x, end_y, threshold_value, skip_frames, mutation_threshold, window_size, polyorder):
    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    # 用于保存每个视频的结果
    all_fill_rates_canny = []
    all_fill_rates_sobel = []
    all_fill_rates_laplacian = []
    min_fill_rates_length = float('inf')

    for video_file in video_files:
        original_filename = os.path.splitext(video_file)[0]
        input_path = os.path.join(input_folder, video_file)

        # 裁剪视频
        cropped_video_path = os.path.join(output_folder, f"{original_filename}_cropped_video.mp4")
        crop_video(input_path, cropped_video_path, start_x, start_y, end_x, end_y)

        # 去抖视频
        smoothed_video_path = os.path.join(output_folder, f"{original_filename}_smoothed_video.mp4")
        smooth_video(cropped_video_path, smoothed_video_path)

        # 二值化视频
        thresholded_video_path = os.path.join(output_folder, f"{original_filename}_thresholded_video.mp4")
        threshold_video(smoothed_video_path, thresholded_video_path, threshold_value)

        # 高斯滤波视频
        gaussian_filtered_video_path = os.path.join(output_folder, f"{original_filename}_gaussian_filtered_video.mp4")
        gaussian_filter_video(thresholded_video_path, gaussian_filtered_video_path)

         # Canny边缘检测
        canny_edges_video_path = os.path.join(output_folder, f"{original_filename}_canny_edges_video.mp4")
        edge_detection_canny(gaussian_filtered_video_path, canny_edges_video_path, 50, 150)
        fps_canny, _, fill_rates_canny = calculate_contour_area_and_fill_rate(canny_edges_video_path, skip_frames, mutation_threshold, window_size, polyorder)
        all_fill_rates_canny.append(fill_rates_canny)

        # Sobel边缘检测
        sobel_edges_video_path = os.path.join(output_folder, f"{original_filename}_sobel_edges_video.mp4")
        edge_detection_sobel(gaussian_filtered_video_path, sobel_edges_video_path)
        fps_sobel, _, fill_rates_sobel = calculate_contour_area_and_fill_rate(sobel_edges_video_path, skip_frames, mutation_threshold, window_size, polyorder)
        all_fill_rates_sobel.append(fill_rates_sobel)

        # Laplacian边缘检测
        laplacian_edges_video_path = os.path.join(output_folder, f"{original_filename}_laplacian_edges_video.mp4")
        edge_detection_laplacian(gaussian_filtered_video_path, laplacian_edges_video_path)
        fps_laplacian, _, fill_rates_laplacian = calculate_contour_area_and_fill_rate(laplacian_edges_video_path, skip_frames, mutation_threshold, window_size, polyorder)
        all_fill_rates_laplacian.append(fill_rates_laplacian)

        # 保存当前视频的填充率最小长度
        min_fill_rates_length = min(min_fill_rates_length, len(fill_rates_canny))
        
    # 截取相同长度的填充率
    all_fill_rates_canny = [fill_rates[:min_fill_rates_length] for fill_rates in all_fill_rates_canny]
    all_fill_rates_sobel = [fill_rates[:min_fill_rates_length] for fill_rates in all_fill_rates_sobel]
    all_fill_rates_laplacian = [fill_rates[:min_fill_rates_length] for fill_rates in all_fill_rates_laplacian]

    # 绘制曲线图
    plot_combined_fill_rate_curve(video_files, fps_canny, all_fill_rates_canny, "Canny", output_folder)
    plot_combined_fill_rate_curve(video_files, fps_sobel, all_fill_rates_sobel, "Sobel", output_folder)
    plot_combined_fill_rate_curve(video_files, fps_laplacian, all_fill_rates_laplacian, "Laplacian", output_folder)



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