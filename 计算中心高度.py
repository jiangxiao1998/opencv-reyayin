import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd

def exponential_moving_average(data, alpha):
    series = pd.Series(data)
    return series.ewm(alpha=alpha, adjust=False).mean()

def smooth_distances(distances, window_length=9, polyorder=3,alpha=0.4):
    # Apply Savitzky-Golay filter
    smoothed_distances = savgol_filter(distances, window_length=window_length, polyorder=polyorder)

    # Apply additional smoothing if needed
    smoothed_distances = exponential_moving_average(smoothed_distances, alpha=alpha)

    return smoothed_distances

def extract_edge_coordinates(edges):
    edge_coordinates = np.column_stack(np.where(edges > 0))
    return edge_coordinates

def find_outer_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    outer_contour = max(contours, key=cv2.contourArea)
    return outer_contour

def find_center(frame):
    edges = cv2.Canny(frame, 50, 150)
    edge_coordinates = extract_edge_coordinates(edges)
    contour_y_values = edge_coordinates[:, 0]
    min_y = int(np.min(contour_y_values))
    max_y = int(np.max(contour_y_values))
    delta_y_max = 12
    delta_y_min = 50
    min_y_coordinates = [(coord[1], coord[0]) for coord in edge_coordinates if min_y <= coord[0] <= (min_y + delta_y_min)]
    max_y_coordinates = [(coord[1], coord[0]) for coord in edge_coordinates if (max_y - delta_y_max) <= coord[0] <= max_y]

    return min_y_coordinates, max_y_coordinates

def annotate_video(input_path, output_path, fps):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    video_fps = int(cap.get(5)) if fps is None else fps
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (frame_width, frame_height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Lists to store coordinates for the first 10 frames
    all_min_y_coordinates = []
    all_max_y_coordinates = []

    # Initialize distances list
    distances = []
    final_distances = []
    # Initialize stable centers
    avg_min_y_center = None
    avg_max_y_center = None

    # Initialize initial distance
    initial_distance = 500
    
    #
    something = 10
    
    # Process the first 10 frames
    for frame_number in tqdm(range(min(10, total_frames)), desc=f"Processing {input_path}"):
        ret, frame = cap.read()
        if not ret:
            break

        min_y_coordinates, max_y_coordinates = find_center(frame)

        # Append coordinates to lists
        all_min_y_coordinates.extend(min_y_coordinates)
        all_max_y_coordinates.extend(max_y_coordinates)

        # Write annotated frame to the output video
        out.write(frame)

        # Update centers based on the first frame
        if frame_number == 0:
            avg_min_y_center = np.mean(all_min_y_coordinates, axis=0).astype(int)
            avg_max_y_center = np.mean(all_max_y_coordinates, axis=0).astype(int)

        # Save coordinates for distance calculation
        distances.append(np.linalg.norm(avg_max_y_center - avg_min_y_center))
        
        # Update distance interval (at the 10th frame)

        if frame_number == 9:
            distance_interval = initial_distance / distances[-1]

    # Reset video capture to process the entire video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process the remaining frames using the stable centers
    for frame_number in tqdm(range(10, total_frames), desc=f"Processing {input_path}"):
        ret, frame = cap.read()
        if not ret:
            break

        # Draw circles on the frame for visualization using stable centers
        cv2.circle(frame, (avg_min_y_center[0], avg_min_y_center[1]), 1, (0, 255, 255), 3)
        cv2.circle(frame, (avg_max_y_center[0], avg_max_y_center[1]), 1, (0, 255, 255), 3)

        # Find contours and update centers based on x values
        min_y_coordinates, max_y_coordinates = find_center(frame)
        if min_y_coordinates:
            avg_min_y_center = np.mean(min_y_coordinates, axis=0).astype(int)
        if max_y_coordinates:
            avg_max_y_center = np.mean(max_y_coordinates, axis=0).astype(int)

        # Save coordinates for distance calculation
        distances.append(np.linalg.norm(avg_max_y_center - avg_min_y_center))
        

        # Check if distance_interval is not None before calculating actual_distance
        if distance_interval is not None:
            actual_distance = distance_interval * distances[-1]
            finaal_dis = actual_distance
            
            
            if len(final_distances) > 0 and (actual_distance >= final_distances[-1]):
            # You can choose to ignore, adjust, or take other actions here
            # Here, I'm replacing the current distance with the previous one
                finaal_dis = final_distances[-1]
            
            final_distances.append(finaal_dis)
#             if something==0:
#                 finaal_dis = actual_distance
#                 final_distances.append(finaal_dis)
#                 something = 10
                
#             something-=1;

            # Data smoothing and filtering
            smoothed_distances = savgol_filter(distances, window_length=5, polyorder=3)

            
            # Annotate actual distance on the frame
            cv2.putText(frame, f"Distance: {actual_distance:.2f} micrometers", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write annotated frame to the output video
            out.write(frame)

#     smoothed_distances = savgol_filter(final_distances, window_length=5, polyorder=3)
#     smoothed_distances = exponential_moving_average(smoothed_distances, 0.2)
#     smoothed_distances = smooth_distances(final_distances)
    
    # 对填充率进行均值化，每10帧取一次均值
    
#     smoothed_distances = savgol_filter(smoothed_distances, window_length=5, polyorder=3)
    smoothed_distances = [np.mean(final_distances[i:i+30]) for i in range(0, len(final_distances), 30)]
#     smoothed_distances = [sum(final_distances[i:i+10]) / 10 for i in range(0, len(final_distances), 10) for _ in range(10)]

    smoothed_distances = smooth_distances(smoothed_distances)
    
    
    cap.release()
    out.release()

    return avg_min_y_center, avg_max_y_center, smoothed_distances, video_fps

def plot_combined_center_height_curve(video_files, distances_list, fps_list, output_folder):
    
    min_frames = min(len(distances) for distances in distances_list)
    min_fps = min(fps_list)
    
    time_points = np.arange(min_frames) / min_fps
    
    for i, distances in enumerate(distances_list):
        video_filename = os.path.splitext(video_files[i])[0]
#         seconds_per_frame = 1 / fps_list[i]  # Assuming a constant frame rate

        # Convert frame numbers to seconds
#         time_points = np.arange(len(distances)) * seconds_per_frame

        # Plot the distance curve
        plt.plot(time_points*30, distances[:min_frames], label=f"{video_filename}")

    plt.xlabel('Time (seconds)')
    plt.ylabel('Center Height(micrometers)')
    plt.title('Combined Center Height Curve')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_folder, "combined_center_height_curve.png")
    plt.savefig(plot_path)
    plt.show()

def process_videos(input_folder, output_folder):
    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    distances_list = []
    fps_list = []

    for video_file in video_files:
        original_filename = os.path.splitext(video_file)[0]
        sobel_edges_video_path = os.path.join(output_folder, f"{original_filename}_sobel_edges_video.mp4")
        output_path = os.path.join(output_folder, f"{original_filename}_annotated_sobel_edges_video.mp4")

        avg_min_y_center, avg_max_y_center, smoothed_distances, video_fps = annotate_video(sobel_edges_video_path, output_path, None)
        print(f"Average Min Y Center: {avg_min_y_center}")
        print(f"Average Max Y Center: {avg_max_y_center}")

        distances_list.append(smoothed_distances)
        fps_list.append(video_fps)

    # Call the function to plot the combined center height curve
    plot_combined_center_height_curve(video_files, distances_list, fps_list, output_folder)

def main():
    # In the main function:
    input_folder = "input_folder"  # Modify to the folder containing processed video files
    output_folder = "output_folder"  # Modify to the folder for annotated results
    process_videos(input_folder, output_folder)

if __name__ == "__main__":
    main()
