# opencv-python
import cv2
# mediapipe人工智能工具包
import mediapipe as mp
import numpy as np
import array as arr
import os
# 导入solution
mp_pose = mp.solutions.pose

# # 导入绘图函数
mp_drawing = mp.solutions.drawing_utils 

# 导入模型
pose = mp_pose.Pose(static_image_mode=False,        # 是静态图片还是连续视频帧
                    model_complexity=2,            # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
                    smooth_landmarks=True,         # 是否平滑关键点
                    min_detection_confidence=0.7,  # 置信度阈值
                    min_tracking_confidence=0.7)   # 追踪阈值

# 三维距离有误差，目前只计算二维距离
def get_landmark(results, i):
    return np.array([results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y])

def extract_frames(input_video, output_video, start_frame, end_frame, max_speed) -> None:
    # 打开输入视频文件
    cap = cv2.VideoCapture(input_video)
    # 获取输入视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    my_image = cv2.imread('./static/basketball.png')
    my_image_height, my_image_width, _= my_image.shape
    # 设置开始和结束帧的范围
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 逐帧读取并写入输出视频
    frame_count = start_frame
    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        # 对帧进行处理（例如，可以在这里添加滤镜或其他操作）
        overlay = frame.copy()
        overlay[(height - my_image_height - 100):(height - 100), (0 + 50):(my_image_width + 50)] = my_image
        # 增加速度
        overlay = cv2.putText(overlay, str(round(max_speed,2)), (85,(height - 300)), cv2.FONT_HERSHEY_SIMPLEX, 4, (64, 64, 64), 12)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        # 将帧写入输出视频
        out.write(frame)

        frame_count += 1

    # 释放资源
    cap.release()
    out.release()

def process_frame(img):
    # 初始化
    body_height =0; mid_hip = np.array([0,0]); right_hip = np.array([0,0]); left_hip = np.array([0,0]); right_shoulder = np.array([0,0])
    left_shoulder = np.array([0,0]); right_knee = np.array([0,0]); left_knee = np.array([0,0]); right_ankle = np.array([0,0]); left_ankle = np.array([0,0])
    
    # 获取图像宽高
    h, w = img.shape[0], img.shape[1]
    
    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = pose.process(img_RGB)

    if results.pose_landmarks: # 若检测出人体关键点
        # 计算人体身高
        left_knee = get_landmark(results, 25)
        left_ankle = get_landmark(results, 27)
        left_hip = get_landmark(results, 23)
        left_shoulder= get_landmark(results, 11)
        right_knee = get_landmark(results, 26)
        right_ankle = get_landmark(results, 28)
        right_hip = get_landmark(results, 24)
        right_shoulder= get_landmark(results, 12)
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2
        left_foot = get_landmark(results, 31)
        right_foot = get_landmark(results, 32)
        mid_ankle = (left_ankle + right_ankle) / 2
        mid_foot = (left_foot + right_foot) / 2
        mid_knee = (left_knee + right_knee) / 2
        # body_height 计算 分三段，肩关节 -> 髋关节， 髋关节 -> 膝关节， 膝关节 -> 踝关节
        body_height = np.linalg.norm(mid_shoulder - mid_hip) + np.linalg.norm(mid_hip - mid_knee) + np.linalg.norm(mid_knee - mid_ankle)
        # 可视化关键点及骨架连线
        # mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        for i in range(33): # 遍历所有33个关键点，可视化
            # 获取该关键点的三维坐标
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            # cz = results.pose_landmarks.landmark[i].z
            
            radius = 5

            if i in [11,12]: # 肩膀
                img = cv2.circle(img,(cx,cy), radius, (223,155,6), -1)
            elif i in [23,24]: # 髋关节
                img = cv2.circle(img,(cx,cy), radius, (1,240,255), -1)
            elif i in [13,14]: # 肘
                img = cv2.circle(img,(cx,cy), radius, (140,47,240), -1)
            elif i in [25,26]: # 膝盖
                img = cv2.circle(img,(cx,cy), radius, (0,0,255), -1)
            elif i in [15]: # 左手腕
                img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
            elif i in [16]: # 右手腕
                img = cv2.circle(img,(cx,cy), radius, (16,144,247), -1)
            elif i in [27,29,31]: # 左脚
                img = cv2.circle(img,(cx,cy), radius, (29,123,243), -1)
            elif i in [28,30,32]: # 右脚
                img = cv2.circle(img,(cx,cy), radius, (193,182,255), -1)
            elif i in [2,5]: # 眼及脸颊
                img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)     
        # 展示图片
        # look_img(img)
    else:
        # 没有检测到人体
        return img,0,0,0,0,0,0
        
    # 记录该帧处理完毕的时间
    # end_time = time.time()
    # 计算每秒处理图像帧数FPS
    # FPS = 1/(end_time - start_time)
    # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    #img = cv2.putText(img, 'FPS  '+str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    return img, body_height, -mid_hip[1], -mid_shoulder[1], -mid_knee[1], -mid_ankle[1], -mid_foot[1]


def speed_cal(ref_height, filename, file_path, method = 'cmj', height_type = 'bodyheight'):
    
    # 获取视频总帧数
    input_path = os.path.join(file_path, filename)
    cap = cv2.VideoCapture(input_path)
    frame_count = int(0)
    while(cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    # print('视频总帧数为',frame_count)
    if frame_count > 1000:
        # 使用raise来增加错误处理，若视频帧数过多，剪辑视频
        return -10, ''

    
    # 初始化各个变量，返回髋关节以及身高数组
    speed_arr = arr.array('f')
    foot_gap_arr = arr.array('f')
    mid_hip = 0
    mid_shoulder = 0
    mid_knee = 0
    mid_ankle = 0
    mid_foot = 0
    body_height_last = 0
    mid_hip_last = 0
    mid_shoulder_last = 0
    mid_knee_last = 0
    mid_ankle_last = 0
    mid_foot_last = 0
    max_speed = 0
    max_speed_2 = 0
    first_jump_frame_tag = False
    frame_count_new = int(0)
    # cv2.namedWindow('Crack Detection and Measurement Video Processing')
    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_n = int(round(fps/30))
    # 进度条绑定视频总帧数
    while(cap.isOpened()):
        success, frame = cap.read()
        if not success:
            break

        else:
            frame, body_height, mid_hip, mid_shoulder, mid_knee, mid_ankle, mid_foot= process_frame(frame)
            # body_height = 0 则表示 人体识别不成功， 不计算速度
            if body_height > 0  and body_height_last > 0 and frame_count_new > 10:
                mid_gap = round((mid_hip - mid_hip_last + mid_shoulder - mid_shoulder_last + mid_knee - mid_knee_last + mid_ankle - mid_ankle_last)/4,3)
                if method == 'cmj' and height_type == 'ankel2shoulder':
                    speed = round( mid_gap * fps * (ref_height / body_height), 3)
                elif method == 'cmj' and height_type == 'bodyheight':
                    # 当输入为身高时，则取 body_height / 0.77 为 推断身高
                    speed = round(mid_gap * fps * (ref_height / body_height * 0.77), 3)
                else:
                    print('不存在该方法')
            else:
                mid_gap = 0
                speed = 0
            speed_arr.append(speed)
            foot_gap_arr.append(mid_foot - mid_foot_last)
            max_speed = max(max_speed, speed)
            if (foot_gap_arr[frame_count_new] > 5 * np.std(foot_gap_arr[(frame_count_new - 5):frame_count_new])) and (not first_jump_frame_tag) and frame_count_new > 20 and speed > 1:
                first_jump_frame_tag = True
                jump_frame = frame_count_new
        
            frame_count_new += 1
            body_height_last = body_height
            mid_hip_last = mid_hip
            mid_shoulder_last = mid_shoulder
            mid_knee_last = mid_knee
            mid_ankle_last = mid_ankle
            mid_foot_last = mid_foot
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
    # 双脚跳采用 起跳帧计算速度
    if method == 'cmj' and jump_frame > 0:
        # 计算 6*fps_n - 1 帧的平均速度, 然后通过该平均速度反推 起跳帧的速度
        max_speed_2 = (np.mean(speed_arr[(jump_frame + 1):(jump_frame + 6*fps_n)])) + (6*fps_n / 2 - 0.5) * 9.8 / fps
    if jump_frame > 0:
        filename = 'keyframe_' + filename
        key_frame_path = os.path.join(file_path, filename)
        extract_frames(input_video = input_path, output_video = key_frame_path, start_frame = (jump_frame - fps_n * 20), end_frame = (jump_frame + fps_n * 20), max_speed=max_speed_2)
        print('视频已保存', key_frame_path)
    else:
        filename = 'nokeyframe_' + filename
        key_frame_path = os.path.join(file_path, filename)
        extract_frames(input_video = input_path, output_video = key_frame_path, start_frame = 0, end_frame = (frame_count_new-1), max_speed=max_speed)
        print('视频已保存', key_frame_path)
    cap.release()
    print(filename)
    if method == 'cmj':
        if max_speed_2 > 0:
            return max_speed_2, filename
        else:
            return max_speed, filename
    else:
        return max_speed, filename