# opencv-python
import cv2
# mediapipe人工智能工具包
import mediapipe as mp
import numpy as np
import array as arr

# 导入solution
mp_pose = mp.solutions.pose

# # 导入绘图函数
mp_drawing = mp.solutions.drawing_utils 

# 导入模型
pose = mp_pose.Pose(static_image_mode=False,        # 是静态图片还是连续视频帧
                    model_complexity=2,            # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
                    smooth_landmarks=True,         # 是否平滑关键点
                    min_detection_confidence=0.5,  # 置信度阈值
                    min_tracking_confidence=0.5)   # 追踪阈值

# 三维距离有误差，目前只计算二维距离
def get_landmark(results, i):
    return np.array([results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y])

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
        nose = get_landmark(results, 0)
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2
        # body_height 计算 分三段，肩关节 -> 髋关节， 髋关节 -> 膝关节， 膝关节 -> 踝关节
        body_height = np.linalg.norm(mid_shoulder - mid_hip) + np.linalg.norm(mid_hip - (left_knee + right_knee)/2) + np.linalg.norm((left_knee + right_knee)/2 - (left_ankle + right_ankle)/2)
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
        scaler = 3
        failure_str = 'No Person'
        
    # 记录该帧处理完毕的时间
    # end_time = time.time()
    # 计算每秒处理图像帧数FPS
    # FPS = 1/(end_time - start_time)
    # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    #img = cv2.putText(img, 'FPS  '+str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    return img, body_height, mid_hip[1], right_hip, left_hip, right_shoulder, left_shoulder, right_knee, left_knee, right_ankle, left_ankle


def speed_cal(ref_height, input_path, method = 'cmj', height_type = 'ankle2shoulder'):
    #filehead = input_path.split('/')[-1]
    #fileroute = input_path.split
    #output_path = "out-" + filehead
    
    # print('视频开始处理',input_path)
    
    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while(cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    # print('视频总帧数为',frame_count)
    if frame_count > 1000:
        # print('帧数过多，请剪辑视频')
        return -100
    
    # 初始化各个变量，返回髋关节以及身高数组
    hip_arr = arr.array('f')
    body_height_arr = arr.array('f')
    speed_arr = arr.array('f')
    mid_gap_arr = arr.array('f')
    right_hip_arr = list()
    left_hip_arr = list()
    right_shoulder_arr = list()
    left_shoulder_arr = list()
    right_knee_arr = list()
    left_knee_arr = list()
    right_ankle_arr = list()
    left_ankle_arr = list()
    body_height_last = 0
    mid_hip_last = 0
    max_speed = 0
    frame_count_new = 0
    scaler = 1
    # cv2.namedWindow('Crack Detection and Measurement Video Processing')
    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 进度条绑定视频总帧数
    while(cap.isOpened()):
        success, frame = cap.read()
        if not success:
            break
        # 处理帧
        # frame_path = './temp_frame.png'
        # cv2.imwrite(frame_path, frame)
        #try:
        
        # except:
        #     print('error')
        #     pass

        else:
            frame, body_height, mid_hip, right_hip, left_hip, right_shoulder, left_shoulder, right_knee, left_knee, right_ankle, left_ankle = process_frame(frame)
            hip_arr.append(-mid_hip)
            body_height_arr.append(body_height)
            right_hip_arr.append(right_hip)
            left_hip_arr.append(left_hip)
            right_shoulder_arr.append(right_shoulder)
            left_shoulder_arr.append(left_shoulder)
            right_knee_arr.append(right_knee)
            left_knee_arr.append(left_knee)
            right_ankle_arr.append(right_ankle)
            left_ankle_arr.append(left_ankle)
            # body_height = 0 则表示 人体识别不成功， 不计算速度
            if body_height > 0  and body_height_last > 0 and frame_count_new > 10:
                mid_gap = round(-(mid_hip - mid_hip_last),3)
                if method == 'cmj' and height_type == 'ankel2shoulder':
                    speed = round(-(mid_hip - mid_hip_last) * fps * (ref_height / body_height), 3)
                elif method == 'cmj' and height_type == 'bodyheight':
                    # 当输入为身高时，则取 body_height / 0.77 为 推断身高
                    speed = round(-(mid_hip - mid_hip_last) * fps * (ref_height / body_height * 0.77), 3)
                else:
                    print('不存在该方法')
                
            else:
                mid_gap = 0
                speed = 0
            speed_arr.append(speed)
            mid_gap_arr.append(mid_gap)
            if speed > max_speed:
                max_speed = speed
            frame_count_new += 1
            body_height_last = body_height
            mid_hip_last = mid_hip
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
    #cv2.destroyAllWindows()
    cap.release()
    return max_speed