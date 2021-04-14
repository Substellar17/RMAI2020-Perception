##v4版本可以识别多车并对应,并行运算

# 场地 493.5 685.0 (245, 423)
# 灯条 12.5(5.5) 13.5

import sys
sys.path.append("./angle_classify")
sys.path.append("./armor_classify")
sys.path.append("./car_classify")

import cv2
import time
import torch
import numpy as np
from classification import *
from position_predict import *
from classification_car import *
from yolo_detect_v2 import output
from utils.utils_mulanchor import *
from models_nolambda_focallossw import *
from classification_angle_camera import *
from multiprocessing.dummy import Pool as ThreadPool
from armor_detect import read_morphology_temp,find_contours
from armor_detect_withlightbox import read_morphology_withlightbox,find_contours_withlightbox

camera = 'left'

mouse_down_pos = []

def on_mouse(event, x, y, flags, param):
    # if event == cv2.EVENT_MOUSEMOVE:
    #     print("mouse move:", x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        print("mouse down:", x, y)
        mouse_down_pos.append((x, y))

def camera_calibration(cap, camera='left'):
    np.set_printoptions(suppress=True)
    # 四个参考点的世界坐标(mm) 右手系
    object_3d_points = np.array(([0, 0, 0],
                                 [3890, 4705, 0],
                                 [2495, 2360, 0],
                                 [2450, 4230, 0]), dtype=np.double)
    ref_point_num = len(object_3d_points)

    ret, frame = cap.read()
    cv2.imshow('camera', frame)
    cv2.setMouseCallback('camera', on_mouse)

    while cap.isOpened() and len(mouse_down_pos) < ref_point_num:
        if not use_video_file:
            ret, frame = cap.read()
            cv2.imshow('camera', frame)
            cv2.waitKey(5)
        else:
            cv2.waitKey(0)

    cv2.destroyWindow('camera')

    # 四个参考点的像素坐标
    object_2d_point = np.asarray(mouse_down_pos, dtype=np.double)
    print("object_2d_point:\n", object_2d_point)

    # TODO 相机标定
    if camera == 'left':
         camera_matrix = np.array([[1.1640e+3, 0, 929.9118],
                                   [0, 1.1627e+3, 559.4615],
                                   [0,0,1]], dtype = "double")
         dist_coeffs = np.transpose([0.0597, -0.2569, 3.7921e-4, 1.6921e-5, 0.1209])

    
    if camera == 'right':
        camera_matrix = np.array([[653.528968471312,0,316.090142900466],
                                  [0,616.850241871879,242.354349211058],
                                  [0,0,1]], dtype="double")
        dist_coeffs = np.transpose([-0.203713353732576, 0.178375149377498, -0.000880727909602325, 
                                    -0.00023370151705564, -0.0916209128198407])

    # success, 旋转向量, 平移向量
    success, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coeffs)
    # 旋转向量变旋转矩阵
    Rcw = cv2.Rodrigues(rvec)[0]

    # 世界坐标到相机坐标旋转矩阵，世界坐标到相机坐标向量（相机坐标系下世界原点的位置），相机内参，相机畸变参数
    return np.array(Rcw), np.array(tvec), camera_matrix, dist_coeffs


def point_sort(box):
    x = [box[0][0],box[1][0],box[2][0],box[3][0]]
    index = np.argsort(x)
    left = [box[index[0]],box[index[1]]]
    right = [box[index[2]],box[index[3]]]
    if left[0][1]< left[1][1]:
        left_up = left[0]
        left_down = left[1]
    else:
        left_up = left[1]
        left_down = left[0]
    if right[0][1]< right[1][1]:
        right_up = right[0]
        right_down = right[1]
    else:
        right_up = right[1]
        right_down = right[0]
    return left_up,left_down,right_up,right_down


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim[1], input_dim[0]))  # resize: w h
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_


def camera_in_armor_coord(left_up,left_down,right_up,right_down):
    #   原理是：PNP算法
    #   找到四个对应点，根据摄像头参数求解实际世界坐标

    image_points = np.array([
        (left_up[0], left_up[1]),
        (right_up[0], right_up[1]),
        (right_down[0], right_down[1]),
        (left_down[0], left_down[1]),
    ], dtype="double")
    # TODO measure armor size
    high = 60 #mm
    width = 137 #mm
    model_points = np.array([
        (-width/2, -high/2, 0),
        (width/2, -high/2, 0),
        (width/2, high/2, 0),
        (-width/2, high/2, 0),
    ])

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points,
                                                                camera_matrix, dist_coeffs,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)

    Rca = cv2.Rodrigues(rotation_vector)[0]
    # 单位mm
    distance = np.sqrt(translation_vector[0]**2+translation_vector[1]**2+translation_vector[2]**2)
    # distance单位化成米
    return Rca, translation_vector, distance/1000


def armor_6(fig):
    array = fig
    fig = cv2.resize(array,(48, 48))
    fig = torch.Tensor(fig)
    fig = fig.permute((2,0,1))
    img = torch.unsqueeze(fig, 0)
    # outputs = net_model(img.cuda())
    outputs = net_model(img)
    _, predicted = torch.max(outputs.data, 1)

    return int(predicted)


def car_6(fig):    
    array = fig
    fig = cv2.resize(array,(56,56))
    fig = torch.Tensor(fig)
    fig = fig.permute((2,0,1))
    img = torch.unsqueeze(fig, 0)
    outputs = net_model_car(img)
    _, predicted = torch.max(outputs.data, 1)

    return int(predicted)


def world_angle_6(fig, pos, camera = 'left'):
    pos_array = pos
    pos_x = pos_array[0]
    pos_y = pos_array[1]
    pos_x = float(pos_x)
    pos_y = float(pos_y)
    pos_array = (pos_x, pos_y)
    pos_array = np.array(pos_array, dtype='float').reshape(1,2)
    pos_array = torch.tensor(pos_array)

    array = fig
    fig = cv2.resize(array, (56, 56))
    fig = torch.Tensor(fig)
    fig = fig.permute(2, 0, 1)
    img = torch.unsqueeze(fig, 0)
    # outputs = net_model_angle(img.cuda(), pos_array.cuda())
    outputs = net_model_angle(img, pos_array)
    _, predicted = torch.max(outputs.data, 1)
    predicted = int(predicted)
    
    # 坐标转换
    pi = math.pi
    alpha = 0
    di = pi / 8
    theta = di * (2 * predicted + 1)
    try:
        if (theta >= pi / 2 + math.atan(pose_x / pose_y) and theta < pi):
            alpha = theta - pi / 2 - math.atan(pose_x / pose_y)
        elif(theta >= pi * 2 - math.atan(pose_y / pose_x) and theta < pi * 2):
            alpha = theta - pi * 3 + math.atan(pose_y / pose_x)
        else:
            alpha = theta - pi + math.atan(pose_y / pose_x)
    except:
        pass

    return alpha, predicted


#-----------------------main----------------------------------#
save_image_log = False

use_video_file = False

width = 1920
height = 1080

if use_video_file:
    cap = cv2.VideoCapture("video_footage/1cars.avi")
else:
    cap = cv2.VideoCapture(1) # video capture on camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)    #设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  #设置长度

if (cap.isOpened() == False):
    print("Error opening video stream or file")

frame_id = 0
Rcw, translation_vector_cam, camera_matrix, dist_coeffs = camera_calibration(cap, camera)


#region load models
print("Loading networks.....")
#-----------------------yolo----------------------------------#
cfgfile = "cfg/yolov3_camera_raw_3_pre_resprune_sp0.001_p0.01_sp0.001_p0.01_sp0.001_p0.01.cfg"
weightsfile = "cfg/yolov3_camera_raw_3_pre_resprune_sp0.001_p0.01_sp0.001_p0.01_sp0.001_p0.01.weights"
names =  "cfg/camera_raw_0817_3.names"
classes = load_classes(names)
num_classes = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA = torch.cuda.is_available()
inp_dim = [416,416]
bbox_attrs = 5 + num_classes

# [tymon] modle: YOLO model 
model = Darknet(cfgfile, inp_dim).to(device)
model.load_darknet_weights(weightsfile)

#-----------------------distance fit--------------------------#
# [tymon] mlp_model: a torch model to calculate distance
mlp_model = load_mlp_model(camera)
mlp_model.eval() # [tymon] eval mode, not to change weights in model
print()
print("yolo loaded")

#-----------------------class model---------------------------#
# [tymon] armor classification
net_model = classification_modelload()
# [tymon] car classification
net_model_car = car_classification_modelload()
#-----------------------anger model---------------------------#
net_model_angle = classification_angle_camer_modelload(camera)

if CUDA:
    model.cuda()
    mlp_model.cuda()

# [tymon] test input is an image in YOLO paper. just to test is it running fine
model(get_test_input(inp_dim, CUDA)).to(device)
model.eval().to(device)

print("Networks successfully loaded")
#endregion

# 4：状态数，包括（x，y，dx，dy）坐标及速度；2：观测量，能看到的是坐标值
kalman = cv2.KalmanFilter(4, 2)
# 系统测量矩阵
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) 
# 状态转移矩阵
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) 
# TODO 系统过程噪声协方差
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03 


time_start = time.time()    
while (cap.isOpened()):
    try:
        ret, frame = cap.read()
        size_img = frame.shape[:2]
    except:
        print('time cost:', time_stop-time_start)
        break

    frame_show = frame.copy()
    frame_id += 1

    if ret == True:
        t_start = time.time()

        # [tymon] run YOLO and get the output. its structure:
        # # 输出为一个list, list中包含每辆车的字典, 字典中有两个key值, 'car_box'与'armor_box', 
        # # car_box为一维数组, 4个值表示车框的位置
        # # armor_box为二维数组, 每行4个值表示一个装甲框的位置
        output_dict = output(frame, CUDA, model, device, num_classes)

        # [Zixing] 遍历每辆车
        for i in range(len(output_dict)):

            # 车辆阵营
            output_dict[i]['car_class'] = -1 # -1：未知阵营 
            # 车辆位姿角度
            output_dict[i]['car_angle'] = []
            # 车辆灯条框
            output_dict[i]['light_box'] = np.zeros((len(output_dict[i]['armor_box'])+1, 4, 2))
            # 车辆位置坐标
            output_dict[i]['position'] = np.zeros((len(output_dict[i]['armor_box'])+1, 2))

            # 灯条计数
            light_cnt = 0

            #-------------基于灯条的位置解算---------------------------------#

            # 识别到该车装甲板
            if len(output_dict[i]['armor_box']) != 0:

                # 裁剪灯条
                y0 = int(round(output_dict[i]['armor_box'][0][1])) - 5
                h = int(round(output_dict[i]['armor_box'][0][3])) - int(round(output_dict[i]['armor_box'][0][1])) + 10
                x0 = int(round(output_dict[i]['armor_box'][0][0])) - 5
                w = int(round(output_dict[i]['armor_box'][0][2])) - int(round(output_dict[i]['armor_box'][0][0])) + 10

                # [tymon] 剪裁出第一个装甲的图, 用它去给车分类
                armor = frame[y0:y0+h, x0:x0+w]
                if np.shape(armor)[0] !=0 and np.shape(armor)[1] !=0:
                    # [tymon] call armor classification model
                    car_class = armor_6(armor)
                    output_dict[i]['car_class'] = car_class

                # 遍历装甲板
                for j in range(len(output_dict[i]['armor_box'])):
                    index = j

                    # 裁剪第 j 号装甲板
                    y0 = int(round(output_dict[i]['armor_box'][j][1])) - 5
                    h = int(round(output_dict[i]['armor_box'][j][3])) - int(round(output_dict[i]['armor_box'][j][1])) + 10
                    x0 = int(round(output_dict[i]['armor_box'][j][0])) - 5
                    w = int(round(output_dict[i]['armor_box'][j][2])) - int(round(output_dict[i]['armor_box'][j][0])) + 10
                    armor = frame[y0:y0+h, x0:x0+w]

                    # armor 有效
                    if np.shape(armor)[0] != 0 and np.shape(armor)[1] != 0:

                        # find contours with light box
                        dst_dilate, robot_resize, factor = read_morphology_withlightbox(armor)
                        _, box = find_contours_withlightbox(dst_dilate, robot_resize, index)

                        # 找到两根灯条之后才能开始算！！
                        if len(box) != 1:
                            light_cnt += 1
                            for l in range(len(box)):
                                box[l][0] = box[l][0]/factor + x0
                                box[l][1] = box[l][1]/factor + y0
                            box = np.int0(box)
                            frame_show = cv2.drawContours(frame_show,[box],0,(0,0,255),2)
                            left_up,left_down,right_up,right_down = point_sort(box)
                            print('%d.jpg'%(frame_id))

                            #-------from camera coordinate system to world coordinate system-----#
                            _, translation_vector, distance = camera_in_armor_coord(left_up,left_down,right_up,right_down )
                            position_world = np.dot(Rcw.T,(translation_vector-translation_vector_cam))
            
                            
                            # TODO： 修正误差

                            x = position_world[0] / 1000
                            y = position_world[1] / 1000

                            output_dict[i]['light_box'][j] = box
                            output_dict[i]['position'][j] = (x, y)

                            print("X world: ", position_world[2])
                            print("Y world: ", position_world[0])
                            print("position world: ", position_world)
                            # print("x = ", x)
                            # print("y = ", y)

                        # TODO
                        if np.sqrt((x0 + w/2 - 257)**2 + (y0 + h/2 - 220)**2) > 50:
                            cv2.rectangle(frame_show, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 1)
            
            # 检测到车但是没检测到灯条
            #-------------基于车辆的位置解算---------------------------------#
            if light_cnt == 0:
                y0 = int(round(output_dict[i]['car_box'][1])) - 5
                h = int(round(output_dict[i]['car_box'][3])) - int(round(output_dict[i]['car_box'][1])) + 10
                x0 = int(round(output_dict[i]['car_box'][0])) - 5
                w = int(round(output_dict[i]['car_box'][2])) - int(round(output_dict[i]['car_box'][0])) + 10
                robot = frame[y0:y0+h,x0:x0+w]

                if np.shape(robot)[0] !=0 and np.shape(robot)[1] !=0:
                    car_class = car_6(robot)
                    output_dict[i]['car_class'] = car_class

                if np.shape(robot)[0] !=0 and np.shape(robot)[1] !=0:
                    dst_dilate, robot_resize, factor = read_morphology_temp(robot)
                    _, box = find_contours(dst_dilate, robot_resize, 0)
                    if len(box) != 1:
                        for l in range(len(box)):
                            box[l][0] = box[l][0]/factor + x0
                            box[l][1] = box[l][1]/factor + y0
                        box = np.int0(box)
                        left_up,left_down,right_up,right_down = point_sort(box)
                        print('%d.jpg'%(frame_id))
                        
                        #-------from camera coordinate system to world coordinate system-----#
                        _, translation_vector, distance = camera_in_armor_coord(left_up, left_down, right_up, right_down)
                        position_world = np.dot(Rcw.T, (translation_vector-translation_vector_cam))

                        # TODO: 修正误差

                        x = position_world[0] / 1000
                        y = position_world[1] / 1000

                        output_dict[i]['position'][-1] = (x,y)

            #-------------MLP 位置预测---------------------------------#
            # if len(output_dict[i]['car_box']) != 0:
            mlp_x, mlp_y = position_prediction(mlp_model, output_dict[i]['car_box'])
            output_dict[i]['position_mlp'] = (mlp_x, mlp_y)

            # fusion
            # [tymon] 因为我们还没训练MLP模型，就先不用它了，只用解算的坐标
            # position_f = position_fusion(output_dict[i])
            # 取出第一个成功的解算坐标
            # TODO 多个灯条结合 and 灯条到车辆中心换算

            position_f = (0, 0)
            for j in range(len(output_dict[i]['position'])):
                if np.max(output_dict[i]['position'][j]) != 0:
                    position_f = output_dict[i]['position'][j]
                    break
            output_dict[i]['position_fusion'] = position_f

            #-------------angle predicted------------------------------#
            # if len(output_dict[i]['car_box']) != 0 :
            y0 = int(round(output_dict[i]['car_box'][1])) - 5
            h = int(round(output_dict[i]['car_box'][3])) - int(round(output_dict[i]['car_box'][1])) + 10
            x0 = int(round(output_dict[i]['car_box'][0])) - 5
            w = int(round(output_dict[i]['car_box'][2])) - int(round(output_dict[i]['car_box'][0])) + 10
            robot = frame[y0:y0+h, x0:x0+w]
            if np.shape(robot)[0] !=0 and np.shape(robot)[1] !=0:
                pos = output_dict[i]['position_fusion']
                angle, _ = world_angle_6(robot, pos, camera)
                output_dict[i]['car_angle'] = angle

            # ------------卡尔曼滤波 平滑位置信息------------------------#
            # TODO 因为车辆分类还没做好，目前默认只有一辆车！！！
            # 如果检测到了多辆车，为了避免混淆，暂时跳过滤波
            # 后面每辆车应该有各自的滤波器
            if len(output_dict) == 1:
                # 如果距离没算出来，仍然使用上次的位置
                if np.max(output_dict[i]['position_fusion']) != 0:
                    pos = output_dict[i]['position_fusion']
                    pos = np.asarray(pos, np.float32).reshape((2, 1))
                    kalman.correct(pos) # 用当前测量来校正卡尔曼滤波器
                # 计算卡尔曼预测值，作为当前位置。后两个是速度，忽略
                output_dict[i]['final_position'] = kalman.predict()[:2]
            # 多辆车，跳过滤波
            else:
                output_dict[i]['final_position'] = output_dict[i]['position_fusion']

            # ------------show results on screen-----------------------#
            y0 = int(round(output_dict[i]['car_box'][1])) - 5
            h = int(round(output_dict[i]['car_box'][3])) - int(round(output_dict[i]['car_box'][1])) + 10
            x0 = int(round(output_dict[i]['car_box'][0])) - 5
            w = int(round(output_dict[i]['car_box'][2])) - int(round(output_dict[i]['car_box'][0])) + 10
            cv2.rectangle(frame_show, (x0, y0), (x0 + w, y0 + h), (0, 0, 255), 1)

            car_class_dict = {0:'blue-1',1:'blue-2',2:'red-1',3:'red-2',4:'grey-1',5:'grey-1'}
            car_class = car_class_dict.get(output_dict[i]['car_class'], 'unknown')
            angle = output_dict[i]['car_angle']
            angle = str(round(angle, 3)) if angle is float else "unknown angle"
            text = 'ID: ' + car_class + ' Pose: ' + angle
            cv2.putText(frame_show, text, (x0,y0-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)

            pos = output_dict[i]['final_position']
            frame_show = cv2.putText(frame_show, 'x=%.2f,y=%.2f'%(pos[0],pos[1]), (x0,y0), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
            # frame_show = cv2.putText(frame_show, 'x=%.2f,y=%.2f'%(position_world[2],position_world[0]), (x0,y0), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
            # print(output_dict)

        t_stop = time.time()
        time_stop = time.time()
        print('t cost:', t_stop - t_start)

        #-----------test log------------------------------#
        img_name = str(frame_id) + '.jpg'
        img_path = './fig4/' + img_name

        cv2.imshow('img', frame_show)
        if save_image_log:
            cv2.imwrite(img_path, frame_show)
        cv2.moveWindow('img', 0, 0)    
        cv2.waitKey(5)

    else:
        print('time cost:', time_stop-time_start)
        break
