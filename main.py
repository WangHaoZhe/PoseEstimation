import cv2 as cv
import time

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

inWidth = 200  # 画面宽度
inHeight = 200  # 画面高度
outWidth = 640  # 显示画面宽度
outHeight = 480  # 显示画面高度
thr = 0.05  # 阈值

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
# 程序中的左右为人体的左右,不是画面

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

# 识别参数
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 10)  # 帧率
cap.set(3, outWidth)
cap.set(4, outHeight)

# 动作坐标
act = [[(100, 100), (250, 250),  # 右上 右下
        (600, 100), (450, 250)],  # 左上 左下
       # 添加其他动作坐标
       ]

# 生成判定上下界
act_left = []
act_right = []
delta_x = 10  # 阈值
delta_y = 30
act_number = 0  # 初始化动作序号,每完成一个动作,序号加一

for acts in act:
    act_right.append([(acts[0][0] - delta_x, acts[0][1] - delta_y),
                      (acts[1][0] + delta_x, acts[1][1] - delta_y),  # 右侧(画面左侧)上界两点
                      (acts[0][0] - delta_x, acts[0][1] + delta_y),
                      (acts[1][0] + delta_x, acts[1][1] + delta_y)])  # 右侧(画面左侧)下界两点

    act_left.append([(acts[2][0] + delta_x, acts[2][1] - delta_y),
                     (acts[3][0] - delta_x, acts[3][1] - delta_y),  # 左侧(画面右侧)上界两点
                     (acts[2][0] + delta_x, acts[2][1] + delta_y),
                     (acts[3][0] - delta_x, acts[3][1] + delta_y)])  # 左侧(画面右侧)下界两点


# 该函数用于检测点是否在平行四边形内
# point为点坐标,格式为(x,y)
# act_side为"left"或"right",该参数用于确定列表
# act_number为动作编号(自0开始)
def detect_point(point, act_side, act_number):
    detect = 0
    if act_side == "left":
        act_triangle_w = act_left[act_number][0][0] - act_left[act_number][1][0]  # 动作三角形参数(详见图片)
        act_triangle_h = act_left[act_number][1][1] - act_left[act_number][0][1]
        point_triangle_w = act_left[act_number][0][0] - point[0]
        point_triangle_h = act_left[act_number][0][1] - point[1]
        boundary = point_triangle_w / act_triangle_w * act_triangle_h
        if boundary <= point_triangle_h <= boundary+2*delta_y and act_left[act_number][1][0] <= point[0] <= act_left[act_number][0][0]:
            detect = 1

    elif act_side == "right":
        act_triangle_w = act_right[act_number][1][0] - act_right[act_number][0][0]  # 动作三角形参数(详见图片)
        act_triangle_h = act_right[act_number][1][1] - act_right[act_number][0][1]
        point_triangle_w = point[0] - act_right[act_number][0][0]
        point_triangle_h = point[1] - act_right[act_number][0][1]
        boundary = point_triangle_w / act_triangle_w * act_triangle_h
        if boundary <= point_triangle_h <= boundary+2*delta_y and act_right[act_number][0][0] <= point[0] <= act_right[act_number][1][0]:
            detect = 1

    return detect


if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert (len(BODY_PARTS) == out.shape[1])

    points = []

    # 绘制界线
    cv.line(frame, act[act_number][0], act[act_number][1], (0, 255, 0), 3)
    cv.line(frame, act[act_number][2], act[act_number][3], (0, 255, 0), 3)

    cv.line(frame, act_left[act_number][0], act_left[act_number][1], (255, 0, 0), 3)
    cv.line(frame, act_left[act_number][2], act_left[act_number][3], (255, 0, 0), 3)

    cv.line(frame, act_right[act_number][0], act_right[act_number][1], (0, 0, 255), 3)
    cv.line(frame, act_right[act_number][2], act_right[act_number][3], (0, 0, 255), 3)

    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # 调用detect_point()检测点是否在阈值允许的范围内,若在则显示绿色方框并进入下一个动作
    # 此处示例程序关键点仅有LShoulder(points[5])与RShoulder(points[2])
    if act_number <= len(act) - 1:
        if points[2] and points[5]:
            detect_right = detect_point(points[2], "right", act_number)
            detect_left = detect_point(points[5], "left", act_number)
            if detect_left and detect_right:
                cv.rectangle(frame, (0, 0), (640, 480), (0, 255, 0), thickness=15)
                act_number += 1
            if act_number == len(act):
                print("Finish!")
                break

    cv.imshow('Video Tutorial', frame)




