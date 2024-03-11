import cv2
import mediapipe as mp
import pandas as pd

# Đọc ảnh từ tệp
img = cv2.imread('../HAR/pic/bird-dog-ready.png')

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "BIRDOGCHUANBI"
no_of_frames = 2

def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img

# Nhận diện pose
frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(frameRGB)

if results.pose_landmarks:
    # Ghi nhận thông số khung xương
    lm = make_landmark_timestep(results)
    lm_list.append(lm)
    # Vẽ khung xương lên ảnh
    img = draw_landmark_on_image(mpDraw, results, img)

cv2.imshow("image", img)
cv2.waitKey(0)
# Thay đổi kích thước hình ảnh
desired_size = (500, 500)  # Thay đổi kích thước mong muốn tại đây
img = cv2.resize(img, desired_size)
# Write vào file csv
df  = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cv2.destroyAllWindows()
