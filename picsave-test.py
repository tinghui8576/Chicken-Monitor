import cv2, time, os
from datetime import date
#vid = ffmpeg.input('/dev/video0', t=20, r=5)
#vid = ffmpeg.output(vid, time.strftime('%M%s') + ".avi", bitrate="500k",)

record_time = 5

cam = cv2.VideoCapture(0)
    
while True:
    cwd = os.getcwd()
    folder = os.path.join(cwd,"Pictures")
    print(folder)
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    if int(time.strftime("%H")) >= 18:
        break
    ret, frame = cam.read()
    img_name = folder + "/" + time.strftime("%H-%M-%S") + ".jpg"
    print(ret, img_name)
    cv2.imwrite(img_name, frame)    
    
    time.sleep(record_time)

cam.release()
cv2.destroyAllWindows()
    #process.terminate()

#ffmpeg.run(vid)
#vid.terminate()
