from tkinter import *
from PIL import ImageTk, Image
from ultralytics import YOLO
import cv2

class YOLOApp:
    def __init__(self, root, title):
        self.root = root
        self.root.title(title)
        
        self.model = YOLO("yolov8n.pt")
        
        # สร้างปุ่ม Start
        self.start_button = Button(self.root, text="Start", command=self.start_prediction)
        self.start_button.pack(pady=20)
        
        # สร้างภาพเปล่าเพื่อแสดงผลลัพธ์
        self.panel = Label(self.root)
        self.panel.pack()

    def start_prediction(self):
        # ค้นหากล้องที่เปิดได้ทุกตัว
        num_cameras = 0
        while True:
            cap = cv2.VideoCapture(num_cameras)
            if not cap.isOpened():
                break
            
            ret, frame = cap.read()
            if ret:
                self.show_results(frame)
                break
            
            num_cameras += 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ทำนายภาพ
            results = self.model.predict(source=frame, show=True)

            # แสดงผลลัพธ์บน GUI
            self.show_results(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def show_results(self, frame):
        # แสดงผลลัพธ์บน GUI
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)

        # อัปเดตภาพใหม่บน GUI
        self.panel.configure(image=frame)
        self.panel.image = frame


# สร้างหน้าต่าง Tkinter
root = Tk()
app = YOLOApp(root, "YOLO Camera")
root.mainloop()
