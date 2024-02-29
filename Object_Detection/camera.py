from tkinter import *
from tkinter import filedialog
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
        
        # สร้างปุ่ม Choose Image
        self.choose_image_button = Button(self.root, text="Choose Image", command=self.choose_image)
        self.choose_image_button.pack(pady=20)
        
        # สร้างภาพเปล่าเพื่อแสดงผลลัพธ์
        self.panel = Label(self.root)
        self.panel.pack()

    def start_prediction(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ทำนายภาพ
            results = self.model.predict(source="0", show=True)

            # แสดงผลลัพธ์บน GUI
            self.show_results(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def choose_image(self):
        # เลือกไฟล์ภาพจากเครื่องคอมพิวเตอร์
        file_path = filedialog.askopenfilename()

        if file_path:
            # อ่านภาพจากไฟล์
            image = cv2.imread(file_path)

            # ทำนายภาพ
            results = self.model.predict(source=image, show=True)

            # แสดงผลลัพธ์บน GUI
            self.show_results(image)

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
