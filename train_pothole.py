from ultralytics import YOLO

def main():
    # 1. Load the pre-trained 'base' model
    model = YOLO("yolov8n.pt") 

    # 2. Start training
    # epochs: How many times to look at the data (10-20 is good for a test)
    # imgsz: Image size (640 is standard)
    model.train(data="pothole.yaml", epochs=20, imgsz=640)

if __name__ == "__main__":
    main()