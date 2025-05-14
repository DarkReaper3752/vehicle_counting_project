import cv2
import numpy as np
from ultralytics import YOLO
from math import sqrt

class VehicleTracker:
    def __init__(self):
        self.next_id = 1
        self.vehicles = {}  # {id: {'center': (x,y), 'type': type, 'not_seen_count': 0}}
        self.counted_ids = set()
        self.MAX_DISTANCE = 50
        self.MAX_UNSEEN_FRAMES = 10

    def calculate_distance(self, center1, center2):
        return sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def update(self, detections):
        current_centers = []
        matched_ids = set()

        # Mevcut araçların görülmeme sayısını artır
        for vehicle in self.vehicles.values():
            vehicle['not_seen_count'] += 1

        # Her tespit için en yakın aracı bul
        for det in detections:
            x1, y1, x2, y2, conf, cls, vehicle_type = det
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            min_distance = float('inf')
            matching_id = None

            # En yakın mevcut aracı bul
            for vid, vehicle in self.vehicles.items():
                if vid not in matched_ids:
                    distance = self.calculate_distance(center, vehicle['center'])
                    if distance < min_distance and distance < self.MAX_DISTANCE:
                        min_distance = distance
                        matching_id = vid

            if matching_id is not None:
                # Mevcut aracı güncelle
                self.vehicles[matching_id].update({
                    'center': center,
                    'type': vehicle_type,
                    'bbox': (x1, y1, x2, y2),
                    'not_seen_count': 0
                })
                matched_ids.add(matching_id)
                current_centers.append((matching_id, vehicle_type, (x1, y1, x2, y2)))
            else:
                # Yeni araç ekle
                self.vehicles[self.next_id] = {
                    'center': center,
                    'type': vehicle_type,
                    'bbox': (x1, y1, x2, y2),
                    'not_seen_count': 0
                }
                current_centers.append((self.next_id, vehicle_type, (x1, y1, x2, y2)))
                self.next_id += 1

        # Uzun süre görünmeyen araçları sil
        self.vehicles = {
            vid: vehicle for vid, vehicle in self.vehicles.items()
            if vehicle['not_seen_count'] < self.MAX_UNSEEN_FRAMES
        }

        return current_centers

def main():
    model = YOLO('models/yolov8n.pt')

    vehicle_classes = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker = VehicleTracker()
    vehicle_counts = {v: 0 for v in vehicle_classes.values()}
    counted_vehicles = set()  # Sayılan araçları takip etmek için

    print("Program başlatıldı! Çıkmak için 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO tespitleri
        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls in vehicle_classes and conf > 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_type = vehicle_classes[cls]
                detections.append([x1, y1, x2, y2, conf, cls, vehicle_type])

        # Araçları takip et
        tracked_vehicles = tracker.update(detections)

        # Sayım çizgisini çiz
        line_y = frame.shape[0] // 2
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

        # Araçları görselleştir ve say
        for vehicle_id, vehicle_type, bbox in tracked_vehicles:
            x1, y1, x2, y2 = bbox
            
            # Dikdörtgen çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Araç bilgisi için etiket
            label = f"ID:{vehicle_id} {vehicle_type}"
            
            # Etiket konumu ve boyutu
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_width = label_size[0]
            label_height = label_size[1]
            
            # Etiket için arka plan
            label_x = x1
            label_y = y1 - 5
            cv2.rectangle(frame, 
                        (label_x, label_y - label_height - 5),
                        (label_x + label_width + 5, label_y + 5),
                        (0, 255, 0), -1)
            
            # Etiketi yaz
            cv2.putText(frame, label,
                       (label_x + 3, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Sayım
            center_y = (y1 + y2) // 2
            if center_y > line_y and vehicle_id not in counted_vehicles:
                vehicle_counts[vehicle_type] += 1
                counted_vehicles.add(vehicle_id)
                print(f"Yeni araç sayıldı: {vehicle_type} (ID: {vehicle_id})")

        # Sayım bilgilerini göster
        info_x = 10
        info_y = 30
        padding = 25

        # Sayım başlığı
        cv2.putText(frame, "Vehicle Counts:",
                   (info_x, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Her araç tipi için sayım
        for i, (vehicle_type, count) in enumerate(vehicle_counts.items(), 1):
            text = f"{vehicle_type}: {count}"
            cv2.putText(frame, text,
                       (info_x, info_y + padding * i),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Vehicle Detection and Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()