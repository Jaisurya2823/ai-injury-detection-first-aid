# Model weights

Place the **project-trained injury YOLO weights** here as `best.pt`.

The repository intentionally does not bundle a medical model. Do not substitute a generic COCO object-detection model and present it as an injury detector.

Enable real inference with:

```text
FIRST_AID_MODEL_MODE=yolo
FIRST_AID_YOLO_WEIGHTS=models/best.pt
```
