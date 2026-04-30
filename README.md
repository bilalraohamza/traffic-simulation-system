# Distributed Real-Time Traffic Simulation and Command System

**Course:** Advanced Programming Design  
**University:** Wuhan Institute of Technology  
**Student:** [Your Name]  

## Project Overview
A distributed smart city traffic monitoring system consisting of a Python-based 
AI Edge Node and a Java-based Command Center communicating over TCP/IP sockets.

## System Architecture
- **Python Edge Node:** YOLOv11n ONNX inference + ByteTrack vehicle tracking + TCP broadcasting
- **Java Command Center:** Multithreaded ServerSocket + SQLite JDBC logging + JavaFX dashboard

## AI Model Performance
- Model: YOLOv11n fine-tuned on combined vehicle dataset (11,218 images)
- mAP@50: 91.4%
- mAP@50-95: 70.4%
- Inference speed: 3.2ms per frame
- Classes: car, truck, bus, motorbike, van, threewheel, ambulance, bicycle

## Tech Stack
| Component | Technology |
|---|---|
| AI Training | Kaggle T4 GPU, Ultralytics YOLOv11 |
| AI Inference | ONNX Runtime (Python + Java) |
| Vehicle Tracking | ByteTrack |
| Network | TCP/IP Sockets |
| Database | SQLite via JDBC |
| GUI | JavaFX + JFreeChart |
| Build Tool | Maven |

## Setup Instructions
### Python Edge Node
```bash
pip install -r requirements.txt
python edge_node.py
```

### Java Command Center
```bash
cd java_command_center
mvn clean javafx:run
```

## Note on Model File
The trained ONNX model (best.onnx) is not included in this repository due to 
file size. It can be reproduced by running the Kaggle training notebook included 
in the /kaggle_notebook/ folder.