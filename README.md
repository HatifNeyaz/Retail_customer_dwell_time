# 🛒 Retail Aisle Analytics: Computer Vision Dwell Time Tracker

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green.svg)

## 📌 Project Overview
Physical retail spaces lack the granular analytics of e-commerce websites. This project bridges that gap by transforming standard CCTV footage into actionable business intelligence. 

This end-to-end Computer Vision pipeline detects customers, assigns persistent tracking IDs, and calculates **Dwell Time** (how long a customer interacts with a specific shelf) using custom-defined spatial polygon zones. The backend tracking data is then exported to a live Streamlit dashboard for high-level KPI monitoring.

### 🎥 Demonstration
*(Watch the pipeline in action)*
<br>
<img src="videos/demo.gif" width="800" alt="Retail Tracking Demo">

---

## 🧠 System Architecture & Data Flow

The system is decoupled into two layers: the **Computer Vision Pipeline** (processing frames) and the **Analytics Dashboard** (processing data).

```mermaid
graph TD
    subgraph Computer Vision Pipeline [main.py]
        A[Raw CCTV Video .mp4] -->|OpenCV| B(YOLOv8 Object Detection)
        B -->|Bounding Boxes| C{ByteTrack Algorithm}
        C -->|Assigns Persistent IDs| D[Supervision Polygon Zone]
        D -->|Spatial Logic| E{Is Customer in Shelf Zone?}
        E -->|Yes| F[Increment Dwell Timer]
        E -->|No| G[Ignore]
    end

    subgraph Output Generation
        F --> H(Annotated .mp4 Output)
        F --> I[(dwell_times.csv)]
    end

    subgraph Data Visualization [app.py]
        I --> J[Streamlit Dashboard]
        J --> K[Plotly Interactive Charts]
        J --> L[KPI Metrics]
    end