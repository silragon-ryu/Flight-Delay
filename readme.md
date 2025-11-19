# ✈️ Dynamic ML Feature Pipeline Viewer  
### Real-Time Feature Engineering • Python + Pandas + PySide6

A lightweight desktop application that **visualizes how raw flight data transforms into ML-ready features**.  
It continuously generates synthetic delay records, applies classic feature-engineering steps, and updates the GUI in real time.

This project is ideal for learning, teaching, or demonstrating how operational ML pipelines work under the hood.

---

## ✨ Features at a Glance

### **1. Feature Hashing — High-Cardinality Categorical Inputs**
- Handles IATA airport codes such as `JFK`, `NRT`, `ICN`, `VIE`, etc.  
- Uses SHA-256 to map each code into a fixed bucket range (`0–99`).  
- Highlights **hash collisions** in yellow so you can see the real trade-offs.  

**Why?**  
Efficient dimensionality reduction for large categorical spaces.

---

### **2. Binary Encoding — Low-Cardinality Categorical Inputs**
- Converts small text categories into numeric values (`Domestic → 0`, `International → 1`).  
- Updates instantly as the data stream changes.

**Why?**  
Simple, fast, and widely used in traditional ML pipelines.

---

### **3. Delay Bucketing — Turning Regression Into Classification**
Transforms continuous delay minutes into color-coded classes:

- 🟢 **Class 0** — On-Time / Early (≤ 10 min)  
- 🟡 **Class 1** — Medium Delay (> 10 to ≤ 45 min)  
- 🔴 **Class 2** — Significant Delay (> 45 min)

**Why?**  
Useful when a probabilistic delay class is more actionable than precise regression values.

---

## 🛠️ Installation

Requires **Python 3.8+**.

```bash
pip install pandas PySide6
