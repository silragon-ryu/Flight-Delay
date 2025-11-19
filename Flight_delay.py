import sys
import hashlib
import random
import pandas as pd
from typing import List, Tuple, Dict, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTableWidget, QTableWidgetItem, QLabel, QGroupBox, QHeaderView,
    QAbstractItemView, QStatusBar, QTableView
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QFont, QColor

# ====================================================================
# PART A: Core ML Transformation Logic 
# ====================================================================

# Expanded airport data including country and domestic/international classification
AIRPORT_DATA = {
    # Domestic US (for demonstration)
    "JFK": {"type": "International", "city": "New York", "country": "USA"},
    "LAX": {"type": "International", "city": "Los Angeles", "country": "USA"},
    "SFO": {"type": "International", "city": "San Francisco", "country": "USA"},
    "ATL": {"type": "Domestic", "city": "Atlanta", "country": "USA"},
    "ORD": {"type": "Domestic", "city": "Chicago", "country": "USA"},
    "DFW": {"type": "Domestic", "city": "Dallas", "country": "USA"},
    # International (Expanded)
    "LHR": {"type": "International", "city": "London", "country": "UK"},
    "CDG": {"type": "International", "city": "Paris", "country": "France"},
    "IST": {"type": "International", "city": "Istanbul", "country": "Turkey"},
    "DXB": {"type": "International", "city": "Dubai", "country": "UAE"},
    "MUC": {"type": "International", "city": "Munich", "country": "Germany"},
    # Asia Expansion (Japan, China, Korea, etc.)
    "NRT": {"type": "International", "city": "Tokyo", "country": "Japan"},
    "HND": {"type": "International", "city": "Tokyo", "country": "Japan"},
    "PEK": {"type": "International", "city": "Beijing", "country": "China"},
    "PVG": {"type": "International", "city": "Shanghai", "country": "China"},
    "ICN": {"type": "International", "city": "Seoul", "country": "South Korea"},
    "KUL": {"type": "International", "city": "Kuala Lumpur", "country": "Malaysia"},
    "SIN": {"type": "International", "city": "Singapore", "country": "Singapore"},
    "BKK": {"type": "International", "city": "Bangkok", "country": "Thailand"},
    "SYD": {"type": "International", "city": "Sydney", "country": "Australia"},
}

IATA_CODES = list(AIRPORT_DATA.keys())
HASH_BUCKETS = 100

# Buckets for the Reframing (Output Label)
# 0: On-Time/Early (Delay <= 10 min)
# 1: Medium Delay (10 min < Delay <= 45 min)
# 2: Significant Delay (Delay > 45 min)
BUCKET_RANGES = [
    (10, "On-Time/Early", QColor("#34D399")),  # Tailwind Green-400
    (45, "Medium Delay", QColor("#FBBF24")),   # Tailwind Yellow-400
    (float('inf'), "Significant Delay", QColor("#F87171")) # Tailwind Red-400
]

def hashed_feature_mapping(iata_code: str, num_buckets: int) -> int:
    """Maps IATA code to a fixed bucket index using Feature Hashing (Input Transformation)."""
    if not iata_code:
        return 0
    encoded_string = iata_code.encode('utf-8')
    # Use SHA-256 for a stable hash
    hash_value = int(hashlib.sha256(encoded_string).hexdigest(), 16)
    return hash_value % num_buckets

def bucketize_delay(arrival_delay_minutes: float) -> Tuple[int, str, QColor]:
    """Converts continuous delay into a discrete classification label (Output Reframing)."""
    for index, (upper_bound, label, color) in enumerate(BUCKET_RANGES):
        if arrival_delay_minutes <= upper_bound:
            return index, label, color
    
    # Fallback
    return len(BUCKET_RANGES) - 1, BUCKET_RANGES[-1][1], BUCKET_RANGES[-1][2]

def generate_delay() -> float:
    """Generates a random, realistic, and dynamic delay (skewed towards 0/+ve)."""
    # A mix of uniform and exponential to simulate more smaller delays
    return round(random.uniform(-15, 5) + random.expovariate(0.05), 2)


# ====================================================================
# PART B: PySide6 GUI Application
# ====================================================================

class FlightDelayApp(QMainWindow):
    """Main window for the Dynamic ML Data Transformation GUI."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic Feature Engineering Viewer (PySide6)")
        self.setGeometry(100, 100, 1300, 650) # Increased width for new column
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_table_data)
        
        self.apply_styles()
        self.setup_ui()
        
        # Initial run and start the animation timer
        self.update_table_data()
        self.timer.start(3000) # Update every 3000 milliseconds (3 seconds)

    def apply_styles(self):
        """Applies a professional dark-themed QSS style sheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D3748; /* Dark Gray */
            }
            QWidget#CentralWidget {
                padding: 15px;
            }
            QGroupBox {
                background-color: #4A5568; 
                border: 2px solid #63B3ED; /* Light blue border */
                border-radius: 8px;
                margin-top: 10px;
                font-weight: bold;
                color: #E2E8F0; 
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
            }
            QTableWidget {
                background-color: #2D3748;
                border: 1px solid #718096;
                border-radius: 6px;
                gridline-color: #4A5568;
                color: #E2E8F0;
                font-size: 14px;
            }
            QHeaderView::section {
                background-color: #4A5568;
                color: #9DECF9; /* Cyan for headers */
                padding: 8px;
                border: 1px solid #718096;
                font-weight: bold;
                font-size: 14px;
            }
            QStatusBar {
                background-color: #1A202C; /* Very dark background */
                color: #A0AEC0;
                font-size: 12px;
                border-top: 1px solid #718096;
            }
        """)

    def setup_ui(self):
        """Sets up the main layout, table, and status bar."""
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # --- Title ---
        title_label = QLabel("Dynamic ML Feature Engineering Viewer")
        title_font = QFont("Inter", 24, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #63B3ED; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        # --- Reframing Legend Group (To show the categories) ---
        reframing_group = QGroupBox("Output Label Reframing (Delay $\rightarrow$ Class)")
        reframing_layout = QHBoxLayout(reframing_group)
        reframing_group.setMaximumHeight(80)
        
        for index, (upper, label, color) in enumerate(BUCKET_RANGES):
            delay_range = ""
            if index == 0:
                delay_range = "($\le$ 10 min)"
            else:
                prev_upper = BUCKET_RANGES[index-1][0]
                delay_range = f"({prev_upper} < min $\le$ {upper} min)" if upper != float('inf') else f"($> {prev_upper}$ min)"
            
            reframing_label = QLabel(f"Class {index}: {label} {delay_range}")
            reframing_label.setStyleSheet(f"color: {color.name()}; font-weight: bold; padding: 5px;")
            reframing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            reframing_layout.addWidget(reframing_label)
            
        main_layout.addWidget(reframing_group)

        # --- Results Table ---
        results_group = QGroupBox("Dynamic Transformation Results")
        results_layout = QVBoxLayout(results_group)
        
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setColumnCount(5) # Increased column count to 5
        self.table.setHorizontalHeaderLabels([
            "IATA Code (Raw Feature)", 
            "Type (Domestic/Intl.) [Binary]", # New Column
            "Actual Delay (Continuous)", 
            f"Hashed Index (0 to {HASH_BUCKETS-1})", 
            "Delay Class (Target Label)"
        ])
        self.table.setRowCount(len(IATA_CODES))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        results_layout.addWidget(self.table)
        
        main_layout.addWidget(results_group)
        
        # --- Status Bar ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(f"System Initialized. Hash Buckets (N) set to {HASH_BUCKETS}. Auto-updating every 3 seconds...")
        

    def update_table_data(self):
        """
        Executes the data transformation using a Pandas DataFrame and 
        updates the table.
        """
        
        # 1. Generate Raw Data into a list of dictionaries
        raw_data = []
        for code, details in AIRPORT_DATA.items():
            raw_data.append({
                'IATA_Code': code,
                'Actual_Delay': generate_delay(),
                'Airport_Type': details['type'] # New raw feature
            })
        
        # Convert to Pandas DataFrame
        df = pd.DataFrame(raw_data)
        
        # 2. Apply Vectorized Transformations using Pandas
        
        # A. Feature Hashing (Input Transformation: High Cardinality IATA)
        df['Hashed_Index'] = df['IATA_Code'].apply(
            lambda x: hashed_feature_mapping(x, HASH_BUCKETS)
        )
        
        # B. Binary Encoding (Input Transformation: Low Cardinality Type)
        # 0 for Domestic, 1 for International
        df['Type_Binary'] = df['Airport_Type'].apply(
            lambda x: 1 if x == 'International' else 0
        )
        
        # C. Reframing/Bucketing (Output Transformation)
        df[['Class_Index', 'Class_Label', 'Class_Color']] = df['Actual_Delay'].apply(
            lambda x: pd.Series(bucketize_delay(x))
        )
        
        # 3. Populate Table and Check Collisions
        collision_count = 0
        hash_map: Dict[int, List[str]] = {}
        
        # Iterate over the DataFrame to update the QTableWidget
        for row, data in df.iterrows():
            iata_code = data['IATA_Code']
            airport_type = data['Airport_Type']
            type_binary = data['Type_Binary']
            delay = data['Actual_Delay']
            hash_index = data['Hashed_Index']
            class_index = data['Class_Index']
            class_label = data['Class_Label']
            class_color = data['Class_Color']
            
            # Collision Check Logic (requires iteration)
            if hash_index not in hash_map:
                hash_map[hash_index] = [iata_code]
            elif iata_code not in hash_map[hash_index]:
                hash_map[hash_index].append(iata_code)
                collision_count += 1
            
            # --- Populate the QTableWidget row (5 columns) ---
            
            # Col 0: IATA Code (Raw Feature)
            item_iata = QTableWidgetItem(iata_code)
            self.table.setItem(row, 0, item_iata)

            # Col 1: Airport Type (Raw Categorical + Binary Encoded)
            item_type = QTableWidgetItem(f"{airport_type} ({type_binary})")
            # Highlight International airports
            if airport_type == 'International':
                 item_type.setForeground(QColor("#667EEA")) # Indigo
            else:
                 item_type.setForeground(QColor("#A0AEC0")) # Default light gray
            self.table.setItem(row, 1, item_type)
            
            # Col 2: Actual Delay (Continuous Feature)
            item_delay = QTableWidgetItem(f"{delay:+.2f} min")
            
            # Highlight early arrivals in blue
            if delay <= 0:
                 item_delay.setForeground(QColor("#63B3ED")) # Blue
            else:
                 item_delay.setForeground(QColor("#E2E8F0")) # Default light gray
            
            self.table.setItem(row, 2, item_delay)

            # Col 3: Hash Index (Input Feature) - Set bold if collision detected
            item_hash = QTableWidgetItem(str(hash_index))
            if len(hash_map[hash_index]) > 1:
                 item_hash.setFont(QFont("Inter", 10, QFont.Weight.Bold))
                 item_hash.setForeground(QColor("#FBBF24")) # Yellow for collisions
            else:
                 item_hash.setFont(QFont("Inter", 10, QFont.Weight.Normal))
                 item_hash.setForeground(QColor("#E2E8F0"))
            self.table.setItem(row, 3, item_hash)
            
            # Col 4: Delay Class (Target Label) - Apply color from the bucket
            item_class = QTableWidgetItem(f"Class {class_index}: {class_label}")
            # Use the QColor object directly from the DataFrame column
            item_class.setForeground(class_color) 
            item_class.setFont(QFont("Inter", 10, QFont.Weight.Bold))
            self.table.setItem(row, 4, item_class)
            
        # Update the status bar with collision information
        if collision_count > 0:
            self.statusBar.showMessage(
                f"Collision Alert: {collision_count} Hash collisions detected among unique IATA codes (N={HASH_BUCKETS}). Data refreshed. Now tracking {len(IATA_CODES)} airports."
            )
        else:
            self.statusBar.showMessage(
                f"Transformation pipeline refreshed (using Pandas). Hash Buckets (N) = {HASH_BUCKETS}. Tracking {len(IATA_CODES)} airports. No new collisions detected. "
            )


if __name__ == "__main__":
    # Ensure high DPI scaling is enabled for modern displays
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    app = QApplication(sys.argv)
    window = FlightDelayApp()
    window.show()
    sys.exit(app.exec())