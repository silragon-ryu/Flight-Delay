import sys
import hashlib
import random
import pandas as pd
from typing import List, Tuple, Dict, Any
import string

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

# This section defines a core set of real airport data (17 entries)
KNOWN_AIRPORT_DATA = {
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

# --- Function to Simulate the Required Number of Airports (347) ---
def generate_simulated_iata_codes(num_desired: int, existing_codes: List[str]) -> List[str]:
    """Generates unique 3-letter strings to meet a total IATA count."""
    count_to_generate = num_desired - len(existing_codes)
    if count_to_generate <= 0:
        return existing_codes
        
    simulated_codes = set(existing_codes)
    # Generates 3-letter codes until the target count is reached
    while len(simulated_codes) < num_desired:
        code = ''.join(random.choices(string.ascii_uppercase, k=3))
        simulated_codes.add(code)
    
    return list(simulated_codes)

# Required number of airports: 347
TARGET_AIRPORT_COUNT = 347

# Generate the full set of IATA codes (17 real + 330 simulated = 347)
IATA_CODES = generate_simulated_iata_codes(TARGET_AIRPORT_COUNT, list(KNOWN_AIRPORT_DATA.keys()))

# Create the full data dictionary for the simulation
AIRPORT_DATA = KNOWN_AIRPORT_DATA.copy()
for code in IATA_CODES:
    if code not in AIRPORT_DATA:
        # Assign random properties to simulated airports 
        airport_type = random.choice(["Domestic", "International"])
        AIRPORT_DATA[code] = {"type": airport_type, "city": f"City_{code}", "country": f"Country_{code}"}

# Hashed Feature Configuration
HASH_BUCKETS = 100

# Buckets for the Reframing (Output Label)
# 0: On-Time/Early (Delay <= 10 min)
# 1: Medium Delay (10 min < Delay <= 45 min)
# 2: Significant Delay (Delay > 45 min)
BUCKET_RANGES = [
    (10, "On-Time/Early", QColor("#34D399")),  # Green
    (45, "Medium Delay", QColor("#FBBF24")),   # Yellow
    (float('inf'), "Significant Delay", QColor("#F87171")) # Red
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
    return round(random.uniform(-15, 5) + random.expovariate(0.05), 2)


# ====================================================================
# PART B: PySide6 GUI Application
# ====================================================================

class FlightDelayApp(QMainWindow):
    """Main window for the Dynamic ML Data Transformation GUI."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic ML Feature Engineering Viewer (PySide6)")
        self.setGeometry(100, 100, 1300, 650)
        
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
            QMainWindow { background-color: #2D3748; } 
            QWidget#CentralWidget { padding: 15px; }
            QGroupBox { 
                background-color: #4A5568; border: 2px solid #63B3ED; 
                border-radius: 8px; margin-top: 10px; font-weight: bold;
                color: #E2E8F0; padding-top: 15px; 
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; }
            QTableWidget { 
                background-color: #2D3748; border: 1px solid #718096;
                border-radius: 6px; gridline-color: #4A5568;
                color: #E2E8F0; font-size: 14px; 
            }
            QHeaderView::section {
                background-color: #4A5568; color: #9DECF9;
                padding: 8px; border: 1px solid #718096;
                font-weight: bold; font-size: 14px; 
            }
            QStatusBar { background-color: #1A202C; color: #A0AEC0; font-size: 12px; border-top: 1px solid #718096; }
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
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "IATA Code (Raw Feature)", 
            "Type (Domestic/Intl.) [Binary]",
            "Actual Delay (Continuous)", 
            f"Hashed Index (0 to {HASH_BUCKETS-1})", 
            "Delay Class (Target Label)"
        ])
        # Set the row count based on the 347 airports
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
        updates the table for all 347 airports.
        """
        
        # 1. Generate Raw Data 
        raw_data = []
        for code in IATA_CODES:
            details = AIRPORT_DATA[code]
            raw_data.append({
                'IATA_Code': code,
                'Actual_Delay': generate_delay(),
                'Airport_Type': details['type']
            })
        
        df = pd.DataFrame(raw_data)
        
        # 2. Apply Vectorized Transformations
        
        # A. Feature Hashing
        df['Hashed_Index'] = df['IATA_Code'].apply(
            lambda x: hashed_feature_mapping(x, HASH_BUCKETS)
        )
        
        # B. Binary Encoding
        df['Type_Binary'] = df['Airport_Type'].apply(
            lambda x: 1 if x == 'International' else 0
        )
        
        # C. Reframing/Bucketing
        df[['Class_Index', 'Class_Label', 'Class_Color']] = df['Actual_Delay'].apply(
            lambda x: pd.Series(bucketize_delay(x))
        )
        
        # 3. Populate Table and Check Collisions
        collision_count = 0
        hash_map: Dict[int, List[str]] = {}
        
        for row, data in df.iterrows():
            iata_code = data['IATA_Code']
            airport_type = data['Airport_Type']
            type_binary = data['Type_Binary']
            delay = data['Actual_Delay']
            hash_index = data['Hashed_Index']
            class_index = data['Class_Index']
            class_label = data['Class_Label']
            class_color = data['Class_Color']
            
            # Collision Check Logic
            if hash_index not in hash_map:
                hash_map[hash_index] = [iata_code]
            elif iata_code not in hash_map[hash_index]:
                hash_map[hash_index].append(iata_code)
                collision_count += 1 
            
            # --- Populate the QTableWidget row ---
            
            # Col 0: IATA Code
            item_iata = QTableWidgetItem(iata_code)
            self.table.setItem(row, 0, item_iata)

            # Col 1: Airport Type
            item_type = QTableWidgetItem(f"{airport_type} ({type_binary})")
            item_type.setForeground(QColor("#667EEA") if airport_type == 'International' else QColor("#A0AEC0"))
            self.table.setItem(row, 1, item_type)
            
            # Col 2: Actual Delay
            item_delay = QTableWidgetItem(f"{delay:+.2f} min")
            item_delay.setForeground(QColor("#63B3ED") if delay <= 0 else QColor("#E2E8F0"))
            self.table.setItem(row, 2, item_delay)

            # Col 3: Hash Index (Highlight collisions)
            item_hash = QTableWidgetItem(str(hash_index))
            if len(hash_map[hash_index]) > 1:
                item_hash.setFont(QFont("Inter", 10, QFont.Weight.Bold))
                item_hash.setForeground(QColor("#FBBF24")) # Yellow for collisions
            else:
                item_hash.setFont(QFont("Inter", 10, QFont.Weight.Normal))
                item_hash.setForeground(QColor("#E2E8F0"))
            self.table.setItem(row, 3, item_hash)
            
            # Col 4: Delay Class (Target Label)
            item_class = QTableWidgetItem(f"Class {class_index}: {class_label}")
            item_class.setForeground(class_color) 
            item_class.setFont(QFont("Inter", 10, QFont.Weight.Bold))
            self.table.setItem(row, 4, item_class)
            
        # Update the status bar with collision information
        total_unique_collisions = sum(1 for index in hash_map if len(hash_map[index]) > 1)
        
        self.statusBar.showMessage(
            f"Transformation pipeline refreshed (using Pandas). Hash Buckets (N) = {HASH_BUCKETS}. "
            f"Tracking {len(IATA_CODES)} airports. Total unique collisions: {total_unique_collisions}. "
            f"Total items involved in collision: {collision_count} (items after the first one hit the bucket). "
        )


if __name__ == "__main__":
    # Ensure high DPI scaling is enabled for modern displays
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    app = QApplication(sys.argv)
    window = FlightDelayApp()
    window.show()
    sys.exit(app.exec())