import sys
import hashlib
import random
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

# Import TensorFlow/Keras components for the Embeddings implementation (Section 1.B)
# NOTE: The import 'from tensorflow.keras.utils import plot_model' has been removed
# to fix the ImportError caused by missing Graphviz dependencies.
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Embedding, Dense, Concatenate, Flatten

# PySide6 imports for visualization
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTableWidget, QTableWidgetItem, QLabel, QGroupBox, QHeaderView,
    QAbstractItemView, QStatusBar
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor

# ====================================================================
# HOMEWORK 1 - DATA LOADING (MLOps Best Practice: Load from .csv)
# ====================================================================

DATA_FILENAME = "airport_data.csv"

def load_airport_data(filename: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """Loads airport data from a CSV file and converts it into a usable structure."""
    try:
        # Load the CSV file into a Pandas DataFrame
        df = pd.read_csv(filename)
        # Ensure IATA codes are unique
        df = df.drop_duplicates(subset=['IATA_Code'], keep='first')
        
        # Convert DataFrame to a dictionary structure for quick lookup in the visualization
        airport_data_dict = df.set_index('IATA_Code').to_dict('index')
        return df, airport_data_dict
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found. Please ensure the CSV is in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        sys.exit(1)

# Load the data once at the start
DF_AIRPORTS, AIRPORT_DATA = load_airport_data(DATA_FILENAME)
IATA_CODES = DF_AIRPORTS['IATA_Code'].tolist()
TARGET_AIRPORT_COUNT = len(IATA_CODES)
VOCABULARY_SIZE = TARGET_AIRPORT_COUNT + 1 # +1 for out-of-vocabulary (O.O.V.) token

# ====================================================================
# HOMEWORK 1 - SECTION 1A: Hashed Feature Implementation (IATA Code)
# ====================================================================

# Hashed Feature Configuration (1.A.1)
HASH_BUCKETS = 100

def hashed_feature_mapping(iata_code: str, num_buckets: int) -> int:
    """
    1. Code: Implements the Hashed Feature solution, mapping an IATA code string 
    to a bucket index (e.g., using 100 buckets).
    """
    if not iata_code:
        return 0
    encoded_string = iata_code.encode('utf-8')
    # Use SHA-256 for a stable and well-distributed hash
    hash_value = int(hashlib.sha256(encoded_string).hexdigest(), 16)
    # The modulo operation maps the large hash to a bucket index
    return hash_value % num_buckets


# ====================================================================
# HOMEWORK 1 - SECTION 1B: Embeddings Implementation (Neural Network Design)
# ====================================================================

# Example: Use 10 as the dimension, a common heuristic (root of vocabulary size)
EMBEDDING_DIMENSION = int(np.sqrt(VOCABULARY_SIZE))

def build_embedding_model(vocab_size: int, embed_dim: int, num_classes: int) -> Model:
    """
    1. Code/Description: Defines a Keras model architecture demonstrating the 
    Embeddings design pattern for the categorical Departure Airport feature.
    """
    
    # Input 1: Categorical Feature (Departure Airport)
    airport_input = Input(shape=(1,), name='airport_input', dtype='int64')
    
    # The Embedding Layer (The key to Embeddings design pattern)
    # The layer learns a dense, low-dimensional vector representation for each airport.
    airport_embedding = Embedding(
        input_dim=vocab_size,         # Total number of unique airports (Vocabulary Size)
        output_dim=embed_dim,         # The dimensionality of the embedding vector (e.g., 10)
        name='airport_embedding'
    )(airport_input)
    
    # Flatten the embedding vector (e.g., shape (1, 10) -> shape (10,))
    flattened_embedding = Flatten()(airport_embedding)
    
    # Input 2: Numerical/Continuous Features 
    # For demonstration, we assume 5 other numerical features (e.g., Scheduled_Time, Distance)
    numerical_input = Input(shape=(5,), name='numerical_input', dtype='float32')
    
    # Combine the Embedding vector and the numerical features
    concatenated = Concatenate()([flattened_embedding, numerical_input])
    
    # Neural Network Layers (MLOps Architecture)
    hidden_layer_1 = Dense(64, activation='relu', name='hidden_1')(concatenated)
    hidden_layer_2 = Dense(32, activation='relu', name='hidden_2')(hidden_layer_1)
    
    # Output Layer (Reframed Classification Task - Section 2)
    # The final prediction is a probability distribution over the delay classes.
    output_layer = Dense(num_classes, activation='softmax', name='delay_prediction_output')(hidden_layer_2)
    
    # Create the final model
    model = Model(inputs=[airport_input, numerical_input], outputs=output_layer)
    
    # Print a summary to show the layer structure and parameter count 
    print("\n--- Keras Model Summary (Embeddings Architecture) ---")
    model.summary()
    
    # The plot_model function call has been removed to avoid dependency errors.
    
    return model

# Initialize the model structure (3 classes from Section 2)
Embedding_NN = build_embedding_model(VOCABULARY_SIZE, EMBEDDING_DIMENSION, 3)


# ====================================================================
# HOMEWORK 1 - SECTION 2: Reframing Implementation (Arrival Delay)
# ====================================================================

# Buckets for the Reframing (Output Label) (2.2)
BUCKET_RANGES = [
    (10, "On-Time/Early", QColor("#34D399")),       # Class 0: Delay <= 10 min
    (45, "Medium Delay", QColor("#FBBF24")),        # Class 1: 10 min < Delay <= 45 min
    (float('inf'), "Significant Delay", QColor("#F87171")) # Class 2: Delay > 45 min
]

def bucketize_delay(arrival_delay_minutes: float) -> Tuple[int, str, QColor]:
    """
    2. Bucketing: Converts continuous delay into a discrete classification label 
    (Reframing the continuous output).
    """
    for index, (upper_bound, label, color) in enumerate(BUCKET_RANGES):
        if arrival_delay_minutes <= upper_bound:
            return index, label, color
    
    # Fallback (should be covered by float('inf'))
    return len(BUCKET_RANGES) - 1, BUCKET_RANGES[-1][1], BUCKET_RANGES[-1][2]

def generate_delay() -> float:
    """Generates a random, realistic, and dynamic delay (skewed towards 0/+ve)."""
    # Uses a distribution that frequently yields values near zero or slightly positive
    return round(random.uniform(-15, 5) + random.expovariate(0.05), 2)


# ====================================================================
# PART C: PySide6 GUI Application (Visualization of 1A and 2)
# ====================================================================

class FlightDelayApp(QMainWindow):
    """Main window for the Dynamic ML Data Transformation GUI."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HW 1: Feature & Problem Reframing Viewer (Loaded from CSV)")
        self.setGeometry(100, 100, 1300, 650)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_table_data)
        
        self.apply_styles()
        self.setup_ui()
        
        self.update_table_data()
        self.timer.start(3000)

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
        title_label = QLabel("HW 1: Flight Delay Prediction - Feature Transformation Viewer (Loaded from CSV)")
        title_font = QFont("Inter", 20, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #63B3ED; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        # --- Reframing Legend Group (Section 2) ---
        reframing_group = QGroupBox("Section 2: Output Label Reframing (Delay $\rightarrow$ Classification)")
        reframing_layout = QHBoxLayout(reframing_group)
        reframing_group.setMaximumHeight(80)
        
        for index, (upper, label, color) in enumerate(BUCKET_RANGES):
            delay_range = f"($\le$ {upper} min)"
            if index > 0 and upper == float('inf'):
                delay_range = f"($> {BUCKET_RANGES[index-1][0]}$ min)"
            elif index > 0:
                delay_range = f"({BUCKET_RANGES[index-1][0]} < min $\le$ {upper} min)"
            
            reframing_label = QLabel(f"Class {index}: {label} {delay_range}")
            reframing_label.setStyleSheet(f"color: {color.name()}; font-weight: bold; padding: 5px;")
            reframing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            reframing_layout.addWidget(reframing_label)
            
        main_layout.addWidget(reframing_group)

        # --- Results Table (Section 1A, 2) ---
        results_group = QGroupBox(f"Section 1A: Hashed Feature Mapping for {TARGET_AIRPORT_COUNT} Airports (N={HASH_BUCKETS})")
        results_layout = QVBoxLayout(results_group)
        
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "IATA Code (Raw Feature)", 
            "Airport Details (City, Country, Type)",
            "Actual Delay (Continuous)", 
            f"Hashed Index (0 to {HASH_BUCKETS-1})", 
            "Delay Class (Target Label)"
        ])
        self.table.setRowCount(TARGET_AIRPORT_COUNT) 
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        results_layout.addWidget(self.table)
        
        main_layout.addWidget(results_group)
        
        # --- Status Bar ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(f"System Initialized. Data loaded from {DATA_FILENAME}. Hash Buckets (N) set to {HASH_BUCKETS}. Auto-updating every 3 seconds...")
        

    def update_table_data(self):
        """Executes the data transformation and updates the table."""
        
        # 1. Prepare Raw Data (Data is read from the global DataFrame now)
        raw_data = []
        for code in IATA_CODES:
            details = AIRPORT_DATA[code]
            raw_data.append({
                'IATA_Code': code,
                'Actual_Delay': generate_delay(),
                'Airport_Type': details['Type'],
                'Airport_City': details['City'],
                'Airport_Country': details['Country']
            })
        
        df = pd.DataFrame(raw_data)
        
        # 2. Apply Vectorized Transformations
        
        # A. Feature Hashing
        df['Hashed_Index'] = df['IATA_Code'].apply(
            lambda x: hashed_feature_mapping(x, HASH_BUCKETS)
        )
        
        # C. Reframing/Bucketing
        df[['Class_Index', 'Class_Label', 'Class_Color']] = df['Actual_Delay'].apply(
            lambda x: pd.Series(bucketize_delay(x))
        )
        
        # 3. Populate Table and Check Collisions
        hash_map: Dict[int, List[str]] = {}
        
        for row, data in df.iterrows():
            iata_code = data['IATA_Code']
            airport_type = data['Airport_Type']
            airport_city = data['Airport_City']
            airport_country = data['Airport_Country']
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
            
            # --- Populate the QTableWidget row ---
            
            # Col 0: IATA Code
            item_iata = QTableWidgetItem(iata_code)
            self.table.setItem(row, 0, item_iata)

            # Col 1: Airport Details
            details_text = f"{airport_city}, {airport_country} ({airport_type})"
            item_details = QTableWidgetItem(details_text)
            item_details.setForeground(QColor("#9DECF9") if airport_type == 'International' else QColor("#A0AEC0"))
            self.table.setItem(row, 1, item_details)
            
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
            f"Transformation refreshed. Airports loaded: {len(IATA_CODES)}. "
            f"Total unique buckets with collisions: {total_unique_collisions}. "
        )


if __name__ == "__main__":
    # Ensure high DPI scaling is enabled for modern displays
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    app = QApplication(sys.argv)
    window = FlightDelayApp()
    window.show()
    sys.exit(app.exec())