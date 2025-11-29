# EEG-Signal-for-Cognitive-and-Sensory-tasks
Use EEG signals recorded during multiple cognitive and sensory tasks to train a
deep learning model capable of classifying mental states based on raw EEG activity.

**Dataset**:
EEG data from 4 subjects recorded using Bio Radio Lab equipment.
Electrode positions: T7, F8, Cz, P4.
Tasks performed:
  - Blink 5 times
  - Eyes closed/relaxation
  - Mental calculations
  - Listening to music
  - Watching a video
  - Listing words in a category
  - Counting colored rectangles
Raw data includes metadata + EEG samples stored as JSON lists.

# Data preparation:
1. Load CSV from Google Drive into pandas DataFrame.
2. Remove irrelevant labels (“unlabeled”, “everyone paired”).
3. Drop non-essential metadata columns.
4. Reset index for clean iteration.
5. Convert 'raw_values' JSON strings → Python lists.
6. Filter by signal quality threshold.
7. Visual inspection using custom plotting functions.

# Label processing:
1. Merge similar task labels into unified classes (e.g., different math-task labels merged).
2. Encode string labels numerically using LabelEncoder.
3. Scale each EEG signal (raw_values) to range [0, 1] with MinMaxScaler.

# Model input preparation:
1. Split into training/test sets (test size = 15%).
2. Reshape raw EEG vectors to model input shape (512 samples × 1 channel).
3. Compute class weights to address label imbalance.

# Model architecture (1D CNN):
Input: (512, 1)
Layers:
  Conv1D → BatchNorm (several blocks)
  Filters: 32 → 64 → 64 → 128 → 256 → 512 → 1024 → 1024
  Strides: mostly 2
  Kernel sizes: 3 or 5
  Dropout: 0.1
  Flatten
  Dense layers: 4096 → 2048 → 1024 → 512 → 128
  L2 regularization in deeper Dense layers
Output: softmax over number of classes

# Training setup:
learning_rate = 0.001
batch_size = 128
epochs = 50
loss = categorical_crossentropy
optimizer = Adam
callbacks:
  - ModelCheckpoint
  - LearningRateScheduler or ReduceLROnPlateau
Class weights applied during training.

# Evaluation:
Compute:
  - accuracy
  - precision
  - recall
  - ROC-AUC (per class or macro)
Visual comparison of predicted vs actual labels.
Plot performance curves and EEG predictions for multiple examples.

# Project outcome:
Model successfully distinguishes EEG patterns across tasks such as blinking,
relaxation, music, math tasks, and visual attention.
Demonstrates the feasibility of using consumer-grade EEG for classification
of cognitive/sensory activities.

# Dependencies:
pip install numpy pandas matplotlib scikit-learn tensorflow keras json

How to run:
1. Upload dataset CSV + notebook to Google Colab.
2. Mount Google Drive in the notebook.
3. Adjust file paths if needed.
4. Run cells sequentially.
5. After training, run evaluation cells to view metrics and predictions.

Notes:
- Scaling EEG values is crucial for stable training.
- Deep CNN depth helps extract temporal patterns from raw EEG.
- Dropout and L2 regularization reduce overfitting.
- Results depend on signal quality and class balance.

License:
Project for GBM562 – EEG Signal for Cognitive and Sensory Tasks
USEK – Spring 2023-2024.
