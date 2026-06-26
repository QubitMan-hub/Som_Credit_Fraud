# Som_Credit_Fraud
This project is an advanced case study that utilizes an unsupervised Machine Learning technique a **Self-Organizing Map (SOM)**, to detect potential fraud or anomalies in credit card applications. 

By analyzing patterns and relationships across multiple customer features, the SOM maps high-dimensional application data onto a 2D grid. This allows for the visual identification of outliers, which often represent fraudulent applications.

## Dataset
The project utilizes a specific dataset named verbatim: **`Credit_Card_Applications.csv`**. 

This dataset contains a mix of categorical and numerical features representing attributes provided by customers during their credit card application process. The final column acts as a target variable indicating whether the application was approved or rejected.

## Technical Workflow
1. **Library Dependencies**: Built using `numpy`, `pandas`, and `matplotlib` for data manipulation and visualization, alongside `minisom` for building the Self-Organizing Map.
2. **Data Preprocessing**: Features are isolated from the class labels and normalized using a `MinMaxScaler` into a strict $(0, 1)$ range to ensure uniform weight updates during training.
3. **SOM Architecture**: A $10 \times 10$ grid is initialized with a structural input length of 15. The model undergoes a random weights initialization followed by 100 training iterations using an initial learning rate of 0.5.
4. **Visualization**: The resulting map plots the Inter-Neuron Distance (MID) to visually isolate outlying nodes (potential fraud clusters) from normal application behavior.

## How to Run
Ensure you have the required packages installed:
```bash
pip install numpy pandas matplotlib minisom scikit-learn
