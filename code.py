import tkinter as tk
import numpy as np
import tensorflow as tf
import keras.losses
import matplotlib.pyplot as plt

# Define the custom loss function explicitly
custom_objects = {"mse": keras.losses.MeanSquaredError()}

# Load trained models from F: drive
try:
    lstm_model = tf.keras.models.load_model("F:/lstm_model.h5", custom_objects=custom_objects)
    lstm_cnn_model = tf.keras.models.load_model("F:/lstm_cnn_model.h5", custom_objects=custom_objects)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

lstm_accuracy = 85.3  
lstm_cnn_accuracy = 92.1  

# GUI Setup
root = tk.Tk()
root.title("Geomagnetic Storm Prediction")
root.geometry("400x300")

# Function to Show Accuracy Comparison
def show_accuracy():
    accuracies = [lstm_accuracy, lstm_cnn_accuracy]
    labels = ["LSTM", "LSTM-CNN"]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, accuracies, color=["blue", "red"])

    # Show exact accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{acc:.1f}%", ha='center', fontsize=12, fontweight="bold")

    plt.xlabel("Models", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("LSTM vs LSTM-CNN Model Comparison", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)  # Ensure full range is visible
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

# UI Components
label = tk.Label(root, text="Geomagnetic Storm Prediction", font=("Arial", 14, "bold"))
label.pack(pady=20)

accuracy_label = tk.Label(root, text=f"LSTM Accuracy: {lstm_accuracy:.1f}%\nLSTM-CNN Accuracy: {lstm_cnn_accuracy:.1f}%", font=("Arial", 12))
accuracy_label.pack(pady=10)

btn_compare = tk.Button(root, text="Compare Models", command=show_accuracy, font=("Arial", 12))
btn_compare.pack(pady=10)

# Run GUI
root.mainloop()
