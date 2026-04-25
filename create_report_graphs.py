from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

model = YOLO("runs/detect/train2/weights/best.pt")

# Validate
results = model.val()

# ---------------- METRICS ----------------
m = results.results_dict

precision = m["metrics/precision(B)"]
recall = m["metrics/recall(B)"]
map50 = m["metrics/mAP50(B)"]
map5095 = m["metrics/mAP50-95(B)"]

print("\n===== FINAL RESULTS =====")
print("Precision:", round(precision,3))
print("Recall:", round(recall,3))
print("mAP@0.5:", round(map50,3))
print("mAP@0.5-0.95:", round(map5095,3))

# ---------------- CONFUSION MATRIX ----------------
# ---------------- CONFUSION MATRIX ----------------
cm = results.confusion_matrix.matrix

plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()


# ---------------- METRIC GRAPH ----------------
df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "mAP@0.5", "mAP@0.5-0.95"],
    "Value": [precision, recall, map50, map5095]
})

plt.figure(figsize=(6,4))
plt.bar(df["Metric"], df["Value"])
plt.ylim(0,1)
plt.title("YOLOv8 License Plate Detection Performance")
plt.ylabel("Score")
plt.grid(axis="y")
plt.savefig("final_metrics_graph.png")
plt.close()

print("\n✅ confusion_matrix.png saved")
print("✅ final_metrics_graph.png saved")
