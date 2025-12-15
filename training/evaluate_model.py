import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Load model
model = tf.keras.models.load_model("model/component_classifier_finetuned.h5")

# Validation directory (same structure as training)
VAL_DIR = "path_to_dataset"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.3
)

val_data = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(96, 96),
    batch_size=8,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# Predictions
preds = model.predict(val_data)
y_pred = (preds > 0.5).astype(int).flatten()
y_true = val_data.classes

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Save results
with open("results/confusion_matrix.txt", "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)
