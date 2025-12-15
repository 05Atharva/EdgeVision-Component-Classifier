## Model Architecture

- Backbone: MobileNetV2 (ImageNet pretrained)
- Input Resolution: 96 × 96 × 3
- Transfer Learning with fine-tuning
- Top 20 layers unfrozen
- Classification Head:
  - GlobalAveragePooling
  - Dense (128, ReLU)
  - Dropout (0.4)
  - Dense (1, Sigmoid)

Designed to be lightweight and suitable for future embedded deployment.
