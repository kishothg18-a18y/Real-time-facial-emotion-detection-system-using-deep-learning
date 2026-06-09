    # evaluate.py
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report, confusion_matrix

    MODEL_PATH = "models/emotion_mobilenetv2.h5"
    DATA_DIR = "datasets/processed"
    IMG_SIZE=(224,224)
    BATCH_SIZE=32

    def evaluate():
        model = load_model(MODEL_PATH)
        test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
            os.path.join(DATA_DIR, "test"),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        preds = model.predict(test_gen, verbose=1)
        y_pred = np.argmax(preds, axis=1)
        y_true = test_gen.classes
        print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys())
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('eval_confusion_matrix.png')
        print("Saved confusion matrix to eval_confusion_matrix.png")

    if __name__ == "__main__":
        evaluate()