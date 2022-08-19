import matplotlib.pyplot as plt
from numpy import argmax
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def acc_loss_plot(history):
    train_acc, train_loss =history.history['accuracy'], history.history['loss']
    test_acc, test_loss = history.history['val_accuracy'], history.history['val_loss']
    acc_loss_list = [(train_acc, test_acc), (train_loss, test_loss)]
    labels=[('train accuracy', 'test accuracy'), ('train loss', 'test loss')]
    x_y_labels = ["Accuracy", "Loss"]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    axs = axes.ravel()
    for i, (acc_loss, label, x_y) in enumerate(zip(acc_loss_list, labels, x_y_labels)):
        axs[i].plot(acc_loss[0], color='blue', linestyle='-', label=label[0])
        axs[i].plot(acc_loss[1], color='purple', linestyle='-', label=label[1])
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(x_y)
        axs[i].legend()
    plt.show()

    

def plot_mnist_images(generator, model, row=4, col=4, sklearn_model=False):
    fig, axes = plt.subplots(row, col, figsize=(15, 10))
    for i, (img, label) in enumerate(generator):
        ax = axes[i // col, i % col]
        ax.imshow(img.reshape(45, 45), cmap='gray')
        if sklearn_model:
            img = img.reshape(1, -1)
            idx = model.predict(img)[0] 
        else:
            pred = model.predict(img) 
            idx = argmax(pred, axis=1)[0]
        ax.set_title(f' Predicted Label: {idx} \n True Label {int(label.item())}')
        if i+1 == row*col:
            break
    plt.tight_layout()
    plt.show()


def plot_cm(test_generator, y_pred):
    cm = confusion_matrix(test_generator.classes, y_pred);
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_generator.class_indices.keys()));
    fig, ax = plt.subplots(figsize=(35,30));
    plt.title('Confusion Matrix');
    cmp.plot(ax=ax, cmap=plt.cm.Blues);










