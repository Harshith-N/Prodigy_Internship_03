import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

data_dir = '/Users/harshithn/Desktop/Prodigy_Internship_3/dogs-vs-cats/train'  

img_size = 128

def load_data(data_dir, img_size):
    data = []
    labels = []
    for img in os.listdir(data_dir):
        try:
            img_path = os.path.join(data_dir, img)
            img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_array is None:
                continue  
            
            if 'cat' in img:
                label = 0  
            elif 'dog' in img:
                label = 1  
            else:
                continue  
            
            resized_img = cv2.resize(img_array, (img_size, img_size))
            data.append(resized_img)
            labels.append(label)
        except Exception as e:
            pass
    return np.array(data), np.array(labels)

data, labels = load_data(data_dir, img_size)

data = data.reshape(data.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=500)  
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

svm_model = SVC(kernel='rbf', gamma='scale', C=1.0)
svm_model.fit(X_train_pca, y_train)

y_pred = svm_model.predict(X_test_pca)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve

cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()

report = classification_report(y_test, y_pred)
print(report)

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower right")
plt.show()


import csv 

test_data_dir = '/Users/harshithn/Desktop/Prodigy_Internship_3/dogs-vs-cats/test1'
test_data, _ = load_data(test_data_dir, img_size)

test_data = test_data.reshape(test_data.shape[0], -1)

test_data = scaler.transform(test_data)

test_data_pca = pca.transform(test_data)

y_pred = svm_model.predict(test_data_pca)

with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image', 'Prediction'])  
    for i, pred in enumerate(y_pred):
        writer.writerow([f'test_{i}.jpg', 'cat' if pred == 0 else 'dog'])