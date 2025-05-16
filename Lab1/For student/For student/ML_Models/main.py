import dataset
import model
import detection
import matplotlib.pyplot as plt

# Part 1: Implement loadImages function in dataset.py and test the following code.
print('Loading images')
train_data = dataset.load_images('data/train')
print(f'The number of training samples loaded: {len(train_data)}')
test_data = dataset.load_images('data/test')
print(f'The number of test samples loaded: {len(test_data)}')

print('Show the first and last images of training dataset')
fig, ax = plt.subplots(1, 2)
ax[0].axis('off')
ax[0].set_title('Car')
ax[0].imshow(train_data[1][0], cmap='gray')
ax[1].axis('off')
ax[1].set_title('Non car')
ax[1].imshow(train_data[-1][0], cmap='gray')
plt.show()

# Part 2: Build and train 3 kinds of classifiers: KNN, Random Forest and Adaboost.
# Part 3: Modify difference values at parameter n_neighbors of KNeighborsClassifier, n_estimators 
# of RandomForestClassifier and AdaBoostClassifier, and find better results.
car_clf = model.CarClassifier(
    model_name="AB", # KNN, RF (Random Forest) or AB (AdaBoost)
    train_data=train_data,
    test_data=test_data
)
car_clf.train()
car_clf.eval()

# Part 4: Implement detect function in detection.py and test the following code.
print('\nUse your classifier with video.gif to get the predictions (one .txt and one .png)')
detection.detect('data/detect/detectData.txt', car_clf)
