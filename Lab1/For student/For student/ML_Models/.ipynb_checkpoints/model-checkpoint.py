import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class CarClassifier:
    def __init__(self, model_name, train_data, test_data):

        '''
        Convert the 'train_data' and 'test_data' into the format
        that can be used by scikit-learn models, and assign 
        training images to self.x_train, 
        training labels to self.y_train,
        testing images to self.x_test, and 
        testing labels to self.y_test.
        These four attributes will be used in 'train' method and 'eval' method.
        '''

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

        # Begin your code (Part 2-1)
        self.x_train = np.array([train_data[0][0].reshape(-1)]) #reshape from dimention 2 to 1
        self.y_train = np.array([train_data[0][1]]) #label
        self.x_test = np.array([test_data[0][0].reshape(-1)])
        self.y_test = np.array([test_data[0][1]]) #label
        
        #iterate and combine all elements in train_data and test_data into np.array
        
        for i in range(1,len(train_data)): #train data
            self.x_train = np.concatenate((self.x_train, np.array([train_data[i][0].reshape(-1)]))) #concatenate: combining arrays
            self.y_train = np.concatenate((self.y_train, np.array([train_data[i][1]])))
            
        for i in range(1,len(test_data)): #test data
            self.x_test = np.concatenate((self.x_test, np.array([test_data[i][0].reshape(-1)])))
            self.y_test = np.concatenate((self.y_test, np.array([test_data[i][1]])))
            
            
        # End your code (Part 2-1)
        
        self.model = self.build_model(model_name)
        
    
    def build_model(self, model_name):
        '''
        According to the 'model_name', you have to build and return the
        correct model.
        '''
        # Begin your code (Part 2-2)
        model = None
        
        
        if(model_name == 'RF'):
            n_estimators = 275 #The number of trees in the forest
            criterion = 'entropy' #or gini is also available, check ref 1
            max_samples = 0.9 # extract max(round(n_samples * max_samples), 1) samples
            max_features = 'sqrt' 
            random_state = 0
            model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_features=max_features, max_samples=max_samples,random_state=random_state)
            
            
            print('number of trees: ', n_estimators)
            print('criterion: ', criterion)
            print('max_samples: ', max_samples)
            print('max_features: ',max_features)
        elif(model_name == 'AB'):
            
            estimator = RandomForestClassifier(n_estimators=10, criterion='entropy')  #The base estimator from which the boosted ensemble is built
            n_estimators = 100  #The maximum number of estimators atwhich boosting is terminated
            learning_rate = 0.7 
            random_state = 2
            #Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier.
            
            algorithm = 'SAMME.R'
            model = AdaBoostClassifier(estimator=estimator,n_estimators=n_estimators, learning_rate=learning_rate,algorithm = algorithm,random_state=random_state)
            
            
            
            print('estimators = ',estimator)
            print('n_estimators = ', n_estimators)
            print('learning_rate = ',learning_rate)
            print('algorithm = ', algorithm)
            
        elif(model_name == 'KNN'):
            n_neighbors = 1
            weights = 'distance'
            algorithm = 'ball_tree'
            #leaf_size = 10
            p = 1
            #n_jobs = 100
            model = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,p=p)
            
            
            print('n_neighbors = ',n_neighbors)
            print('weights = ',weights)
            print('algorithm = ',algorithm)
            #print('leaf_size = ',leaf_size)
            print('p = ',p)
           #print('n_jobs = ', n_jobs)
            
            
            
            
        return model
        # End your code (Part 2-2)

    def train(self):
        '''
        Fit the model on training data (self.x_train and self.y_train).
        '''
        # Begin your code (Part 2-3)
        trainingModel = self.model.fit(self.x_train, self.y_train)
        return trainingModel
        # End your code (Part 2-3)
    
    def eval(self):
        y_pred = self.model.predict(self.x_test)
        print(f"Accuracy: {round(accuracy_score(y_pred, self.y_test), 4)}")
        print("Confusion Matrix: ")
        print(confusion_matrix(y_pred, self.y_test))
    
    def classify(self, input):
        return self.model.predict(input)[0]
        

