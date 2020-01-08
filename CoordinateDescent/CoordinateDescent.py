import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import sys
import math
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

class LogisticRegr():
    def __init__(self, iters):
        self.iters =  iters;
    def regression(self, train_data):
        logisticRegr = LogisticRegression(C = sys.maxsize, solver = 'lbfgs', max_iter = 100)
        logisticRegr.fit(train_data_norm, train_label)
        predictions = logisticRegr.predict(train_data_norm)
        #-log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp)) 
        #implementing logistic loss function
        print(log_loss(train_label, predictions))

class CoordinateDescent():
    def __init__(self, iters, gamma = 0.001, option = "Greedy"):
        self.iters =  iters;
        self.option = option;
        self.gamma = gamma;
        initial_weights = np.zeros(13)

    def loss(self, weight, train_label, train_data_norm):
        train_label = np.asarray(train_label)
        train_data_norm = np.asarray(train_data_norm)
        loss = 0;
        for i in range(0,130):
        #(1*1)*(1*d), this is for iterating through i
            loss = loss + math.log(1 + math.exp(train_label[i]*np.dot(weight, train_data_norm[i]))) 
        return loss;

    def loss_diff(self, weight, j, train_label, train_data_norm):
        loss = 0;
        train_label = np.asarray(train_label)
        train_data_norm = np.asarray(train_data_norm)
        for i in range(0,130):
            #(1*1)*(1*d), this is for iterating through i
            #loss = loss + ((-1 * np.dot(train_label[i], train_data[i]))/(1 + math.exp(train_label[i]*np.dot(weights, train_data[i]))))
            loss = loss + (((train_label[i] * train_data_norm[i][j]))/(1 + math.exp(train_label[i]* weight[j] * train_data_norm[i][j])))
            return loss;
        
    def weight_random_update(self,j, init_weight):
        updated_weight = init_weight[j];
        for i in range (0,130):        
            updated_weight = updated_weight + ((self.gamma)*loss_diff(init_weight, j))  #we dont consider all the dimensions
        init_weight[j] = updated_weight;

    def coordinate_descent_random(self, train_data, train_label):
        for iter in range(0, self.iters):
            rand_coord = np.arange(13)
            np.random.shuffle(rand_coord)
            for i in range(0,13):
                coord = rand_coord[i];
                #the initial_weights will change continuously
                self.weight_random_update(coord , self.initial_weights)  #this will change the weight of the weight of "coord" coordinate
                #print("The weights passed",init_weight)

        print(self.loss(self.initial_weights, train_data, train_label))

    def duplicate_array(A,B):
        for i in range(0,13):
            B[i] = A[i]

    def coordinate_descent_greedy(self, train_data, train_label):
        init_weight_test_each_coord = np.zeros(13)
        for iter in range(0, self.iters*13):
            #for each of the dimensions
            loss_each_coord = np.zeros(13)
            grad_each_coord = np.zeros(13)
            duplicate_array(self.initial_weights , init_weight_test_each_coord)
            for i in range(0,13):
                #coord = rand_coord[i];
                #in our algorithm we choose the coordinate by seeing for which weight update would the loss be mi
                ###################################################### 
                #weight_random_update(i, init_weight_test_each_coord);
                #loss_each_coord[i] = loss(init_weight_test_each_coord);
                ################################################
                grad_each_coord[i] = self.loss_diff(init_weight,i, train_data, train_label )
                #duplicate_array(init_weight, init_weight_test_each_coord)
            #coord = np.argmin(loss_each_coord)
            print(grad_each_coord)
            coord = np.argmin(grad_each_coord)
            weight_random_update(coord, self.initial_weights)


if __name__ == '__main__':
    #reading the data points
    a = list(np.arange(1,14).astype('str'))
    a = ["Feat_" + j for j in a]
    a = ','.join(a)
    a = ("label," + a).split(",")
    col_Names = a
    df = pd.read_csv('wine_data.txt', names = col_Names)

    #read the train_label and removing the data points with label 3
    train_label = df['label']
    y_train = np.ravel(train_label)
    idx = np.where(y_train == 3)

    #dropping the rows with label 3
    df = df.drop(df.index[idx])   
    train_label_mat = df.as_matrix(columns = ['label'])
    for i in range(0, len(train_label)):
        if(train_label[i] == 2):
            train_label[i] = -1
    df = df.drop(columns = ['label'])
    train_data = df.as_matrix()

    scaler = MinMaxScaler()
    train_data_norm = scaler.fit_transform(train_data)

    logRegr = LogisticRegr(100);
    LogisticRegr.regression(train_data_norm);

    coordDescRand = CoordinateDescent(100);
    coordDescRand.coordinate_descent_random(train_data_norm. train_label);

    coordDescGreedy = CoordinateDescent(100);
    coordDescRand.coordinate_descent_greedy(train_data_norm. train_label);

    

