import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

class randomSamples():
    def __init__(self, N, option = 'kmeans'):
        self.N = N; #no of data samples
        self.training_samples = np.zeros(shape=(N,784))
        self.label_training_samples = np.zeros(N)
        self.option = option

    def get_error(self, X_train, Y_train, X_test, Y_test):
        random_val = np.arange(60000)
        np.random.shuffle(random_val)
        random_val=random_val[0:self.N]  #choosing N samples at random
        x_train = X_train[random_val]
        y_train = Y_train[random_val]
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train) 
        y_predict = knn.predict(X_test)
        return np.sum(y_predict == Y_test)

#this is the class for prototype selection
class Prototype():
    def __init__(self, N, option = 'kmeans'):
        self.N = N; #no of data samples
        self.training_samples = np.zeros(shape=(N,784))
        self.label_training_samples = np.zeros(N)
        self.option = option


    #this function will sort the samples ofmat_labels in increasing order and then choose N no of mat_labels at regular intervals
    def choose_samples(self, mat, mat_labels, N):
        arr_sum = mat.sum(axis = 1)
        print(arr_sum.shape)
        num_samples_label = (arr_sum.shape[0])
        req_sample_per_label =(N/10)
        arr_sort_labels = np.argsort(arr_sum) #the zero label difference 
        samples_step = (num_samples_label/(req_sample_per_label)) #5923/1000 will give 5 samples
        
        random_no_arr = []  #labels of the random thousand samples to be chosen
        i = 0
        while (i + samples_step < num_samples_label) :    #go till req sample_per_label
            array_rand = np.arange(i, i + samples_step - 1)
            np.random.shuffle(array_rand)
            random_no_arr.append(array_rand[0]); #selecting the random element
            i = i + samples_step
            array_rand = np.arange(i, num_samples_label)  #i to num_samples_label
            np.random.shuffle(array_rand)
            random_no_arr.append(array_rand[0])
        
            np.asarray(random_no_arr)
            np.random.shuffle(random_no_arr)
            random_no_arr = random_no_arr[0:req_sample_per_label]
            labels_pickuparr = arr_sort_labels[random_no_arr]
            return labels_pickuparr

    def choose_training_sample(self, label, N, array_sort, indices_array):
            each_sample = N/10
            indices = np.arange(label * each_sample , ((label + 1) * (each_sample)))
        
            for i in range(0,len(indices)):
                self.training_samples[(label * each_sample) + i] = array_sort[indices_array[i]]
                self.label_training_samples[(label * each_sample) + i] = label

    def choose_sample_kmeans(self, mat_label, N):
            kmeans = KMeans(n_clusters=N)
            kmeans = kmeans.fit(mat_label)
            centroids = kmeans.cluster_centers_
            return centroids


    def data_evaluation(self, X_train, Y_train):
        inds_zero = []
        for index,item in enumerate(Y_train):
            if item == 0:
                inds_zero.append(index)
        inds_one = []
        for index,item in enumerate(Y_train):
            if item == 1:
                inds_one.append(index)
        inds_two = []
        for index,item in enumerate(Y_train):
            if item == 2:
                inds_two.append(index)
        inds_three = []
        for index,item in enumerate(Y_train):
            if item == 3:
                inds_three.append(index)
        inds_four = []
        for index,item in enumerate(Y_train):
            if item == 4:
                inds_four.append(index)
        inds_five = []
        for index,item in enumerate(Y_train):
            if item == 5:
                inds_five.append(index)
        inds_six = []
        for index,item in enumerate(Y_train):
            if item == 6:
                inds_six.append(index)
        inds_seven = []
        for index,item in enumerate(Y_train):
            if item == 7:
                inds_seven.append(index)
        inds_eight = []
        for index,item in enumerate(Y_train):
            if item == 8:
                inds_eight.append(index)
        inds_nine = []
        for index,item in enumerate(Y_train):
            if item == 9:
                inds_nine.append(index)
        np.asarray(inds_zero)
        np.asarray(inds_one)
        np.asarray(inds_two)
        np.asarray(inds_three)
        np.asarray(inds_four)
        np.asarray(inds_five)
        np.asarray(inds_six)
        np.asarray(inds_seven)
        np.asarray(inds_eight)
        np.asarray(inds_nine)
        
        mat_zero = np.zeros(shape=(len(inds_zero),784))
        mat_one = np.zeros(shape=(len(inds_one),784))
        mat_two = np.zeros(shape=(len(inds_two),784))
        mat_three = np.zeros(shape=(len(inds_three),784))
        mat_four = np.zeros(shape=(len(inds_four),784))
        mat_five = np.zeros(shape=(len(inds_five),784))
        mat_six = np.zeros(shape=(len(inds_six),784))
        mat_seven = np.zeros(shape=(len(inds_seven),784))
        mat_eight = np.zeros(shape=(len(inds_eight),784))
        mat_nine = np.zeros(shape=(len(inds_nine),784))
        
        #getting the normalized 
        mat_zero = X_train[inds_zero]
        mat_zero_mod = pow((mat_zero - mat_zero.mean(axis=0)),2)
        mat_one = X_train[inds_one]
        mat_one_mod = pow((mat_one - mat_one.mean(axis=0)),2)
        mat_two = X_train[inds_two]
        mat_two_mod = pow((mat_two - mat_two.mean(axis=0)),2)
        mat_three = X_train[inds_three]
        mat_three_mod = pow((mat_three - mat_three.mean(axis=0)),2)
        mat_four = X_train[inds_four]
        mat_four_mod = pow((mat_four - mat_four.mean(axis=0)),2)
        mat_five = X_train[inds_five]
        mat_five_mod = pow((mat_five - mat_five.mean(axis=0)),2)
        mat_six = X_train[inds_six]
        mat_six_mod = pow((mat_six - mat_six.mean(axis=0)),2)
        mat_seven = X_train[inds_seven]
        mat_seven_mod = pow((mat_seven - mat_seven.mean(axis=0)),2)
        mat_eight = X_train[inds_eight]
        mat_eight_mod = pow((mat_eight - mat_eight.mean(axis=0)),2)
        mat_nine = X_train[inds_nine]
        mat_nine_mod = pow((mat_nine - mat_nine.mean(axis=0)),2)

        if(self.option == 'Incrsort'):
 
            self.labels_pickuparr = self.choose_samples(mat_zero_mod, mat_zero, 1000) 
            self.training_sample(0, 1000, mat_zero ,labels_pickuparr)

            self.labels_pickuparr = self.choose_samples(mat_one_mod, mat_one, 1000)  
            self.training_sample(1, 1000, mat_one ,labels_pickuparr)

            self.labels_pickuparr = self.choose_samples(mat_two_mod, mat_two, 1000)  
            self.training_sample(2, 1000, mat_two ,labels_pickuparr)

            self.labels_pickuparr = self.choose_samples(mat_three_mod, mat_three, 1000)  
            self.training_sample(3, 1000, mat_three ,labels_pickuparr)

            self.labels_pickuparr = self.choose_samples(mat_four_mod, mat_four, 1000)  
            self.training_sample(4, 1000, mat_four ,labels_pickuparr)

            self.labels_pickuparr = self.choose_samples(mat_five_mod, mat_five, 1000)  
            self.training_sample(5, 1000, mat_five ,labels_pickuparr)

            self.labels_pickuparr = self.choose_samples(mat_six_mod, mat_six, 1000)  
            self.training_sample(6, 1000, mat_six ,labels_pickuparr)

            self.labels_pickuparr = self.choose_samples(mat_seven_mod, mat_seven, 1000)  
            self.training_sample(7, 1000, mat_seven ,labels_pickuparr)

            self.labels_pickuparr = self.choose_samples(mat_eight_mod, mat_eight, 1000)  
            self.training_sample(8, 1000, mat_eight ,labels_pickuparr)

            self.labels_pickuparr = self.choose_samples(mat_nine_mod, mat_nine, 1000)  
            self.training_sample(9, 1000, mat_nine ,labels_pickuparr)
        
        if (self.option == 'kmeans'):
                samples_zero = np.zeros(shape=(N/10,784))
                samples_zero = choose_sample_kmeans(mat_zero, N/10)
                for i in range (0,99):
                    self.training_samples[i] = self.samples_zero[i];
                    self.label_samples[i] = 0;
                    
                samples_one = np.zeros(shape=(N/10,784))
                samples_one = choose_sample_kmeans(mat_one, N/10)
                for i in range (100,99):
                    self.training_samples[i] = samples_one[i - 100];
                    self.label_samples[i] = 1;
                samples_two = np.zeros(shape=(N/10,784))
                samples_two = self.choose_sample_kmeans(mat_two, N/10)
                for i in range (200,299):
                    self.training_samples[i] = samples_two[i - 200];
                    self.label_samples[i] = 2;
                samples_three = np.zeros(shape=(N/10,784))
                samples_three = self.choose_sample_kmeans(mat_three, N/10)
                for i in range (300,399):
                    self.training_samples[i] = samples_three[i - 300];
                    self.label_samples[i] = 3;
                samples_four = np.zeros(shape=(N/10,784))
                samples_four = self.choose_sample_kmeans(mat_four, N/10)
                for i in range (400,499):
                    self.training_samples[i] = samples_four[i - 400];
                    self.label_samples[i] = 4;
                samples_five = np.zeros(shape=(N/10,784))
                samples_five = self.choose_sample_kmeans(mat_five, N/10)
                for i in range (500,599):
                    self.training_samples[i] = samples_five[i - 500];
                    self.label_samples[i] = 5;
                samples_six = np.zeros(shape=(N/10,784))
                samples_six = self.choose_sample_kmeans(mat_six, N/10)
                for i in range (600,699):
                    self.training_samples[i] = samples_six[i - 600];
                    self.label_samples[i] = 6;
                samples_seven = np.zeros(shape=(N/10,784))
                samples_seven = self.choose_sample_kmeans(mat_seven, N/10)
                for i in range (700,799):
                    self.training_samples[i] = samples_seven[i - 700];
                    self.label_samples[i] = 7;
                samples_eight = np.zeros(shape=(N/10,784))
                samples_eight = self.choose_sample_kmeans(mat_eight, N/10)
                for i in range (800,899):
                    self.training_samples[i] = samples_eight[i - 800];
                    self.label_samples[i] = 8;
                samples_nine = np.zeros(shape=(N/10,784))
                samples_nine = self.choose_sample_kmeans(mat_nine, N/10)
                for i in range (900,999):
                    self.training_samples[i] = samples_nine[i - 900];
                    self.label_samples[i] = 9;
    
 
    def get_sel_error(self, X_test, Y_test):
        
        knn_sel = KNeighborsClassifier(n_neighbors=1)
        knn_sel.fit(training_samples, label_training_samples)
        y_sel_predict = knn_sel.predict(X_test)
        return np.sum(y_sel_predict == Y_test)

        
if __name__ == '__main__':
    X_train, Y_train = loadlocal_mnist(
        images_path='/Users/apurb/Documents/train-images-idx3-ubyte/train-images.idx3-ubyte', 
        labels_path='/Users/apurb/Documents/train-labels-idx1-ubyte/train-labels.idx1-ubyte')

    X_test, Y_test = loadlocal_mnist(
        images_path='/Users/apurb/Documents/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte', 
        labels_path='/Users/apurb/Documents/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')


    randomKNN = randomSamples(1000);
    print("The error for selecting 1000 samples at random is", randomKNN.get_error(X_train, Y_train, X_test, Y_test));

    incrSort = Prototype(1000,'IncrSort')
    incrSort.data_evaluation(X_train, Y_train);
    print("The error for selecting 1000 samples at random is", incrSort.get_sel_error(X_test, Y_test));

    kmeansSort = Prototype(1000,'kmeans')
    kmeansSort.data_evaluation(X_train, Y_train);
    print("The error for selecting 1000 samples at random is", kmeansSort.get_sel_error(X_test, Y_test));



           


