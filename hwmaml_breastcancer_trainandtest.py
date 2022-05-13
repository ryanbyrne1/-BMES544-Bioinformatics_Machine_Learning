from sklearn import svm

def hwmaml_breastcancer_trainandtest(Xtrain,Ttrain, Xtest,Ttest):
    #By Ryan Byrne
    #Inputs:
    #Xtrain - training data
    #Ttrain - training class labels
    #Xtest - testing data
    #Ttest - testing class labels
        
    #Create svm variable
    clf = svm.SVC()
    
    #Fit data to SVM
    clf.fit(Xtrain, Ttrain)
    
    #Variable to count number of errors
    numerror = 0
    
    #Number of samples in test data
    numTest = len(Ttest)
    
    #Predict label for each test data
    for i in range(numTest):
       
        #Select data to test
        TestRow = [Xtest[i, :]]
        
        #predict class
        result = clf.predict(TestRow)
        
        #If predicted class does not match test label increase number of errors
        if (result[0] != Ttest[i]):
            numerror += 1
    
    #Return number of errors
    return numerror