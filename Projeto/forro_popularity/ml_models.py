# regression   models
from sklearn.neighbors import KNeighborsRegressor #KNeighborsRegressor
from sklearn.linear_model import LinearRegression #LinearRegression
from sklearn.tree import DecisionTreeRegressor #Decision Tree Regression
from sklearn.svm import SVR #SVR
from sklearn.neural_network import MLPRegressor #Multi-layer Perceptron Regression

from sklearn.ensemble import RandomForestRegressor #Random Forest Regression
from sklearn.ensemble import GradientBoostingRegressor #Gradient Boosting Regression
from sklearn.ensemble import AdaBoostRegressor #AdaBoost Regression


# classifier models
from sklearn.linear_model import LogisticRegression #Logistc Regression
from sklearn.neighbors import KNeighborsClassifier #KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #Decision Tree Classifier
from sklearn.svm import SVC #SVC
from sklearn.neural_network import MLPClassifier #Multi-layer Perceptron Classifier

from sklearn.ensemble import RandomForestClassifier #Random Forest Classifier
from sklearn.ensemble import GradientBoostingClassifier #Gradient Boosting Classifier
from sklearn.ensemble import AdaBoostClassifier #AdaBoost Classifier


def ml_models (model="DT",names=False,parameters=[],task="C"):
    #models = ["KNN","LM","DT","SVM", "RF", "MLP", "GB", "ADA"]
    #types = ["C","R"]
    if names == True:
        print("KNN: KNeighborsRegressor\nLM: LinearRegression\nDT: DecisionTreeRegressor\nRF: RandomForestRegressor",
          "\nGB: GradientBoostingRegressor\nSVM: SVR\nADA: AdaBoostRegressor\nMLP: MLPRegressor")
    
    if model == "KNN":
        if task == "C":
            return KNeighborsClassifier(n_neighbors=10, weights="distance",random_state=0)
        elif task == "R":
            return KNeighborsRegressor(n_neighbors=10, weights="distance",random_state=0)
        
    if model == "LM":
        if task == "C":
            return LogisticRegression(multi_class="multinomial", max_iter=10,random_state=0)
        elif task == "R":
            return LinearRegression()
    
    if model == "DT":
        if task == "C":
            return DecisionTreeClassifier(max_depth=parameters[0],min_samples_leaf=parameters[1],criterion=parameters[2],random_state=0)
        elif task == "R":
            return DecisionTreeRegressor(max_depth=parameters[0],min_samples_leaf=parameters[1],criterion=parameters[2],random_state=0)
        
    if model=="MLP":
        if task == "C":
            return MLPClassifier(hidden_layer_sizes=(10,10), max_iter=10,random_state=0)
        elif task == "R":
            return MLPRegressor(hidden_layer_sizes=(10,10), max_iter=10,random_state=0)
    
    if model == "ADA":
        if task == "C":
            return AdaBoostClassifier(n_estimators=10,random_state=0)
        elif task == "R":
            return AdaBoostRegressor(n_estimators=10,random_state=0)
        
    if model == "RF":
        if task == "C":
            return RandomForestClassifier(n_estimators=100,max_depth=10,random_state=0)
        elif task == "R":
            return RandomForestRegressor(n_estimators=100,max_depth=10,min_samples_leaf=10,random_state=0)
        
    if model == "GBRT":
        if task == "C":
            return GradientBoostingClassifier(n_estimators=100,max_depth=10,random_state=0)
        elif task == "R":
            return GradientBoostingRegressor(n_estimators=100,max_depth=10,min_samples_leaf=10,random_state=0)

    return None
