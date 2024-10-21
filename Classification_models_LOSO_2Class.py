from model_functions import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
#import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

'''
Bu dosya, farklı sınıflandırma algoritmaları kullanarak tek denekli çapraz doğrulama gerçekleştirmek için gerekli fonksiyonları tanımlar. 
Verileri eğitime hazır hale getirmek için model_functions dosyasındaki fonksiyonları kullanır.
'''


def log_reg_cross(rbps,targets, PCA_components = 0,reg_parameter=1.0,max_iter=500):
    
    '''
    Lojistik regresyon sınıflandırıcısı kullanarak rbps ve hedeflerdeki özellikler üzerinde eğitim gerçekleştirir ve 
    bir konuyu dışarıda bırakma yöntemini kullanarak çapraz doğrulama yapar.
    
    Parametreler
    ----------
    rbps : list[ndarray]
        Her bir deneğe karşılık gelen özellik dizilerinin listesi.
    targets : ndarray
        Her özellik dizisi için sayısal sınıf etiketleri dizisi.
    n_neighbors: int
        kullanılacak en yakın komşu sayısı
    PCA_components: int
       Temel bileşenler analizi için tutulacak bileşen sayısı. Varsayılan değer PCA yapmadığımız durum olan 0'dır.
    reg_parameter: float
        Düzenlemenin gücü ile ters orantılı parametre
    max_iter: int
        Çözücülerin yakınsaması için geçen iterasyon sayısı.
           
    Geri döndürdüğü değer
    -------
    train_metrics_dict : dictionary
        Eğitim verileri üzerinde doğruluk, duyarlılık, özgüllük ve f1 puanlarını içeren sözlük.
    test_metrics_dict : dictionary
        Doğrulama verileri üzerinde doğruluk, duyarlılık, özgüllük ve f1 puanlarını içeren sözlük.
    
    '''
    confusion_matrices_train = []
    confusion_matrices_test = []
    labels = np.unique(targets)
    for i in range(len(targets)):
        train_X, train_y = train_prep(rbps,targets,exclude=i,flatten_final=True)
        test_X = rbps[i].reshape(rbps[i].shape[0],-1)
        test_y = targets[i]*np.ones(rbps[i].shape[0])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        
        if PCA_components != 0:
            pca = PCA(n_components = PCA_components)
            train_X = pca.fit_transform(train_X, y = None)
            test_X = pca.transform(test_X)
        
        log_reg = LogisticRegression(max_iter = max_iter,C=reg_parameter)
        log_reg.fit(train_X, train_y)
        
        confusion_matrices_train += [confusion_matrix(train_y, log_reg.predict(train_X),labels=labels)]
        confusion_matrices_test += [confusion_matrix(test_y, log_reg.predict(test_X),labels=labels)]
    
    confusion_matrices_train = np.array(confusion_matrices_train)
    confusion_matrices_test = np.array(confusion_matrices_test)
    total_confusion_train = np.sum(confusion_matrices_train, axis= 0)
    total_confusion_test = np.sum(confusion_matrices_test, axis= 0)
    
    train_metrics_dict = {'acc':accuracy(total_confusion_train), 'sens':sensitivity(total_confusion_train), 
                            'spec':specificity(total_confusion_train), 'f1':f1(total_confusion_train)}
    test_metrics_dict = {'acc':accuracy(total_confusion_test), 'sens':sensitivity(total_confusion_test), 
                            'spec':specificity(total_confusion_test), 'f1':f1(total_confusion_test)}
    
    
    return train_metrics_dict, test_metrics_dict








def kNN_cross(rbps,targets,n_neighbors, PCA_components = 0):
    
    '''
    K en yakın komşu sınıflandırması kullanarak rbps ve hedeflerdeki özellikler üzerinde eğitim gerçekleştirir ve 
    bir konuyu dışarıda bırakma yöntemini kullanarak çapraz doğrulama yapar.
    
    Parametreler
    ----------
    rbps : list[ndarray]
        Her bir deneğe karşılık gelen özellik dizilerinin listesi.
    targets : ndarray
        Her özellik dizisi için sayısal sınıf etiketleri dizisi.
    n_neighbors: int
        kullanılacak en yakın komşu sayısı
    PCA_components: int
       Temel bileşenler analizi için tutulacak bileşen sayısı. Varsayılan değer PCA yapmadığımız durum olan 0'dır.
    reg_parameter: float
        Düzenlemenin gücü ile ters orantılı parametre
    max_iter: int
        Çözücülerin yakınsaması için geçen iterasyon sayısı.
           
    Geri döndürdüğü değer
    -------
    train_metrics_dict : dictionary
        Eğitim verileri üzerinde doğruluk, duyarlılık, özgüllük ve f1 puanlarını içeren sözlük.
    test_metrics_dict : dictionary
        Doğrulama verileri üzerinde doğruluk, duyarlılık, özgüllük ve f1 puanlarını içeren sözlük.
    '''

    confusion_matrices_train = []
    confusion_matrices_test = []
    labels = np.unique(targets)
    for i in range(len(targets)):
        train_X, train_y = train_prep(rbps,targets,exclude=i,flatten_final=True)
        test_X = rbps[i].reshape(rbps[i].shape[0],-1)
        test_y = targets[i]*np.ones(rbps[i].shape[0])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        
        if PCA_components != 0:
            pca = PCA(n_components = PCA_components)
            train_X = pca.fit_transform(train_X, y = None)
            test_X = pca.transform(test_X)
        
        ThreeNN = KNeighborsClassifier(n_neighbors=n_neighbors)
        ThreeNN.fit(train_X, train_y)
        
        confusion_matrices_train += [confusion_matrix(train_y, ThreeNN.predict(train_X),labels=labels)]
        confusion_matrices_test += [confusion_matrix(test_y,ThreeNN.predict(test_X),labels=labels)]
    
    confusion_matrices_train = np.array(confusion_matrices_train)
    confusion_matrices_test = np.array(confusion_matrices_test)
    total_confusion_train = np.sum(confusion_matrices_train, axis= 0)
    total_confusion_test = np.sum(confusion_matrices_test, axis= 0)
    
    train_metrics_dict = {'acc':accuracy(total_confusion_train), 'sens':sensitivity(total_confusion_train), 
                            'spec':specificity(total_confusion_train), 'f1':f1(total_confusion_train)}
    test_metrics_dict = {'acc':accuracy(total_confusion_test), 'sens':sensitivity(total_confusion_test), 
                            'spec':specificity(total_confusion_test), 'f1':f1(total_confusion_test)}
    
    
    return train_metrics_dict, test_metrics_dict




def RF_cross(rbps, targets, n_estimators = 100, min_samples_split = 16, PCA_components = 0):
    
    '''
    Rastgele orman sınıflandırması kullanarak rbps ve hedeflerdeki özellikler üzerinde eğitim gerçekleştirir ve 
    bir konuyu dışarıda bırakma yöntemini kullanarak çapraz doğrulama yapar.
    
    Parameters
    ----------
    rbps : list[ndarray]
            Her bir deneğe karşılık gelen özellik dizilerinin listesi.
    targets : ndarray
            Her özellik dizisi için sayısal sınıf etiketleri dizisi.
    n_estimators: int
            ormandaki ağaç sayısı
    min_samples_split: int or floar
           Rastgele orman sınıflandırıcısında bir iç düğümü bölmek için gereken minimum örnek sayısı. int ise, 
           min_samples_split'i minimum sayı olarak kabul edin. Float ise, min_samples_split bir kesirdir ve 
           ceil(min_samples_split * n_samples) her bölme için minimum örnek sayısıdır. 
           Varsayılan değer, çapraz doğrulama doğruluğunu en üst düzeye çıkardığı tespit edilen 16'dır.  
    PCA_components: int
           Temel bileşenler analizi için tutulacak bileşen sayısı. Varsayılan değer PCA yapmadığımız durum olan 0'dır.
    Geri döndürdüğü değer
    -------
    train_metrics_dict : dictionary
    test_metrics_dict : dictionary
    
    '''
    confusion_matrices_train = []
    confusion_matrices_test = []
    labels = np.unique(targets)
    for i in range(len(targets)):
        train_X, train_y = train_prep(rbps,targets,exclude=i,flatten_final=True)
        test_X = rbps[i].reshape(rbps[i].shape[0],-1)
        test_y = targets[i]*np.ones(rbps[i].shape[0])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        
        if PCA_components != 0:
            pca = PCA(n_components = PCA_components)
            train_X = pca.fit_transform(train_X, y = None)
            test_X = pca.transform(test_X)
       
        
        RF = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split)
        RF.fit(train_X, train_y)
    
        
        confusion_matrices_train += [confusion_matrix(train_y, RF.predict(train_X),labels=labels)]
        confusion_matrices_test += [confusion_matrix(test_y,RF.predict(test_X),labels=labels)]
    
    confusion_matrices_train = np.array(confusion_matrices_train)
    confusion_matrices_test = np.array(confusion_matrices_test)
    total_confusion_train = np.sum(confusion_matrices_train, axis= 0)
    total_confusion_test = np.sum(confusion_matrices_test, axis= 0)
    
    train_metrics_dict = {'acc':accuracy(total_confusion_train), 'sens':sensitivity(total_confusion_train), 
                            'spec':specificity(total_confusion_train), 'f1':f1(total_confusion_train)}
    test_metrics_dict = {'acc':accuracy(total_confusion_test), 'sens':sensitivity(total_confusion_test), 
                            'spec':specificity(total_confusion_test), 'f1':f1(total_confusion_test)}
    
    
    return train_metrics_dict, test_metrics_dict
    
def svm_cross(rbps,targets,kernel,reg_parameter=1.0, degree=2, PCA_components = 0):
    """
    Sklearn'in svm C-Destek vektör sınıflandırmasını kullanarak rbps ve hedeflerdeki özellikler üzerinde eğitim gerçekleştirir ve 
    bir konuyu dışarıda bırakma yöntemini kullanarak çapraz doğrulama yapar.

    Parameters
    ----------
    rbps : list[ndarray]
            Her bir deneğe karşılık gelen özellik dizilerinin listesi.
    targets : ndarray
            Her özellik dizisi için sayısal sınıf etiketleri dizisi.
    kernel : 
        SVM için kullanılan çekirdek. Seçenekler için sklearn belgelerine bakın.
    reg_parameter : float, optional
        SVM'deki düzenlileştirmenin gücüyle ters orantılı parametre, varsayılan olarak 1.0
    degree : int, optional
        kernel = 'poly' olduğunda kernel için kullanılan polinom derecesi, varsayılan olarak 2. Çekirdek != 'poly' olduğunda göz ardı edilir.
    PCA_components: int
           Temel bileşenler analizi için tutulacak bileşen sayısı. Varsayılan değer PCA yapmadığımız durum olan 0'dır.

    Geri döndürdüğü değer
    -------
    train_metrics_dict : dictionary
    test_metrics_dict : dictionary
    """    
    confusion_matrices_train = []
    confusion_matrices_test = []
    labels = np.unique(targets)
    for i in range(len(targets)):
        train_X, train_y = train_prep(rbps,targets,exclude=i,flatten_final=True)
        test_X = rbps[i].reshape(rbps[i].shape[0],-1)
        test_y = targets[i]*np.ones(rbps[i].shape[0])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        
        if PCA_components != 0:
            pca = PCA(n_components = PCA_components)
            train_X = pca.fit_transform(train_X, y = None)
            test_X = pca.transform(test_X)
        
        
        svc = SVC(C = reg_parameter, kernel=kernel, degree=degree)
        svc.fit(train_X, train_y)

        
        confusion_matrices_train += [confusion_matrix(train_y, svc.predict(train_X),labels=labels)]
        confusion_matrices_test += [confusion_matrix(test_y,svc.predict(test_X),labels=labels)]

    confusion_matrices_train = np.array(confusion_matrices_train)
    confusion_matrices_test = np.array(confusion_matrices_test)
    total_confusion_train = np.sum(confusion_matrices_train, axis= 0)
    total_confusion_test = np.sum(confusion_matrices_test, axis= 0)

    train_metrics_dict = {'acc':accuracy(total_confusion_train), 'sens':sensitivity(total_confusion_train), 
                            'spec':specificity(total_confusion_train), 'f1':f1(total_confusion_train)}
    test_metrics_dict = {'acc':accuracy(total_confusion_test), 'sens':sensitivity(total_confusion_test), 
                            'spec':specificity(total_confusion_test), 'f1':f1(total_confusion_test)}


    return train_metrics_dict, test_metrics_dict

def mlp_cross(rbps,targets, PCA_components = 0, alpha = 0.0001, learning_rate_init = 0.001, hidden_layer_sizes = (3,1), max_iter = 200, solver = 'adam', random_state = None):
    
    '''
    Çok katmanlı perceptron sınıflandırması kullanarak rbps ve hedeflerdeki özellikler üzerinde eğitim gerçekleştirir ve 
    bir konuyu dışarıda bırakma yöntemini kullanarak çapraz doğrulama gerçekleştirir.
    
    Parameters
    ----------
    rbps : list[ndarray]
        Her bir deneğe karşılık gelen özellik dizilerinin listesi.
    targets : ndarray
        Her özellik dizisi için sayısal sınıf etiketleri dizisi.
    PCA_components: int
        Temel bileşenler analizi için tutulacak bileşen sayısı. Varsayılan değer PCA yapmadığımız durum olan 0'dır.
    alpha: float
        L2 düzenlileştirme teriminin gücünü tanımlayan parametre.
    max_iter: int
        Çözücü için maksimum iterasyon sayısı.
    learning_rate_init: float
        Kullanılan ilk öğrenme oranı.
    hidden_layer_sizes: 
        Her gizli katmandaki nöron sayısını tanımlayan çift.
    solver: {‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
        Ağırlık optimizasyonu için kullanılacak çözücü.
    random_state: int, default=None
        Eğitim sürecinin çeşitli bölümleri için rastgele sayı üretimini belirler.
           
    
    '''
    confusion_matrices_train = []
    confusion_matrices_test = []
    labels = np.unique(targets)
    for i in range(len(targets)):
        train_X, train_y = train_prep(rbps,targets,exclude=i,flatten_final=True)
        test_X = rbps[i].reshape(rbps[i].shape[0],-1)
        test_y = targets[i]*np.ones(rbps[i].shape[0])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        
        if PCA_components != 0:
            pca = PCA(n_components = PCA_components)
            train_X = pca.fit_transform(train_X, y = None)
            test_X = pca.transform(test_X)
        
        mlp = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, max_iter = max_iter,alpha = alpha, learning_rate_init = learning_rate_init,random_state=random_state)
        mlp.fit(train_X, train_y)
        
        confusion_matrices_train += [confusion_matrix(train_y, mlp.predict(train_X),labels=labels)]
        confusion_matrices_test += [confusion_matrix(test_y, mlp.predict(test_X),labels=labels)]
    
    confusion_matrices_train = np.array(confusion_matrices_train)
    confusion_matrices_test = np.array(confusion_matrices_test)
    total_confusion_train = np.sum(confusion_matrices_train, axis= 0)
    total_confusion_test = np.sum(confusion_matrices_test, axis= 0)
    
    train_metrics_dict = {'acc':accuracy(total_confusion_train), 'sens':sensitivity(total_confusion_train), 
                            'spec':specificity(total_confusion_train), 'f1':f1(total_confusion_train)}
    test_metrics_dict = {'acc':accuracy(total_confusion_test), 'sens':sensitivity(total_confusion_test), 
                            'spec':specificity(total_confusion_test), 'f1':f1(total_confusion_test)}
    
    
    return train_metrics_dict, test_metrics_dict
    
    
   

  


