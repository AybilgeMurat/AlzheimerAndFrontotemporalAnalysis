
import numpy as np
import pandas as pd
import mne
from config import DATA_PATH, PROCESSED_DATA_PATH
import pickle
from scipy.integrate import simpson


#test için ayırdığımız sinyaller
TEST = [4, 6, 8, 20, 33, 49, 53, 63, 66, 71, 72, 82]

CHANNELS = {0: 'Fp1', 1: 'Fp2', 2: 'F3', 3: 'F4', 4: 'C3', 5: 'C4', 6: 'P3', 7: 'P4', 8: 'O1', 9: 'O2', 
 10: 'F7', 11: 'F8', 12: 'T3', 13: 'T4', 14: 'T5', 15: 'T6', 16: 'Fz', 17: 'Cz', 18: 'Pz'}

channels_dict = {'Fp1' : 0, 'Fp2': 1 , 'F3': 2, 'F4' : 3, 'C3': 4, 'C4': 5, 'P3': 6, 'P4' : 7, 'O1' : 8, 'O2':9, 
'F7':10 , 'F8': 11, 'T3' : 12, 'T4' :13, 'T5' : 14, 'T6' : 15, 'Fz': 16, 'Cz': 17,'Pz': 18}


def load_subject(subject_id,path=DATA_PATH):
    return mne.io.read_raw_eeglab(path + '/derivatives/sub-' + str(subject_id).zfill(3)
                                    + '/eeg/sub-' + str(subject_id).zfill(3) + '_task-eyesclosed_eeg.set', preload = True,verbose='CRITICAL')

def subject_psd(raw,seg_length,fmin=0.5,fmax=45):
    """
    Welch yöntemini kullanarak belirli bir denek için her EEG kanalının psd'sini hesaplar.

    Parametreler
    ----------
    raw : 
        RawEEGLAB data yükler.
    seg_length : float
        Her Welch segmentinin saniye cinsinden uzunluğu. (Frekans çözünürlüğünü belirler). 
    fmin : float, optional
        Alt frekans, varsayılan olarak 0,5.
    fmax : int, optional
        Üst frekans ,varsayılan olarak 45.

    Geri döndürdüğü değer
    -------
    psd :
        mne Spectrum nesnesinde saklanan her bir EEG kanalının psd'si.
    """        
    return raw.compute_psd(method='welch', fmin=fmin,fmax=fmax,n_fft=int(seg_length*raw.info['sfreq']),verbose=False)

def epochs_psd(raw,duration,overlap,seg_length,fmin=0.5,fmax=45,tmin=None,tmax=None):
    """
    EEG kayıt verilerini belirli bir denek için örtüşen epoklara böler ve Welch yöntemini kullanarak her EEG kanalının psd'sini hesaplar.

    Parametreler
    ----------

    duration : float
        Her epoğun saniye cinsinden süresi.
    overlap : float
        epoklar arasındaki örtüşme, saniye cinsinden.
    seg_length : float
        Her Welch segmentinin saniye cinsinden uzunluğu. (Frekans çözünürlüğünü belirler). 
    fmin : float, optional
        Alt frekans, varsayılan olarak 0,5.
    fmax : int, optional
        Üst frekans ,varsayılan olarak 45.
    tmin : float, optional
        Saniye cinsinden dahil edilecek ilk zaman.
    tmax : float, optional
        Saniye cinsinden dahil edilecek son zaman.

    Geri döndürdüğü değer
    -------
    epochs_psds :
        mne EpochsSpectrum nesnesinde saklanan her bir epok için her bir EEG kanalının psd'si.
    """    
    epochs = mne.make_fixed_length_epochs(raw,duration=duration,preload=True,overlap=overlap,verbose=False)
    return epochs.compute_psd(method='welch', fmin=fmin,fmax=fmax,tmin=tmin,tmax=tmax,n_fft=int(seg_length*raw.info['sfreq']),verbose=False)

def load_data(duration,overlap,seg_length,fmin=0.5,fmax=45,classes={'A':1,'F':2,'C':0},path=DATA_PATH):
    """
    Belirtilen sınıflardan tüm denekleri yükler, EEG kayıtlarını epoklara böler, Welch yöntemini kullanarak her EEG kanalının psd'sini hesaplar 
    ve ardından bu psd'leri atanan sınıf etiketleriyle birlikte döndürür.

    Parametreler
    ----------
    duration : float
        Her epoğun saniye cinsinden süresi.
    overlap : float
        epoklar arasındaki örtüşme, saniye cinsinden.
    seg_length : float
        Her Welch segmentinin saniye cinsinden uzunluğu. (Frekans çözünürlüğünü belirler). 
    fmin : float, optional
        Alt frekans, varsayılan olarak 0,5.
    fmax : int, optional
        Üst frekans ,varsayılan olarak 45.
    classes : dict, optional
        Anahtarları dahil edilecek sınıflar ve değerleri sayısal etiketler olan sözlük. 
        Varsayılan {'A':1,'F':2,'C':0}.
    path : str, optional
        config.py de bulunan dosya yolu

    Geri döndürdüğü değer
    -------
    subject_data : list[ndarray]
        Her biri (num_epochs) x (num_channels) x len(freqs) şeklinde olan ve her dizinin bir deneğe karşılık geldiği dizilerin listesi.
    freqs : ndarray
        Psd'lerin ölçüldüğü frekansların dizisi.
    targets : ndarray
        subject_data içindeki konular için sayısal sınıf etiketleri. 
    """    
    subject_table = pd.read_csv(path + '/participants.tsv',sep='\t')
    target_labels = subject_table['Group']
    subject_data = []
    targets = []
    for subject_id in range(1,len(target_labels)+1):
        if target_labels.iloc[subject_id-1] not in classes:
            continue
        raw = load_subject(subject_id,path=path)
        epochs_psds = epochs_psd(raw,duration=duration,overlap=overlap,seg_length=seg_length,fmin=fmin,fmax=fmax)
        epochs_psds_array, freqs = epochs_psds.get_data(return_freqs=True)
        subject_data.append(epochs_psds_array)
        targets.append(classes[target_labels.iloc[subject_id-1]])
    return subject_data, freqs, np.array(targets)

def save_psds(subject_data,freqs,targets,filename,path=PROCESSED_DATA_PATH):
    """load_data tarafından oluşturulan psd verilerini toplar ve veri işleme klasörüne kaydeder"""
    with open(path + '/' + filename,'wb') as file:
        pickle.dump({'subject_data':subject_data,'freqs':freqs,'targets':targets},file)

def load_psds(filename,path=PROCESSED_DATA_PATH):
    """Veri işleme klasöründen seçili psd verilerini yükler"""
    with open(path + '/' + filename,'rb') as file:
        psds = pickle.load(file)
    return psds['subject_data'], psds['freqs'], psds['targets']

def freq_ind(freqs,freq_bands):
    """freq_bands içindeki frekanslara karşılık gelen freqs dizisindeki indislerin listesini döndürür"""
    indices = []
    for i in range(len(freq_bands)):
        indices.append(np.argmin(np.abs(freqs-freq_bands[i])))
    return indices

def absolute_band_power(psds,freqs,freq_bands,endpoints=freq_ind):
    """
    Psds dizisindeki her satırın her EEG kanalının her frekans bandındaki mutlak bant gücünü hesaplar.

    Parametreler
    ----------
    psds : ndarray
        (num_rows) x (num_channels) x len(freqs) şeklinde psd dizisi
    freqs : ndarray
        1 boyutlu frekans dizisi.
    freq_bands : array_like
        Frekans bantlarının sınırlarını tanımlayan frekansların listesi.
    endpoints : 
        Freq_bands ile frekansları eşleştirmek için kullanılan fonksiyon.

    Geri döndürdüğü değer
    -------
    abps: ndarray
        (num_rows) x (num_channels) x (len(freq_bands)-1) şeklinde mutlak bant gücü değerleri dizisi
    """    
    indices = endpoints(freqs,freq_bands)
    absolute_bands_list = []
    for i in range(len(indices)-1):
        absolute_bands_list.append(simpson(psds[...,indices[i]:indices[i+1]+1],freqs[indices[i]:indices[i+1]+1],axis=-1))
    return np.transpose(np.array(absolute_bands_list),(1,2,0))

def relative_band_power(psds,freqs,freq_bands,endpoints=freq_ind):
    """
    Psds dizisindeki her satırın her EEG kanalının her frekans bandındaki göreli bant gücünü hesaplar.

    Parametreler
    ----------
    psds : ndarray
        (num_rows) x (num_channels) x len(freqs) şeklinde psd dizisi
    freqs : ndarray
        1 boyutlu frekans dizisi.
    freq_bands : array_like
        Frekans bantlarının sınırlarını tanımlayan frekansların listesi.
    endpoints : 
        Freq_bands ile frekansları eşleştirmek için kullanılan fonksiyon.

    Geri döndürdüğü değer
    -------
    rbps: ndarray
        (num_rows) x (num_channels) x (len(freq_bands)-1) şeklinde göreli bant gücü değerleri dizisi.
    """    
    indices = endpoints(freqs,freq_bands)
    total_power = np.expand_dims(simpson(psds[...,indices[0]:indices[-1]+1],freqs[indices[0]:indices[-1]+1],axis=-1),axis=-1)
    return np.divide(absolute_band_power(psds,freqs,freq_bands,endpoints=endpoints),total_power)

def align_test_labels(test=TEST,classes=['A','C','F']):
    """
    Test seti etiketlerini sınıflandırıcının eğitim aldığı sınıflara göre hizalar.
    """
    if classes == ['A','C','F']:
        return [subject_id-1 for subject_id in test]
    if classes == ['A','C']:
        return [subject_id-1 for subject_id in test if subject_id <= 65]
    if classes == ['C','F']:
        return [subject_id-37 for subject_id in test if subject_id >= 37]
    if classes == ['A','F']:
        return ([subject_id-1 for subject_id in test if subject_id <= 36] 
                + [subject_id-30 for subject_id in test if subject_id >= 66])

def remove_class(features,targets,class_):
    """
    Yüklenen verilerden bir sınıfı kaldırır. Üç sınıfın tümü yüklendiğinde ancak modelleme için yalnızca ikisi kullanıldığında kullanılır.
    """
    if class_ == 'F':
        return features[:65],targets[:65]
    if class_ == 'A':
        return features[36:],targets[36:]
    if class_ == 'C':
        return features[:36]+features[65:], np.append(targets[:36], targets[65:])

def remove_test(features,targets,test):
    """
    Test konularını özellik dizileri listesinden kaldırır. Bu fonksiyonu kullanmadan önce, 
    söz konusu sınıflandırma problemine bağlı olarak etiketler önce align_test_labels ile hizalanmalıdır.
    """
    features_train = [features[i] for i in range(len(features)) if i not in test]
    target_train = [targets[i] for i in range(len(targets)) if i not in test]
    return features_train, target_train

def select_test(features,targets,test):
    """
    Özellik dizileri listesinden test deneklerini seçer. Bu fonksiyonu kullanmadan önce, 
    söz konusu sınıflandırma problemine bağlı olarak etiketler önce align_test_labels ile hizalanmalıdır.
    """
    features_train = [features[i] for i in range(len(features)) if i in test]
    target_train = [targets[i] for i in range(len(targets)) if i in test]
    return features_train, target_train

def remove_channel(input_rbp, channels_to_remove):
    """
    tüm 19 EEG kanalını içeren özellik dizilerinin giriş listesinden bir EEG kanalları listesini kaldırır.
    
    Parametreler
    ----------
    input_rbp : list[ndarray]
            Tüm 19 kanalı içeren her bir deneğe karşılık gelen özellik dizilerinin listesi. 
            Kanal, diziler için sondan ikinci dizi indeksine karşılık gelir.
    channels_to_remove: list
            Kaldırılacak EEG kanallarının listesi.
    
    Geri döndürdüğü değer
    -------
    updated_rbp: list[ndarray]
            Kaldırılan kanalları içermeyen her bir özneye karşılık gelen özellik dizilerinin listesi.
    """
    
    updated_rbp = []
    all_channels = np.arange(0,19)
    channels_removed_ind = [channels_dict[ch] for ch in channels_to_remove]
    resulting_channels = np.delete(all_channels, channels_removed_ind)
    updated_rbp = [rbp[:, resulting_channels, :] for rbp in input_rbp]
    return updated_rbp



def train_prep(features,targets,exclude=None,flatten_final=True):
    """
    İlk (epochs) boyut boyunca birleştirerek eğitim için hedeflerde karşılık gelen etiketlere sahip özellik dizilerinin bir listesini hazırlar. 
    İsteğe bağlı olarak tek denekli çapraz doğrulama için bir deneği hariç tutar

    Parametreler
    ----------
    features : list[ndarray]
       Her bir deneğe karşılık gelen özellik dizilerinin listesi.
    targets : ndarray
        Her özellik dizisi için sayısal sınıf etiketleri dizisi.
    exclude : int, optional
        İndeksi 'exclude' olan öznenin özellik dizisini çıktıdan hariç tutar. 
        Varsayılan değer olan 'None' tüm özneleri içeride tutar.
    flatten_final : bool, optional
        Varsayılan değer True, ilk boyut hariç çıktı özellik dizisinin tüm boyutlarını düzleştirir. 
        Bunu False olarak ayarlamak, özellikleri num_channels*num_frequency_bands şeklinde 2 boyutlu bir dizi olarak korur

    Geri döndürdüğü değer
    -------
    features_array : ndarray
        Her satırı bir eğitim örneğine karşılık gelen özellik dizisi.
    targets_array : ndarray
        features_array içindeki eğitim örnekleri için 1 boyutlu etiket dizisi.
    """
    total_subjects = len(targets)
    target_list = []
    for i in range(total_subjects):
        num_epochs = features[i].shape[0]
        target_list.append(targets[i]*np.ones(num_epochs))
    if exclude==None: 
        features_array = np.concatenate(features)
        targets_array = np.concatenate(target_list)
    else:
        features_array = np.concatenate(features[:exclude] + features[exclude+1:])
        targets_array  = np.concatenate(target_list[:exclude] + target_list[exclude+1:])
    if flatten_final:
        features_array = features_array.reshape((features_array.shape[0],-1))
    return features_array, targets_array

def accuracy(confusion):
    return np.trace(confusion)/np.sum(confusion)

def sensitivity(confusion):
    return confusion[1,1]/(confusion[1,1]+confusion[1,0])

def specificity(confusion):
    return confusion[0,0]/(confusion[0,0]+confusion[0,1])

def precision(confusion):
    return confusion[1,1]/(confusion[1,1]+confusion[0,1])
    
def f1(confusion):
    return 2*(precision(confusion)*sensitivity(confusion))/(precision(confusion)+sensitivity(confusion))




