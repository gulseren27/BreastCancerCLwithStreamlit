
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

class App:
    def __init__(self):
        self.dataset_name = None
        self.classifier_name = None
        self.Init_Streamlit_Page()

        self.params = dict()
        self.clf = None
        self.X, self.y = None, None
    def run(self):
        file_path="C:/Users/Gulseren/Documents/GitHub/ml_methods_streamlit/data.csv"
        self.get_dataset(file_path)
        self.preprocess_data()
        self.add_parameter_ui()
        self.generate()
    def Init_Streamlit_Page(self):
        st.title('Streamlit Example')

        st.write("""
        # Explore different classifier and datasets
        Which one is the best?
        """)

        self.dataset_name = 'Breast Cancer'
        st.write(f"## {self.dataset_name} Dataset")
        

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Random Forest')
        )

        
    def get_dataset(self,file_path):
        
        self.df = pd.read_csv(file_path)

        
        # X ve y verilerini ayır
        self.y = self.df['diagnosis']  
        self.X = self.df.drop('area_worst', axis=1)



        st.write('Shape of dataset:', self.X.shape)
        st.write('number of classes:', len(np.unique(self.y)))


        # Veri kümesinin ilk 10 satırını içeren DataFrame'i oluştur
        self.df = pd.concat([self.X, self.y], axis=1)  # Y ile birleştirilmiş DataFrame
        self.df = pd.DataFrame(self.X, columns=self.X.columns)
        st.write("İlk 10 Satır:")
        st.write(self.df.head(10))
        
        # Veri setinin sütunlarını göster
        st.write("Sütunlar:")
        st.write(self.df.columns)


        # 'diagnosis' sütunundaki 'M' ve 'B' değerlerini 1 ve 0 olarak değiştirme
        self.df['diagnosis'] = self.df['diagnosis'].map({'M': 1, 'B': 0})

        # X ve y verilerini ayırma
        self.X = self.df.drop('diagnosis', axis=1)  # Hedef sütunu düşür
        self.y = self.df['diagnosis']  # Hedef sütunu ayır


    def preprocess_data(self):

        # Gereksiz sütunları temizleme
        df_cleaned = self.df.drop(self.df.columns[4:33], axis=1)

        
        # Datanın son 10 satırını göster
        st.write("Son 10 Satır:")
        st.write(df_cleaned.tail(10))

        return df_cleaned
    
    
    
    def add_parameter_ui(self):
        if self.classifier_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 15.0)
            self.params['C'] = C
        elif self.classifier_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15)
            self.params['K'] = K
        else:
           alpha = st.sidebar.slider('Alpha', 0.1, 10.0, 1.0)
           self.params = {'alpha': alpha}


    def get_classifier(self):
        if self.classifier_name == 'SVM':
            self.clf  = SVC(C=self.params['C'])
        elif self.classifier_name == 'KNN':
            self.clf  = KNeighborsClassifier(n_neighbors=self.params['K'])
        else:
            self.clf  = MultinomialNB(alpha=self.params['alpha'])

    def create_sinusoid(self):
        fig = plt.figure()
        Fs = 8000
        f = 5
        sample = 8000
        x = np.arange(sample)
        y = np.sin(2 * np.pi * f * x / Fs)
        plt.plot(x, y)
        plt.xlabel('sample(n)')
        plt.ylabel('voltage(V)')
        return fig


    def korelasyon_matris(self):
        correlation_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Korelasyon Matrisi')
        plt.show()

        # 'Malignant' ve 'Benign' türlerine göre veriyi ayır
        malignant_data = self.df[self.df['diagnosis'] == 1]
        benign_data = self.df[self.df['diagnosis'] == 0]

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='radius_mean', y='texture_mean', data=malignant_data, label='Malignant')
        sns.scatterplot(x='radius_mean', y='texture_mean', data=benign_data, label='Benign')
        plt.title('radius_mean vs texture_mean')
        plt.xlabel('radius_mean')
        plt.ylabel('texture_mean')
        plt.legend()
        plt.show()

    def generate(self):
        self.get_classifier()
        #### CLASSIFICATION ####
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1234)

        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.write(f'Classifier = {self.classifier_name}')
        st.write(f'Accuracy =', acc)
        st.write(f'Precision =', precision)
        st.write(f'Recall =', recall)
        st.write(f'F1 Score =', f1)

        conf_matrix = confusion_matrix(y_test, y_pred)
        st.write(f'Confusion Matrix =', conf_matrix)

        #### PLOT DATASET ####
        # Project the data onto the 2 primary principal components
        pca = PCA(2)
        X_projected = pca.fit_transform(self.X)

        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]

        fig = plt.figure()
        plt.scatter(x1, x2,
                c=self.y, alpha=0.8,
                cmap='viridis')
        st.pyplot(fig)

        # fig = self.create_sinusoid()
        #
        # #plt.show()
        # st.pyplot(fig)
        #
        # for i in range(100):
        #     st.write("hello")
        #     time.sleep(1)

        # st.video("https://youtu.be/yVV_t_Tewvs")
