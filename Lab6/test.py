import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import RandomForestClassifier


st.title('Лабораторная работа №6')

def load_data():
    data = pd.read_csv('data/winequality-red.csv', sep=',')
    return data

# Преобразуем целевой признак для задачи регрессии в признак для решения задачи классификации.
def regr_to_class(y: int) -> int:
    if y<6:
        result = 0
    else:
        result = 1        
    return result 


def process_data():
    scaller_1 = MinMaxScaler()
    data_vis2['fixed acidity'] = scaller_1.fit_transform(data_vis2[['fixed acidity']])
    scaller_2 = MinMaxScaler()
    data_vis2['volatile acidity'] = scaller_2.fit_transform(data_vis2[['volatile acidity']])
    scaller_3 = MinMaxScaler()
    data_vis2['residual sugar'] = scaller_3.fit_transform(data_vis2[['residual sugar']])
    scaller_4 = MinMaxScaler()
    data_vis2['chlorides'] = scaller_4.fit_transform(data_vis2[['chlorides']])
    scaller_5 = MinMaxScaler()
    data_vis2['free sulfur dioxide'] = scaller_5.fit_transform(data_vis2[['free sulfur dioxide']])
    scaller_6 = MinMaxScaler()
    data_vis2['total sulfur dioxide'] = scaller_6.fit_transform(data_vis2[['total sulfur dioxide']])
    scaller_7 = MinMaxScaler()
    data_vis2['pH'] = scaller_7.fit_transform(data_vis2[['pH']])
    scaller_8 = MinMaxScaler()
    data_vis2['sulphates'] = scaller_8.fit_transform(data_vis2[['sulphates']])
    scaller_9 = MinMaxScaler()
    data_vis2['alcohol'] = scaller_9.fit_transform(data_vis2[['alcohol']])

# Отрисовка ROC-кривой
def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, 
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    #plt.figure()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")    
    

data_load_state = st.text('Загрузка данных...')
data = load_data()
data_load_state.text('Данные загружены!')
st.write('Первые 5 строк набора данных')
st.write(data.head())
data_vis1 = data
data_vis1['Q'] = \
data_vis1.apply(lambda row: regr_to_class(row['quality']),axis=1)
st.write('Целевой признак изменен')
st.write(data_vis1.head())
data_vis2 = data_vis1
process_data()

data_clear = data_vis2.drop('quality', 1)

data_clear_x = data_clear.drop('Q', 1)
data_clear_y = data_clear['Q']

vine_X_train, vine_X_test, vine_y_train, vine_y_test = train_test_split(data_clear_x, data_clear_y, test_size = 0.3, random_state = 3)

st.write('Размер обучающей выборки:', vine_X_train.shape, vine_y_train.shape)
st.write('Размер тестовой выборки:', vine_X_test.shape, vine_y_test.shape)
st.write('**Отмасштабированный набор данных:**')
st.write(data_clear.head())
st.sidebar.header('Метод случайного леса')
par1 = st.sidebar.slider('Количество деревьев в лесу:', min_value=10, max_value=140, value=50, step=10)
par2 = st.sidebar.slider('Максимальная глубина дерева:', min_value=2, max_value=8, value=3, step=1)
par3 = st.sidebar.selectbox('Критерий:',('gini', 'entropy'), index=0, help='Функция измерения качества разбиения')

st.write('Вы выбрали параметры для модели случайного леса:')
st.write('**1. Количество деревьев в лесу:**', par1)
st.write('**2. Глубина дерева:**', par2)
st.write('**3. Функция качества разбиения:**', par3)

model = RandomForestClassifier(n_estimators=par1, max_depth=par2, criterion =par3)


def accuracy_score_for_classes(
    y_true: np.ndarray, 
    y_pred: np.ndarray) -> Dict[int, float]:
    
    # Для удобства фильтрации сформируем Pandas DataFrame 
    d = {'t': y_true, 'p': y_pred}
    df = pd.DataFrame(data=d)
    # Метки классов
    classes = np.unique(y_true)
    # Результирующий словарь
    res = dict()
    # Перебор меток классов
    for c in classes:
        # отфильтруем данные, которые соответствуют 
        # текущей метке класса в истинных значениях
        temp_data_flt = df[df['t']==c]
        # расчет accuracy для заданной метки класса
        temp_acc = accuracy_score(
            temp_data_flt['t'].values, 
            temp_data_flt['p'].values)
        # сохранение результата в словарь
        res[c] = temp_acc
    return res

def print_accuracy_score_for_classes(
    y_true: np.ndarray, 
    y_pred: np.ndarray):
    
    accs = accuracy_score_for_classes(y_true, y_pred)
    if len(accs)>0:
        st.write('**Accuracy по классам:**')
    for i in accs:
        st.write('Класс {}: **accuracy = {}**'.format(i, accs[i]))


def print_models(model, X_train, X_test, y_train, y_test):
    tea = st.text('Обучение модели...')
    model.fit(X_train, y_train)
    tea.text('Модель обучена!')
    # Предсказание значений
    Y_pred = model.predict(X_test)
    # Предсказание вероятности класса "1" для roc auc
    Y_pred_proba_temp = model.predict_proba(X_test)
    Y_pred_proba = Y_pred_proba_temp[:,1]
   
    roc_auc = roc_auc_score(y_test.values, Y_pred_proba)

    print_accuracy_score_for_classes(y_test, Y_pred)

    #Отрисовка ROC-кривых 
    fig, ax = plt.subplots(ncols=2, figsize=(10,5))    
    draw_roc_curve(y_test.values, Y_pred_proba, ax[0])
    plot_confusion_matrix(model, X_test, y_test.values, ax=ax[1],
                        display_labels=['0','1'], 
                        cmap=plt.cm.Blues, normalize='true')
    fig.suptitle('Случайный лес')
    st.pyplot(fig)

st.write('# Оценка качества модели:')
print_models(model, vine_X_train, vine_X_test, vine_y_train, vine_y_test)


#fig = plot_confusion_matrix(model, vine_X_test, vine_y_test, 
#                     display_labels=['Bad', 'Good'], cmap=plt.cm.Blues)
#st.pyplot(fig)
#scores = cross_val_score(KNeighborsClassifier(n_neighbors=cv_knn), 
#    data_X, data_y, scoring='accuracy', cv=cv_slider)