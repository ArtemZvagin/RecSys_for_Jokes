import pickle
import pandas as pd 

from surprise.dataset import Dataset, Reader
from surprise import KNNBaseline, SVD

# Загрузка данных
train_df = pd.read_csv('data/train_joke_df.csv')

reader = Reader(rating_scale=(-10, 10))
data = Dataset.load_from_df(train_df[['UID', 'JID', 'Rating']],  reader)

# Обучение моделей
sim_options = {'name': 'pearson_baseline', 'user_based': False, 'k': 2}
algo_knn = KNNBaseline(sim_options=sim_options, verbose=False)
algo_svd = SVD(n_factors=500, random_state=42)

trainset = data.build_full_trainset()    
algo_knn.fit(trainset)
algo_svd.fit(trainset)

# Сохранение для дальнейшего использования
path = 'my_project/src/models/'
with open(path + 'model_knn.pkl', 'wb') as file:
    pickle.dump(algo_knn, file)
    
with open(path + 'model_svd.pkl', 'wb') as file:
    pickle.dump(algo_svd, file)