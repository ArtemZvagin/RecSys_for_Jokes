import pickle
import pandas as pd

folder_name = 'my_project/src/models/'

with open(folder_name + 'model_knn.pkl', 'rb') as file:
    algo_knn = pickle.load(file)

with open(folder_name +'model_svd.pkl', 'rb') as file:
    algo_svd = pickle.load(file)


def predict_use_two_algo(test, coef=0.5):  
    rating = []    
    for uid, iid in test[['UID', 'JID']].values:
        first = algo_knn.predict(uid=uid, iid=iid).est 
        second = algo_svd.predict(uid=uid, iid=iid).est 
        rating.append(first * coef + second * (1 - coef)) 
    test['Rating'] = rating
    
    result = pd.DataFrame(data=sorted(test.UID.unique()), columns=['UID'])
    res = []
    for uid in result.UID:
        sorted_ratings = test[test.UID == uid].sort_values(by='Rating', ascending=False)
        res.append([sorted_ratings.Rating.iloc[0], sorted_ratings.JID.values[:10].tolist()])
    result['Rating'] = res 
    return result