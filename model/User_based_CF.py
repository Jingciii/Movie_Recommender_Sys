
"""
Created on Mon April 15 2019
@author: Jingci Wang
"""

"""
This is a class for User-user recommender system model
"""

from .setting import loadMovieNames

nameDict = loadMovieNames()

class UserCF(object):
    '''This class calculates a similarity matrix from latent embeddings.
    Input: embeddings - a pandas dataframe of items and latent dimensions.
    '''

    def __init__(self, embeddings, rating_matrix):
        '''
        embeddings: user*item matrix
        rating_matrix: user*item matrix
        '''
        self.embeddings = embeddings
        self.rating_matrix = rating_matrix
        self.ids = embeddings.index.tolist()
        self.epsilon = 0.001

    def calculate_cosine_similarity_matrix(self):
        '''Calculates a cosine similarity matrix from the embeddings'''
        sub_rowmean = self.rating_matrix.sub(self.rating_matrix.mean(axis=1), axis=0).fillna(0)
        #sub_rowmean = self.embeddings
        self.similarity_matrix = pd.DataFrame(cosine_similarity(
            X=sub_rowmean),
            index=self.ids)
        self.similarity_matrix.columns = self.ids
        #return self.similarity_matrix
        print('Finish calculating similarity matrix')

    def predict_similar_users(self, seed_user, n):
        '''Use the similarity_matrix to return n most similar items.'''
        similar_users = pd.DataFrame(self.similarity_matrix.loc[seed_user])
        similar_users.columns = ["similarity_score"]
        similar_users = similar_users.sort_values('similarity_score', ascending=False)
        similar_users = similar_users.head(n)
        similar_users.reset_index(inplace=True)
        similar_users = similar_users.rename(index=str, columns={"index": "user_id"})
        return similar_users.to_dict()
    
    def rating_predict(self, seed_user, seed_item, k):
        similar_users = self.predict_similar_users(seed_user, k)
        mu = self.embeddings.mean().mean()
        bi = self.embeddings.loc[:, seed_item].mean() - mu
        bx = self.embeddings.loc[seed_user, :].mean() - mu
        num = sum([(similar_users['similarity_score'][str(i)] * (self.embeddings.loc[similar_users['user_id'][str(i)], seed_item] - bi - self.embeddings.loc[similar_users['user_id'][str(i)], :].mean() + mu)) for i in range(1, k)])
        den = sum(similar_users['similarity_score'].values())
        
        b_xi = mu + bx + bi
        
        return b_xi + num / (den + self.epsilon)

    def rmse_(self, pred, actual):
        return math.sqrt(mean_squared_error(pred, actual))

    def recommend(self, user, k):
        '''Recommend top k movies that user might like most'''
        
        item = ratings['movieId'].drop_duplicates()
        item['pred'] = item['movieId'].apply(lambda x: self.rating_predict(user, x['itemId'], self.k), axis=1)
        item.sort_values(by='pred', ascending=False, inplace=True)
        topid = item['movieId'][:k].values.flatten().tolist()
        return list(map(lambda x: nameDict[x], topid))
        
    
   # def rating_predict_matrix(self):
   #     rating = self.embeddings.values.T.dot(self.similarity_matrix.values)
   #     simsum = np.array([np.abs(self.similarity_matrix).sum(axis=1)])
   #     pred = pd.DataFrame(rating / (simsum + 0.001), columns=self.ids, index=self.embeddings.columns)
        
   #     return pred