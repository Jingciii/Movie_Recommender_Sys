

"""
Created on Mon April 15 2019
@author: Jingci Wang
"""

"""
This is a class for Item-item recommender system model
"""

from .setting import loadMovieNames

nameDict = loadMovieNames()


class ItemCF(object):
    

    def __init__(self, ratings):
        '''
        embeddings: item*user matrix
        rating_matrix: item*user matrix
        '''

        self.ratings = ratings
        self.embeddings = ratings.pivot(
                                index='userId',
                                columns='movieId',
                                values='rating'
                                            ).fillna(0)

        self.rating_matrix = ratings.pivot(
                            index='userId',
                            columns='movieId',
                            values='rating'
                                            )

        #self.embeddings = embeddings
        #self.rating_matrix = rating_matrix
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

    def predict_similar_items(self, seed_item, n):
        '''Use the similarity_matrix to return n most similar items.'''
        similar_items = pd.DataFrame(self.similarity_matrix.loc[seed_item])
        similar_items.columns = ["similarity_score"]
        similar_items = similar_items.sort_values('similarity_score', ascending=False)
        similar_items = similar_items.head(n)
        similar_items.reset_index(inplace=True)
        similar_items = similar_items.rename(index=str, columns={"index": "item_id"})
        return similar_items.to_dict()
    
    def rating_predict(self, seed_item, seed_user, k):
        similar_items = self.predict_similar_items(seed_item, k)
        ''' Calculate bias terms '''
        mu = self.embeddings.mean().mean()
        bx = self.embeddings.loc[:, seed_user].mean() - mu
        bi = self.embeddings.loc[seed_item, :].mean() - mu
        num = sum([(similar_items['similarity_score'][str(i)] * (self.embeddings.loc[similar_items['item_id'][str(i)], seed_user] - bx - self.embeddings.loc[similar_items['item_id'][str(i)], :].mean() + mu)) for i in range(1, k)])
        den = sum(similar_items['similarity_score'].values())
        
        b_xi = mu + bx + bi
        
        return b_xi + num / (den + self.epsilon)

    def rmse_(self, pred, actual):
        return math.sqrt(mean_squared_error(pred, actual))


    def recommend(self, user, k):
        '''Recommend top k movies that user might like most'''
        
        item = ratings['movieId'].drop_duplicates()
        item['pred'] = item['movieId'].apply(lambda x: self.rating_predict(x['movieId'], user, self.k), axis=1)
        item.sort_values(by='pred', ascending=False, inplace=True)
        topid = item['movieId'][:k].values.flatten().tolist()
        return list(map(lambda x: nameDict[x], topid))
        
    
    #def rating_predict_matrix(self):
    #    rating = self.embeddings.values.T.dot(self.similarity_matrix.values)
    #    simsum = np.array([np.abs(self.similarity_matrix).sum(axis=1)])
    #    pred = pd.DataFrame(rating / (simsum + 0.001), columns=self.ids, index=self.embeddings.columns)
        
    #    return pred