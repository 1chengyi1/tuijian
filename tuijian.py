import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
data_df = pd.read_excel('data2.xlsx', usecols=( 0,1,2,3,4), names=['user','user_id','location_id', 'location','score'])
location_matrix = data_df.pivot_table(index='user_id', columns='location', values='score')
def recommend_movies(pred, data_df=data_df, location_matrix=location_matrix):
        scores = pd.DataFrame(data_df.groupby('location')['score'].mean())
        scores['number_of_scores'] = data_df.groupby('location')['score']
        scores.sort_values('number_of_scores', ascending=False).head(10)
        AFO_user_score = location_matrix[pred]
        similar_to_air_force_one=location_matrix.corrwith(AFO_user_score)
        corr_AFO = pd.DataFrame(similar_to_air_force_one, columns=['correlation'])
        corr_AFO.dropna(inplace=True)
        result=corr_AFO.sort_values(by='correlation', ascending=False)
        similar_location_titles=result[1:4]
        return similar_location_titles
print(recommend_movies('世纪钟广场'))