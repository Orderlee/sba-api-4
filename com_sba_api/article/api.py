from flask_restful import Resource, reqparse
from com_sba_api.article.dto import ArticleDto
from com_sba_api.article.dao import ArticleDao

class Article(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('user_id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('item_id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('title', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('content', type=str, required=False, help='This field cannot be left blank')

    
    def post(self):
        data = self.parset.parse_args()
        article = ArticleDto(data['title'], data['content'], data['user_id'], data['item_id'])
        try: 
            article.save()
        except:
            return {'message': 'An error occured inserting the article'}, 500
        return article.json(), 201
    
    
    def get(self, id):
        article = ArticleDao.find_by_id(id)
        if article:
            return article.json()
        return {'message': 'Article not found'}, 404

    def put(self, id):
        data = Article.parser.parse_args()
        article = ArticleDao.find_by_id(id)

        article.title = data['title']
        article.content = data['content']
        article.save()
        return article.json()

class Articles(Resource):
    def get(self):
        return {'articles': list(map(lambda article: article.json(), ArticleDao.find_all()))}
        # return {'articles':[article.json() for article in ArticleDao.find_all()]}