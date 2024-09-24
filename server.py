import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any , Dict
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        self.valid_locations = [
            "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
            "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
            "El Paso, Texas", "Escondido, California", "Fresno, California",
            "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
            "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
            "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
        ]
        self.reviews=pd.read_csv('data/reviews.csv').to_dict('records')

    def save_review(self,review: Dict[str, Any])->None:
        old_df=pd.read_csv('data/reviews.csv')

        new_df = pd.DataFrame([review])
        df=pd.concat([old_df, new_df], ignore_index=True)
        df.to_csv('data/reviews.csv', index=False)
        
        # print('saved to csv')

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def filter_review(self,reviews,location,start_date,end_date):
        filtered_review=reviews
        if location and location in self.valid_locations:
            filtered_review=[review for review in filtered_review if review['Location']==location]
        if start_date:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            filtered_review = [review for review in filtered_review if datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S") >= start_date]
        if end_date:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            filtered_review = [review for review in filtered_review if datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S") <= end_date]

        return filtered_review
    
    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Create the response body from the reviews and convert to a JSON byte string
            # response_body = json.dumps(self.reviews, indent=2).encode("utf-8")
            
            # Write your code here
            res_with_sentiment=[]
            for i in self.reviews:
                sentiment=self.analyze_sentiment(i['ReviewBody'])
                res={
                    'ReviewId':i['ReviewId'],
                    'ReviewBody':i['ReviewBody'],
                    'Location':i['Location'],
                    'Timestamp':i['Timestamp'],
                    'sentiment':sentiment
                }
                res_with_sentiment.append(res)
            
            res_with_sentiment_sorted=sorted(res_with_sentiment,key=lambda x: x['sentiment']['compound'],reverse=True)

            #Parsing the query parameters
            d=parse_qs(environ['QUERY_STRING'])

            loc=d.get('location',[None])[0]
            start=d.get('start_date',[None])[0]
            end=d.get('end_date',[None])[0]

            response=self.filter_review(res_with_sentiment_sorted,loc,start,end)
            
            f_res=json.dumps(response, indent=2).encode("utf-8")


            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(f_res)))
             ])
            
            return [f_res]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                c_len = int(environ.get("CONTENT_LENGTH", 0))

                request_body = environ['wsgi.input'].read(c_len)
                
                if environ.get("CONTENT_TYPE") == "application/json":
                        data = json.loads(request_body)
                else:
                    data = parse_qs(request_body.decode())

                if not data or 'Location' not in data or 'ReviewBody' not in data:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Missing required parameters"}).encode("utf-8")]
                
                loc=data['Location'][0]
                review=data['ReviewBody'][0]
                TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

                if loc not in self.valid_locations:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid location"}).encode("utf-8")]

                response={
                    "ReviewId":str(uuid.uuid4()),
                    'Location':loc,
                    'Timestamp':datetime.now().strftime(TIMESTAMP_FORMAT),
                    'ReviewBody':review
                }
                
                self.reviews.append(response)
                self.save_review(response)

                res_body=json.dumps(response).encode('utf-8')
                start_response("201 OK", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(res_body)))
                ])
                return [res_body]

            except Exception as e:
                print(e)
                start_response("500 Internal Server Error", [("Content-Type", "application/json")])
                return [json.dumps({"error": "Internal server error", "message": str(e)}).encode("utf-8")]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()