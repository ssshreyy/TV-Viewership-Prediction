from flask import Flask, request, redirect, url_for
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
import Exporter
import pandas as pd

app = Flask(__name__)
api = Api(app)

CORS(app)


@app.route("/search", methods = ['POST','GET'])
def search():
    if request.method == 'POST':
        username = request.form.get('username')
        query = request.form.get('query')
        since = request.form.get('since')
        until = request.form.get('until')
        maxNo = request.form.get('maxNo')
        top = request.form.get('top')
        tweetSearchParameters=[]
        if username:
            tweetSearchParameters.append('--username')
            tweetSearchParameters.append(username)
        if query:
            tweetSearchParameters.append('--query')
            tweetSearchParameters.append(query)
        if since:
            tweetSearchParameters.append('--since')
            tweetSearchParameters.append(since)
        if until:
            tweetSearchParameters.append('--until')
            tweetSearchParameters.append(until)
        if maxNo:
            tweetSearchParameters.append('--maxtweets')
            tweetSearchParameters.append(maxNo)
        if top == True:
            tweetSearchParameters.append('--toptweets')

        Exporter.main(tweetSearchParameters)

        data = pd.read_csv('tweet-data.csv', usecols=range(12), index_col=False, low_memory=False)

        # return jsonify([{'tweetSearchParameters':data['Text'][1]}])
        return(pd.DataFrame.to_json(data, orient='index'))

    else:
        return 0

if __name__ == '__main__':
     app.run(port=5003)