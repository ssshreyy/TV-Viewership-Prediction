import pandas as pd
import codecs,bisect,datetime

def fun(tweet_data,start,end): 
    return int(end)-int(start)

def date_change(str_date):
    if str_date:
        return datetime.datetime.strptime(str_date, '%d-%m-%Y').strftime('%Y-%m-%d')

def viewers_change(str_views):
    if str_views == 'NaN':
        return '0'
    return str(int(float(str_views) * 1000000))

# def date_change(str_date):
#     return datetime.datetime.strptime(str_date, '%B %d, %Y')

# def viewers_change(str_views):
#     return str(int(float(str_views.strip().split('[')[0]) * 1000000))

def main():

    viewer_data = pd.read_csv('simpsons_episodes.csv', index_col=False, usecols=range(13))
    tweet_data = pd.read_csv('tweet.csv', index_col=False, usecols=range(12))
    
    tweet_data = tweet_data.sort_values('Date', ascending = True)

    viewer_data['original_air_date'] = list(map(date_change,viewer_data['original_air_date']))
    viewer_data['us_viewers_in_millions'] = list(map(viewers_change,viewer_data['us_viewers_in_millions']))
    # viewer_data['Original air date'] = list(map(date_change,viewer_data['Original air date']))
    # viewer_data['U.S. viewers(millions)'] = list(map(viewers_change,viewer_data['U.S. viewers(millions)']))

    first_date = bisect.bisect_left(viewer_data['original_air_date'] , '2009-01-01')
    last_date = bisect.bisect_left(viewer_data['original_air_date'] , '2019-01-01')
    temp1 = first_date
    temp2 = first_date
    print('hello')
    print(first_date)
    print(last_date)
    for i in range(first_date,last_date-2):
        temp1 = str(viewer_data['original_air_date'][i])
        temp2 = str(viewer_data['original_air_date'][i+1])
        print(i,temp1,temp2)
        # print(temp1)
        # print(temp2)

        start = bisect.bisect_left(tweet_data['Date'], temp1)
        end = bisect.bisect_left(tweet_data['Date'], temp2)
        print(start,end,fun(tweet_data,start,end))
        
    # viewer_data['score'] = final
    # print(final)
    # start = bisect.bisect_left(viewer_data['original_air_date'] ,datetime.datetime.strptime('2007-01-01', '%Y-%m-%d'))

if __name__ == "__main__":
    main()