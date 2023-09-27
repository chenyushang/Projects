import pickle
from os.path import exists

import dash_bootstrap_components as dbc
import pandas as pd
import requests
from dash import Dash, html, dcc, dash_table, Input, Output, State

from movie_data_loader import load_movie_full


def load_movie_dataframe(path='movie_full.pkl'):
    file_exists = exists(path)
    if file_exists:
        file = open(path, 'rb')
        movie_full_ = pickle.load(file)
        file.close()
        return movie_full_
    else:
        file = open(path, 'wb')
        movie_full_ = load_movie_full()
        pickle.dump(movie_full_, file)
        file.close()
        return movie_full_


movie_full = load_movie_dataframe()
movie_top = movie_full.head(1000)
dff = pd.DataFrame(columns=['title', 'rating', 'genres', 'movieId', 'tmdbId', 'tag'])
movie_db_api_key = '79366b2dfeefcde05c7a8a2583cc694c'
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
df_gen = movie_full.genres.unique()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
heading = html.H1('Dash Movie Recommender', style={'textAlign': 'center', 'color': "red"})


# These buttons are added to the app just to show the Boostrap theme colors
content = html.Div(
    [

        dbc.Row(
            [
                dbc.Col(
                    dbc.Button('Search', id='search-button', className='me-1', n_clicks=0),
                ),
                dbc.Col(
                    dbc.Input(placeholder='Type a tag', id='tag-filter', type='text', className='me-1'),
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Label('Ratings', style = {'color': 'red',  "font-size": '20px'}),
                ),
                dbc.Col(
                    dcc.Dropdown(options=[
                        {'label': 'Descending', 'value': 'Descending'},
                        {'label': 'Ascending', 'value': 'Ascending'},
                    ],
                        value='Descending',
                        id='ratings-order', className='me-1'),
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Label('Movies Recommendations', style = {'color': 'red',  "font-size": '50px'}),
                ),
                dbc.Col(
                    dcc.Dropdown(df_gen, id='dropdown'),
                        #html.Div(id='output'),
                    ),
                ]
            ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Carousel(id='carousel-placeholder',
                                 items=[],
                                 controls=True,
                                 indicators=True,
                                 className="carousel-fade"), width=3,
                ),
                dbc.Col(
                    dash_table.DataTable(
                        id='movie-table',
                        style_table={'overflowY': 'auto'},
                        data=dff.to_dict(orient='records'),
                        columns=[{'id': c, 'name': c} for c in dff.columns],
                        style_cell=dict(textAlign='left'),
                        editable=False,
                        filter_action="native",
                        sort_action="native",
                        column_selectable="single",
                        row_deletable=False,
                        page_action="native",
                        page_current=0,
                        page_size=12,
                        export_format="csv",
                        style_data={
                            'backgroundColor': 'rgb(50, 50, 50)',
                            'color': 'white'
                                    }
                    ), width=9
                ),
            ]
        ),
        dbc.Row([
            dbc.Col(
                html.Label('Top 12 Horror Movies', style = {'color': 'red',  "font-size": '50px'}),
                ),
            ]),
        dbc.Row([
            dbc.Col(
                dbc.Carousel(id='carousel-placeholder2',
                                 items=[],
                                 controls=True,
                                 indicators=True,
                                 className="carousel-fade"), width=3,
                ),
            dbc.Col(
                    dash_table.DataTable(
                        id='Horror-table',
                        style_table={'overflowY': 'auto'},
                        data=dff.to_dict(orient='records'),
                        columns=[{'id': c, 'name': c} for c in dff.columns],
                        style_cell=dict(textAlign='left'),
                        editable=False,
                        filter_action="native",
                        sort_action="native",
                        column_selectable="single",
                        row_deletable=False,
                        page_action="native",
                        page_current=0,
                        page_size=12,
                        export_format="csv",
                        style_data={
                            'backgroundColor': 'rgb(50, 50, 50)',
                            'color': 'white'
                                    }
                    ), width=9
                ),
            ]),
        dbc.Row([
            dbc.Col(
                html.Label('Top 12 Comedy Movies', style = {'color': 'red',  "font-size": '50px'}),
                ),
            ]),
        dbc.Row([
            dbc.Col(
                dbc.Carousel(id='carousel-placeholder3',
                                 items=[],
                                 controls=True,
                                 indicators=True,
                                 className="carousel-fade"), width=3,
                ),
            dbc.Col(
                    dash_table.DataTable(
                        id='Comedy-table',
                        style_table={'overflowY': 'auto'},
                        data=dff.to_dict(orient='records'),
                        columns=[{'id': c, 'name': c} for c in dff.columns],
                        style_cell=dict(textAlign='left'),
                        editable=False,
                        filter_action="native",
                        sort_action="native",
                        column_selectable="single",
                        row_deletable=False,
                        page_action="native",
                        page_current=0,
                        page_size=12,
                        export_format="csv",
                        style_data={
                            'backgroundColor': 'rgb(50, 50, 50)',
                            'color': 'white'
                                    }
                    ), width=9
                ),
            ]),
        dbc.Row([
            dbc.Col(
                html.Label('The Crappiest 12 Movies Forever', style = {'color': 'red',  "font-size": '50px'}),
                ),
            ]),
        dbc.Row([
            dbc.Col(
                dbc.Carousel(id='carousel-placeholder4',
                                 items=[],
                                 controls=True,
                                 indicators=True,
                                 className="carousel-fade"), width=3,
                ),
            dbc.Col(
                    dash_table.DataTable(
                        id='bad-table',
                        style_table={'overflowY': 'auto'},
                        data=dff.to_dict(orient='records'),
                        columns=[{'id': c, 'name': c} for c in dff.columns],
                        style_cell=dict(textAlign='left'),
                        editable=False,
                        filter_action="native",
                        sort_action="native",
                        column_selectable="single",
                        row_deletable=False,
                        page_action="native",
                        page_current=0,
                        page_size=12,
                        export_format="csv",
                        style_data={
                            'backgroundColor': 'rgb(50, 50, 50)',
                            'color': 'white'
                                    }
                    ), width=9
                ),
            ]),
    ],
)

app.layout = dbc.Container(children=[heading, content], style = {'background-color': 'rgb(50, 50, 50)'})

@app.callback(
    Output('carousel-placeholder', 'items'),
    Output('movie-table', 'data'),
    Output('dropdown', 'options'),
    [Input('search-button', 'n_clicks'),
     State('tag-filter', 'value'),
     State('ratings-order', 'value'),
     Input('dropdown', 'value')]
)
def create_content(n_clicks, tag_filter, ratings_order, dropdown):
    global movie_full, dff, df_gen

    if (tag_filter == '') or (tag_filter == None):
        dff_pre = movie_full.sample(n=12)
    else:
        dff_pre = movie_full[movie_full['tag'] == tag_filter]

    dff = dff_pre.sort_values(by=['rating'], ascending=ratings_order == 'Ascending')
    dff = dff[['title', 'rating', 'genres', 'movieId', 'tmdbId', 'tag']]

    movies_to_display = []
    for index, row in dff.iterrows():
        tmdbid = str(row['tmdbId'])

        url = f'https://api.themoviedb.org/3/movie/{tmdbid}?api_key={movie_db_api_key}&language'
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            poster_path = data['poster_path']
            full_path = 'https://via.placeholder.com/150' if not poster_path \
                else f'https://image.tmdb.org/t/p/w500/{poster_path}'
        else:
            full_path = 'https://via.placeholder.com/150'

        movie_information = {
            'key': index,
            'src': full_path,
        }
        movies_to_display.append(movie_information)
        if (tag_filter != None) and (tag_filter != '') :
            if (dropdown != None):
                df_filtered = dff[dff['genres'].eq(dropdown)]
                df_gen = dff_pre.genres.unique()
            else:
                df_filtered = dff
                df_gen = df_filtered.genres.unique()
        else:
            if (dropdown != None):
                df_filtered = movie_full[movie_full['genres'] == dropdown]
                df_gen = movie_full.genres.unique()
            else:
                df_filtered = dff
                df_gen = movie_full.genres.unique()
    
    return movies_to_display, df_filtered.to_dict(orient='records'), df_gen

@app.callback(
    Output('carousel-placeholder3', 'items'),
    Output('Comedy-table', 'data'),
    [Input('dropdown', 'value')]
    )
def create_content2(dropdown):
    dff_pre = movie_full[movie_full['genres'] == 'Comedy']
    dff = dff_pre.sort_values(by=['rating'], ascending=False)
    dff = dff.head(12)
    dff = dff[['title', 'rating', 'genres', 'movieId', 'tmdbId', 'tag']]
    movies_to_display = []
    for index, row in dff.iterrows():
        tmdbid = str(row['tmdbId'])

        url = f'https://api.themoviedb.org/3/movie/{tmdbid}?api_key={movie_db_api_key}&language'
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            poster_path = data['poster_path']
            full_path = 'https://via.placeholder.com/150' if not poster_path \
                else f'https://image.tmdb.org/t/p/w500/{poster_path}'
        else:
            full_path = 'https://via.placeholder.com/150'

        movie_information = {
            'key': index,
            'src': full_path,
        }
        movies_to_display.append(movie_information)
    return movies_to_display, dff.to_dict(orient='records')

@app.callback(
    Output('carousel-placeholder2', 'items'),
    Output('Horror-table', 'data'),
    [Input('dropdown', 'value')]
    )
def create_content3(dropdown):
    dff_pre = movie_full[movie_full['genres'] == 'Horror']
    dff = dff_pre.sort_values(by=['rating'], ascending=False)
    dff = dff.head(12)
    dff = dff[['title', 'rating', 'genres', 'movieId', 'tmdbId', 'tag']]
    movies_to_display = []
    for index, row in dff.iterrows():
        tmdbid = str(row['tmdbId'])

        url = f'https://api.themoviedb.org/3/movie/{tmdbid}?api_key={movie_db_api_key}&language'
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            poster_path = data['poster_path']
            full_path = 'https://via.placeholder.com/150' if not poster_path \
                else f'https://image.tmdb.org/t/p/w500/{poster_path}'
        else:
            full_path = 'https://via.placeholder.com/150'

        movie_information = {
            'key': index,
            'src': full_path,
        }
        movies_to_display.append(movie_information)
    return movies_to_display, dff.to_dict(orient='records')

@app.callback(
    Output('carousel-placeholder4', 'items'),
    Output('bad-table', 'data'),
    [Input('dropdown', 'value')]
    )
def create_content4(dropdown):
    dff = movie_full.sort_values(by=['rating'], ascending=True)
    dff = dff.head(12)
    dff = dff[['title', 'rating', 'genres', 'movieId', 'tmdbId', 'tag']]
    movies_to_display = []
    for index, row in dff.iterrows():
        tmdbid = str(row['tmdbId'])

        url = f'https://api.themoviedb.org/3/movie/{tmdbid}?api_key={movie_db_api_key}&language'
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            poster_path = data['poster_path']
            full_path = 'https://via.placeholder.com/150' if not poster_path \
                else f'https://image.tmdb.org/t/p/w500/{poster_path}'
        else:
            full_path = 'https://via.placeholder.com/150'

        movie_information = {
            'key': index,
            'src': full_path,
        }
        movies_to_display.append(movie_information)
    return movies_to_display, dff.to_dict(orient='records')
'''
@app.callback(
    Output('movie-table', 'data'),
    Input('dropdown', 'value'),
    Input('sort', 'value')
)
def callback_func(dropdown,sort):
    df_filtered = movie_top[movie_top['genres'].eq(dropdown)]
    if sort == 'Descending':
        df_filtered2 = df_filtered.sort_values(by = ['rating'], ascending=False)
    else:
        df_filtered2 = df_filtered.sort_values(by = ['rating'])
    return df_filtered2.to_dict(orient='records')
'''
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

