import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from ast import literal_eval
from typing import List, Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity



app = FastAPI()


# CARGA DE DATOS


df_userdata = pd.read_csv('DATA/df_userdata.csv', low_memory= True,  encoding='utf-8')

df_countreviews = pd.read_csv('DATA/df_countreviews.csv',low_memory= True,  encoding='utf-8')

df_genre = pd.read_csv('DATA/df_genre.csv', low_memory= True,  encoding='utf-8')

df_genre_def = pd.read_csv('DATA/df_genre_def.csv', low_memory= True,  encoding='utf-8')

df_developer = pd.read_csv("DATA/df_developer.csv", low_memory= True,  encoding='utf-8')

df_sentimiento = pd.read_csv("DATA/df_sentimiento.csv", low_memory= True,  encoding='utf-8')

df_recomendacion = pd.read_csv("DATA/df_recomendacion.csv", low_memory= True,  encoding='utf-8')



# 1. Funcion dinero gastado por usuario.

#@app.get("/user/{user_id}", response_model=List[str])
#def userdata(User_id: str):
    #user_data = df_userdata[df_userdata['user_id'] == User_id]
    
    #if user_data.empty:
        #return f"User_id {User_id} inexistente", None, None
    
    #money_spent = user_data['price'].sum() - user_data['discount_price'].sum()
    #recommendation_percentage = (user_data['recommend'].mean()) * 100
    #item_count = user_data['items_count'].values[0]
    
    #return f"Usuario: {User_id}", f"Dinero gastado: ${money_spent:.2f}", f"Porcentaje de recomendación: {recommendation_percentage:.2f}%", f"Cantidad de items: {item_count}"

# --------------------------------

#2. Funcion cantidad de usuarios que realizaron comentarios.

#def countreviews(start_date, end_date):
    #df_reviews = df_countreviews[df_countreviews["posted"].between(start_date, end_date)]
    #users_with_recommendations = df_reviews.groupby("user_id")["recommend"].sum()
    #users_count = len(users_with_recommendations)
    #recommendations_percentage = users_with_recommendations / users_count

    #return {"users": users_count, "recommendations": recommendations_percentage.mean(),}

#class DateRange(BaseModel):
    #start_date: str
    #end_date: str

#Ruta para la funcion countreviews
#@app.post("/countreviews", response_model=dict)
#async def get_review_counts(date_range: DateRange):
    #result = countreviews(date_range.start_date, date_range.end_date)
    #return result

# --------------------------------

# 3. Funcion puesto por género.

#def genre(genero: str, df_genre: pd.DataFrame):
    #generos_unicos = [
        #"Action", "Casual", "Indie", "Simulation", "Strategy", "Free to Play", "RPG", "Sports", 
        #"Adventure", "Racing", "Early Access", "Massively Multiplayer", "Animation & Modeling", 
        #"Video Production", "Utilities", "Web Publishing", "Education", "Software Training", 
        #"Design & Illustration", "Audio Production", "Photo Editing", "Accounting", 'VR', 'Tutorial', 
        #'Golf', 'Horror', 'Lovecraftian', 'Survival Horror', 'First-Person', 'Based On A Novel', 'FPS'
    #]
    #genero_suma = {gen: 0 for gen in generos_unicos}
    #df_genre = df_genre[df_genre['genres'].notna()]
    #for index, row in df_genre.iterrows():
        #generos = literal_eval(row['genres'])
        #for g in generos:
            #if g in generos_unicos:
                #genero_suma[g] += row['playtime_forever']
    #genero_suma_ordenado = dict(sorted(genero_suma.items(), key=lambda item: item[1], reverse=True))
    #puesto = list(genero_suma_ordenado.keys()).index(genero)

    #return puesto + 1

# Ruta para la función genre
#@app.get("/genre")
#async def get_genre_puesto(genero: str = Query(..., title="Género a consultar")):
    #puesto = genre(genero, df_genre)
    #return {"genero": genero, "puesto": puesto}

# --------------------------------

# 4. Funcion top 5 de usuarios con más horas de juego.

#def userforgenre(genero: str, df_genre_def: pd.DataFrame):
    "horas_por_usuario = {}
    "for index, row in df_genre_def.iterrows():
        "generos = literal_eval(row['genres'])
        "if genero in generos:
            #user_id = row['user_id']
            #playtime_forever = row['playtime_forever']
            #if user_id in horas_por_usuario:
                #horas_por_usuario[user_id] += playtime_forever
            #else:
                #horas_por_usuario[user_id] = playtime_forever
    #top_usuarios = sorted(horas_por_usuario.items(), key=lambda item: item[1], reverse=True)[:5]

    #resultados = []
    #for user_id, horas in top_usuarios:
        #user_url = df_genre_def[df_genre_def['user_id'] == user_id]['user_url'].values[0]
        #resultados.append({'user_id': user_id, 'horas_de_juego': horas, 'user_url': user_url})

    #return resultados

# Clase de modelo para recibir los parámetros de entrada
#class GenreQuery(BaseModel):
    #genero: str

# Ruta para la función userforgenre
#@app.get("/userforgenre")
#async def get_top_users_by_genre(genero: str = Query(..., title="Género a consultar")):
    #top_usuarios = userforgenre(genero, df_genre_def)
    #return top_usuarios

# --------------------------------

# 5. Funcion contenido por desarrollador.

def developer(desarrollador: str, df_developer: pd.DataFrame):
    df_developer['release_date'] = pd.to_datetime(df_developer['release_date'], errors='coerce')
    df_filtered = df_developer[df_developer['developer'] == desarrollador]
    df_filtered = df_filtered.dropna(subset=['release_date'])

    result_data = []

    for year in range(df_filtered['release_date'].min().year, df_filtered['release_date'].max().year + 1):
        year_items = len(df_filtered[df_filtered['release_date'].dt.year == year])
        year_free_items = len(df_filtered[(df_filtered['release_date'].dt.year == year) & (df_filtered['price'] == 0)])

        if year_items > 0:
            free_percentage = (year_free_items / year_items) * 100
        else:
            free_percentage = 0

        result_data.append({'Año': year, 'Cantidad de Ítems': year_items, 'Contenido Free': f'{free_percentage:.0f}%'})

    result_df = pd.DataFrame(result_data)

    return result_df.to_dict(orient='records')

# Ruta para la función developer
@app.get("/developer")
async def get_developer_stats(desarrollador: str = Query(..., title="Nombre del Desarrollador")):
    resultado = developer(desarrollador, df_developer)
    return resultado

# --------------------------------

# 6. Funcion cantidad de reseñas por año de lanzamiento.

#def sentiment_analysis(año: int, df_sentimiento: pd.DataFrame):
    #df_sentimiento['release_date'] = pd.to_datetime(df_sentimiento['release_date'], errors='coerce')
    #df_reviews_filtered = df_sentimiento[df_sentimiento['release_date'].dt.year == año]
    #sentiment_counts = df_reviews_filtered['sentiment_analysis'].value_counts()
    #sentiment_dict = {
        #'Negative': sentiment_counts.get(0, 0),
        #'Neutral': sentiment_counts.get(1, 0),
        #'Positive': sentiment_counts.get(2, 0)
    #}

    #return sentiment_dict

# Ruta para la función sentiment_analysis
#@app.get("/sentiment")
#async def get_sentiment_analysis(año: int = Query(..., title="Año")):
    #resultado = sentiment_analysis(año, df_sentimiento)
    #return resultado

#  -----------------------------


# Modelo de Aprendizaje No Supervisado. RECOMENDACION.

#item_similarities = cosine_similarity(df_recomendacion.drop(['item_name', 'item_id'], axis=1))

#def recomendacion_juego(id_producto: int, num_recomendaciones: int = 5):
    #idx = df_recomendacion[df_recomendacion['item_id'] == id_producto].index[0]
    #sim_scores = list(enumerate(item_similarities[idx]))
    #sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #sim_scores = sim_scores[1:num_recomendaciones + 1]
    #item_indices = [i[0] for i in sim_scores]
    #recomendaciones = df_recomendacion['item_name'].iloc[item_indices].tolist()
    #return recomendaciones

# Ruta para la función recomendacion_juego
#@app.get("/recommendation")
#async def get_recommendation(id_producto: int = Query(..., title="ID del Producto"), num_recomendaciones: int = Query(5, title="Número de Recomendaciones")):
    #recomendaciones = recomendacion_juego(id_producto, num_recomendaciones)
    #return recomendaciones
