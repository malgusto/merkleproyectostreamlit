#Importaciones necesarias
import streamlit as st
import joblib 
import pandas as pd
from sklearn.cluster import KMeans
import spacy
from goose3 import Goose
from sklearn.cluster import KMeans

# from capturescreenshot import *
st.set_page_config(page_title="Clasificador de noticias | Proyecto Merkle",page_icon="./assets/favicon.png")

@st.cache_data
def cargar_modelo_kmeans():
    vectorizadorKmeans = joblib.load("./models/vectorizer_kmeans")
    modeloKmeans = joblib.load("./models/modelo_kmeans")
    categoriasKmeans = joblib.load("./models/categorias-k-means")
    return vectorizadorKmeans, modeloKmeans, categoriasKmeans

@st.cache_data
def cargar_modelo_lda_gensim():
    modeloldag = joblib.load("./models/Mejor_modelo_LDA_GENSIM")
    categoriasldag = joblib.load("./models/Categorias_mejor_modelo_LDA_GENSIM")
    diccionariolda = joblib.load("./models/Diccionario_LDA_GENSIM")
    return modeloldag, categoriasldag, diccionariolda

@st.cache_data
def cargar_modelo_lda_sklearn():
    vectorizerldask = joblib.load("./models/vectorizerldask")
    modeloldask = joblib.load("./models/Mejor_modelo_LDA_SKLEARN")
    categoriasldask = joblib.load("./models/Categorias_mejor_modelo_LDA_SKLEARN")
    return vectorizerldask, modeloldask, categoriasldask

@st.cache_data
def cargar_modelo_lsa_gensim():
    modelolsag = joblib.load("./models/Mejor_modelo_LSA_GENSIM")
    categoriaslsag = joblib.load("./models/Categorias_mejor_modelo_LSA_GENSIM")
    diccionariolsag = joblib.load("./models/Diccionario_LSA_GENSIM")
    return modelolsag, categoriaslsag, diccionariolsag

# Cargar los modelos una vez al inicio de la aplicación
vectorizadorKmeans, modeloKmeans, categoriasKmeans = cargar_modelo_kmeans()
modeloldag, categoriasldag, diccionariolda = cargar_modelo_lda_gensim()
vectorizerldask, modeloldask, categoriasldask = cargar_modelo_lda_sklearn()
modelolsag, categoriaslsag, diccionariolsag = cargar_modelo_lsa_gensim()

# Función que predice la noticia
def predecir_articulo(noticia):

  nlp = spacy.load('es_core_news_sm', disable=['parser', 'senter', 'ner', 'attribute_ruler'])

  def limpiar(doc):
      doc_procesado = []
      for token in nlp(doc.lower()):
          # Filtrar los tokens
          if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_space and not token.is_punct and not token.is_stop and not token.like_num and not token.like_email:
              # Si es un sustantivo, verbo, adjetivo, adverbio o conjunción y no signos de puntuación, ni es un número o email, añadir el lexema de la palabra al documento procesado
              doc_procesado.append(token.lemma_)
      return doc_procesado
    
  g=Goose()
  articulo=g.extract(url=noticia)
  titulo_noticia = articulo.title
  #Extraemos el contenido de la pagina
  nueva_noticia_limpiada = articulo.cleaned_text
  limpiada= limpiar(nueva_noticia_limpiada.lower())

  #Datos a sacar sobre el articulo
  numparrafos = len(articulo.cleaned_text.split("\n"))
  numeropalabras = len(articulo.cleaned_text.split())
  numerofrases = len(articulo.cleaned_text.split("."))
  
  sobre_texto=f"El artículo consta de un total de {numparrafos} párrafos, {numeropalabras} palabras y {numerofrases} frases."  
  

  #############
  # K-Means
  #############    
  # Vectorizar nueva noticia
  limpiada_kmeans=" ".join(limpiada)
  nuevas_noticias_vectorizadas=vectorizadorKmeans.transform([limpiada_kmeans])
  # Predecir
  cluster_predicho = modeloKmeans.predict(nuevas_noticias_vectorizadas)
  etikmeans=categoriasKmeans[cluster_predicho[0]]
  
  #############
  # LDA-Gensim
  #############    
  corpus = diccionariolda.doc2bow(limpiada)
  # Categoriza la nueva noticia usando el modelo LDA
  categorias_noticia = modeloldag.get_document_topics(corpus)
  tema_principal = max(categorias_noticia, key=lambda x: x[1])[0]
  ### HE tenido que poner esto porque da un out of range, ni idea porque
  if tema_principal > 0:
     etildag = categoriasldag[tema_principal - 1]
  else:
     etildag = categoriasldag[tema_principal]

  
  #############
  # LDA-SKL
  #############    
  noticia_procesadaldask = " ".join(limpiada)
  noticia_vectorizada = vectorizerldask.transform([noticia_procesadaldask])

  # Transformación del artículo con el modelo LDA
  distribucion_topicos = modeloldask.transform(noticia_vectorizada)

  # Identificación del tópico dominante
  topico_dominante = distribucion_topicos.argmax(axis=1)[0]
  etildask=categoriasldask[topico_dominante]
  
  #############
  # LSA-Gensim
  #############
  # Convierte el texto preprocesado en el formato del corpus
  corpus = diccionariolsag.doc2bow(limpiada)

  # Categoriza la nueva noticia usando el modelo LDA
  lsa_vector = modelolsag[corpus]
  #   ''' Enlazarlo con las categorías '''
  topico_dominante = max(lsa_vector, key=lambda x: x[1])[0]

  etilsag=categoriaslsag[topico_dominante]
  
  
    
  # url_screenshot="http://127.0.0.1:8000/"+captscreehshot(noticia)
    

  datos_web=[titulo_noticia,etikmeans,etildag,etildask,etilsag,sobre_texto]
  
  return datos_web


# UI
left_co, cent_co,last_co = st.columns([1,4,1])
with cent_co:
    st.image('./assets/merkle-logo.png',width=350)

   
st.header("¿En qué consiste el proyecto?")
st.markdown("Este es un proyecto realizado por un grupo de alumnos del ***Curso de especializacion de Big Data e Inteligencia Artificial*** del ***I.E.S. Doctor Fleming***. En el proyecto tenemos que crear un modelo que clasifique noticias a partir de un dataframe de 50.000 noticias.")

st.header("Categorizar noticia:")
url = st.text_input("Introduzca la URL del artículo:", "")

left_co1, cent_co1,last_co1 = st.columns([1,4,1])
with last_co1:
    botonpredecir=st.button("Predecir")
 
    
if botonpredecir:
    if url:
        result = predecir_articulo(url)
        st.subheader("Resultados:")
        st.write(f"**Título:** {result[0]}")
        st.write(f"**Etiqueta K-Means:** {result[1]}")
        st.write(f"**Etiqueta LDA-Gensim:** {result[2]}")
        st.write(f"**Etiqueta LDA-SKL:** {result[3]}")
        st.write(f"**Etiqueta LSA-Gensim:** {result[4]}")
    else:
        st.error("Por favor, introduzca una URL válida.")