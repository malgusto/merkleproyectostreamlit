#Importaciones necesarias
import streamlit as st
import joblib 
import pandas as pd
from sklearn.cluster import KMeans
import spacy, es_core_news_lg
from goose3 import Goose
from sklearn.cluster import KMeans

# from capturescreenshot import *

# Cargando K-Means
vectorizadorKmeans=joblib.load("./models/vectorizer_kmeans")
modeloKmeans=joblib.load('./models/modelo_kmeans')
categoriasKmeans=joblib.load('./models/categorias-k-means')

# Cargando LDA-Gensim
modeloldag = joblib.load('./models/Mejor_modelo_LDA_GENSIM')
categoriasldag = joblib.load('./models/Categorias_mejor_modelo_LDA_GENSIM')
diccionariolda = joblib.load('./models/Diccionario_LDA_GENSIM')

# Cargando LDA-SKL
vectorizerldask=joblib.load('./models/vectorizerldask')
modeloldask = joblib.load('./models/Mejor_modelo_LDA_SKLEARN')
categoriasldask = joblib.load('./models/Categorias_mejor_modelo_LDA_SKLEARN')

#Cargando LSA-Gensim
modelolsag = joblib.load('./models/Mejor_modelo_LSA_GENSIM')
categoriaslsag = joblib.load('./models/Categorias_mejor_modelo_LSA_GENSIM')
diccionariolsag = joblib.load('./models/Diccionario_LSA_GENSIM')


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
  etildag=categoriasldag[tema_principal]
  
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
  ''' Enlazarlo con las categorías '''
  topico_dominante = max(lsa_vector, key=lambda x: x[1])[0]

  etilsag=categoriaslsag[topico_dominante]
  
  
    
  # url_screenshot="http://127.0.0.1:8000/"+captscreehshot(noticia)
    

  datos_web=[titulo_noticia,etikmeans,etildag,etildask,etilsag,sobre_texto]
  
  return datos_web


# UI
st.title("Predicción de Artículos")

url = st.text_input("Introduzca la URL del artículo:", "")

if st.button("Predecir"):
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