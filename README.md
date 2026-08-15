# Detección de Violencia de Género en Redes Sociales con NLP 

Este repositorio contiene la implementación de un modelo de Clasificación de Texto basado en Procesamiento de Lenguaje Natural (PLN) y aprendizaje profundo, diseñado para detectar y clasificar violencia por razón de género en plataformas sociales.

## Descripción del Proyecto
La violencia digital es una problemática creciente. Este proyecto (desarrollado como Trabajo para titulación) busca aplicar Inteligencia Artificial para identificar automáticamente discursos de odio y acoso enfocados en género. Utiliza arquitecturas avanzadas de lenguaje para entender el contexto y la semántica de los mensajes.

## Tecnologías Utilizadas
* **Lenguaje:** Python
* **Librerías de Deep Learning:** PyTorch / TensorFlow 
* **Modelos:** Arquitecturas basadas en Transformers (ej. BERT, RoBERTa)
* **Manipulación de datos:** Pandas, NumPy, Scikit-learn

## Arquitectura y Metodología
El proyecto sigue un pipeline estructurado de Ciencia de Datos enfocado en Procesamiento de Lenguaje Natural:

1. **Recopilación de Datos:** Identificación de fuentes e integración de los conjuntos de datos.
2. **Análisis Exploratorio de Datos (EDA):** Visualización de nubes de palabras y análisis de las palabras más comunes por clase para entender la distribución del vocabulario.
3. **División de los Datos:** Separación de los datos en conjuntos de Entrenamiento (Train), Prueba (Test) y Validación.
4. **Balanceo de Datos:** Aplicación de técnicas para equilibrar las clases, incluyendo Easy Data Augmentation y técnicas de submuestreo (Near Miss, Cluster Centroids y PSC).
5. **Preprocesamiento:** Limpieza textual exhaustiva, normalización y tokenización de los comentarios.
7. **Entrenamiento y Evaluación:** 
   * *Fase 1 (Rendimiento):* Evaluación de los modelos utilizando métricas de Accuracy, F1-score y Recall.
   * *Fase 2 (Aplicación):* Detección directa de comentarios con violencia de género.
8. **Implementación del Modelo:** Búsqueda de modelos de clasificación, definición e implementación del modelo final óptimo.

## Cómo ejecutar este proyecto

### Requisitos previos
Es necesario tener Python 3.x instalado. Se recomienda usar un entorno virtual.

### Instalación
Clona este repositorio e instala las dependencias:
```bash
git clone [https://github.com/TifannyMtz/deteccion-violencia-genero-nlp.git](https://github.com/TifannyMtz/deteccion-violencia-genero-nlp.git)
cd deteccion-violencia-genero-nlp
pip install -r requirements.txt
