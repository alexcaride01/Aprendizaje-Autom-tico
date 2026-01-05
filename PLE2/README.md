# PLE2 – Modelado Predictivo de Calorías Quemadas en un Gimnasio

**Autor**: Alex Caride Cid
**Asignatura**: Aprendizaje Automático

## Objetivo de la práctica

El objetivo principal de la práctica es desarrollar un modelo predictivo capaz de estimar las calorías quemadas por los usuarios de un gimnasio, a partir de variables fisiológicas y hábitos de entrenamiento.El trabajo lo he  estructurado en cuatro fases que representan el ciclo completo de un proyecto de aprendizaje automático: preprocesamiento, análisis exploratorio, modelado y evaluación.

## Estructura del proyecto

El proyecto se compone de cuatro notebooks principales, junto con los archivos de datos y el modelo entrenado:

  **PLE2_eda.ipynb**  
  Incluye el análisis exploratorio de datos (EDA), la obtención de estadísticas descriptivas, la visualización de distribuciones y relaciones entre variables.

 **PLE2_pre1_preprocesamiento.ipynb**  
  Contiene las tareas de limpieza y preparación del dataset, eliminación de variables redundantes y codificación de variables categóricas. El resultado se guarda en un nuevo archivo CSV preprocesado.

 **PLE2_pre2_modelado.ipynb**  
  Desarrolla y compara distintos modelos de regresión (Regresión Lineal, Árbol de Decisión y SVR). Se realiza validación cruzada y búsqueda de hiperparámetros mediante GridSearchCV, seleccionando el modelo con mejor rendimiento.

 **PLE2_pre3_evaluacion.ipynb**  
  Evalúa el modelo final sobre el conjunto de test, calcula métricas de desempeño (R², MSE, MAPE) y añade una nueva observación al dataset para realizar una inferencia completa. También se guarda el modelo final entrenado en formato .pkl.

Archivos complementarios:

 - ple2_gimnasio.csv: dataset original.  
 - ple2_gimnasio_preprocesado.csv: dataset limpio y preparado.  
 - ple2_gimnasio_modelo.pkl: modelo final entrenado.


## Resultados y Conclusiones

El modelo final que he seleccionado fue la Regresión Lineal, que logró explicar aproximadamente el 98 % de la variabilidad de las calorías quemadas. En el conjunto de entrenamiento obtuvo un coeficiente de determinación R² de 0.9787, y en el conjunto de prueba un valor de 0.9797, lo que demuestra una excelente capacidad de generalización y la ausencia de sobreajuste. Los errores medios fueron bajos, con un MSE de 1612 y un MAPE aproximado del 3 %, lo que indica que las predicciones del modelo son bastantes precisas en relación con los valores reales.

 

