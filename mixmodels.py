#Plan: fonction d'entrée mixmodels avec dictionnaire params: X, y, [{'model': model1, 'weight': weight1}, {'model': model2, 'weight': weight2}, etc.]
# mixmodels produit les prédictions associées à chaque modèle, et transmet les params: y, [{'prediction': prediction1, 'weight': weight1}, {...}, etc.]
# mixpredictions qui, dans le cas d'une classification, calcule la 'longueur' associée à chaque item y apparaissant (dans le cas d'une régression: moyenne
# pondérée des estimations). L'tem ayant la plus grande longueur est pris comme prédiction, qui est comparée à y (critère: précision)
