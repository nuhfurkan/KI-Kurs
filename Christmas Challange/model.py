import numpy as np
import pandas
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

class Model:
    def __init__(self):
        """
        Hier kannst du deine Modelle initialisieren. Du musst dein Modell
        als attribute der Klasse speichern, damit du es später trainieren kannst
        und dann mit dem trainierten Modell vorhersagen treffen kannst.
        """
        self.dtree = DecisionTreeClassifier()
        ...

    def fit(self, X: np.array, y: np.array) -> None:
        """Diese Methode wird aufgerufen, um das Modell zu trainieren.

        Parameters
        ----------
        X : np.array
            Die Daten, die das Modell trainieren soll.
        y : np.array
            Die Zielwerte, die das Modell trainieren soll.
        """
        self.dtree = self.dtree.fit(X, y)
        ...

    def predict(self, X: np.array) -> np.array:
        """Diese Methode wird aufgerufen, um das Modell zu testen.

        Parameters
        ----------
        X : np.array
            Die Daten, die das Modell testen soll.

        Returns
        -------
        np.array
            Die vorhergesagten Werte.
        """
        return self.dtree.predict(X)

