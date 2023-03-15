# **TD1 : Classification**

_**Source :**_

* [Cours TD1](https://www.youtube.com/watch?v=5_qrxVq1kvc&ab_channel=AlfredoCanziani)
* [SVD](https://www.youtube.com/watch?v=mBcLRGuAFUk&ab_channel=MITOpenCourseWare)

Imaginons que l'on prenne une image d'un 1 mégapixel (1M de pixels).

Dimension de l'image : 1000 x 1000 x 3 (1000 pixels de long x 1000 pixels de large x la couleur).
Nous allons donc avoir 3M valeurs et donc 3M de dimensions, ce qui est énorme.

Prenons 2 photos : une d'un chien et une d'un chat.
Si l'une des photos est dans un point de l'espace, il faut déplacer l'autre pour qu'elles soient différenciables et non
superposées.
Et pour faire cela, on bouge un point avec l'algèbre linéaire, notamment avec la multiplication matricielle qui permet
de faire des transformation linéaire comme : la rotation, l'étrirement (scaling ou changement d'échelle horizontale ou
verticale), réflexion (inversion), la translation, la transformation affine.
La translation - qui permet de déplacer des choses - est-il une opération linéaire ? OUI ! On a donc besoin de la
translation pour déplacer un point.

Nous avons donc un espace où nous avons tous nos points.
- d'abord, on veut l'abaisser avec une translation
- puis on veut l'agrandir, pour cela on va zoomer et donc faire un changement d'échelle (scaling). Pour se faire, nous
avons besoin d'une matrice diagonale, on va donc diagonaliser notre matrice de points.

Maintenant, comment procéder à la classification ? On veut donc essentiellement déplacer ces points dans différentes
régions.

Regardons comment un NeuralNet fait une classification.
Nous commençons par un plan en 2D avec des points de coordonnées X et Y dispatchés en 5 branches sous forme de spirale.
La couleur représente leur classe, qui représente une 3ème dimension.
Le réseau va avoir ce plan mais sans les couleurs, on va donc lui demander de séparer nos points par classe.

<p align="center"><img height="200" src="C:\Users\sarah\PycharmProjects\Cours-de-Yann_LeCun\images\classification.PNG" title="Spirale" width="300"/></p> 

Le réseau va donc étirer l'espace jusqu'à ce que tous les points qui appartiennent à la même couleur/classe se retrouvent
dans le même sous-espace.
Une fois la convergence atteinte, les points des classes sont linéairement séparables. On peut donc ensuite utiliser la
régression logistique ou une régression "un-contre-tous".

**La dernière matrice est représentée par ces 5 flèches.**
La dimension de la **_sortie_** de notre réseau est de **5** car nous avons 5 classes. La matrice aura donc **5 lignes**
et **2** colonnes car nous nous partons d'un espace à 2D (ce seront en quelques sortes les deux coordonnées de la pointe
de la flèche).

###### **_Pourquoi avons-nous besoin de matrices et de non linéarité pour avoir ce résultat ?_** 

Ici, nous avons un réseau à 2 matrices : 

- la première matrice correspond à l'entrée (input) à 2 dimensions


```mermaid
A --> B
C --> B
```



```python
print("\ntest")

```

