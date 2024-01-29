import numpy as np
import time
from PIL import Image
from anytree import Node, RenderTree, PreOrderIter, AsciiStyle
import sys
import getopt
import get_options

# Cette fonction est un code modifié du cours qui se trouve sur: https://github.com/gabilodeau/INF8770/blob/master/Codage%20LZW.ipynb
def compress_Huffman(message: str, returns_statistics=False) -> dict:    
    compress_results = {}

    # Liste qui sera modifié jusqu'à ce qu'elle contienne seulement la racine de l'arbre
    ArbreSymb =[[message[0], message.count(message[0]), Node(message[0])]] 
    # dictionnaire obtenu à partir de l'arbre.
    dictionnaire = [[message[0], '']]
    nbsymboles = 1
    
    # Recherche des feuilles de l'arbre
    for i in range(1,len(message)):
        if not list(filter(lambda x: x[0] == message[i], ArbreSymb)):
            ArbreSymb += [[message[i], message.count(message[i]), Node(message[i])]]
            dictionnaire += [[message[i], '']]
            nbsymboles += 1        

    ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])
    
    while len(ArbreSymb) > 1:
        # Fusion des noeuds de poids plus faibles
        symbfusionnes = ArbreSymb[0][0] + ArbreSymb[1][0]

        # Création d'un nouveau noeud
        noeud = Node(symbfusionnes)
        temp = [symbfusionnes, ArbreSymb[0][1] + ArbreSymb[1][1], noeud]

        # Ajustement de l'arbre pour connecter le nouveau avec ses parents 
        ArbreSymb[0][2].parent = noeud
        ArbreSymb[1][2].parent = noeud

        # Enlève les noeuds fusionnés de la liste de noeud à fusionner.
        del ArbreSymb[0:2]

        # Ajout du nouveau noeud à la liste et tri.
        ArbreSymb += [temp]
        
        # Pour affichage de l'arbre ou des sous-branches
        #print('\nArbre actuel:\n\n')
        
        ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])  

    ArbreCodes = Node('')
    noeud = ArbreCodes
    
    parcoursprefix = [node for node in PreOrderIter(ArbreSymb[0][2])]
    parcoursprefix = parcoursprefix[1:len(parcoursprefix)] #ignore la racine

    Prevdepth = 0  # pour suivre les mouvements en profondeur dans l'arbre
    for node in parcoursprefix:  # Liste des noeuds 
        if Prevdepth < node.depth:  # On va plus profond dans l'arbre, on met un 0
            temp = Node(noeud.name + '0')
            noeud.children = [temp]
            if node.children:  # On avance le "pointeur" noeud si le noeud ajouté a des enfants.
                noeud = temp
        elif Prevdepth == node.depth:  # Même profondeur, autre feuille, on met un 1
            temp = Node(noeud.name + '1')
            noeud.children = [noeud.children[0], temp]  # Ajoute le deuxième enfant
            if node.children:  # On avance le "pointeur" noeud si le noeud ajouté a des enfants.
                noeud = temp
        else:
            for i in range(Prevdepth-node.depth):  # On prend une autre branche, donc on met un 1
                noeud = noeud.parent  # On remontre dans l'arbre pour prendre la prochaine branche non explorée.
            temp = Node(noeud.name + '1')
            noeud.children = [noeud.children[0], temp]
            if node.children:
                noeud = temp

        Prevdepth = node.depth    

    #print('\nArbre des codes:\n\n',RenderTree(ArbreCodes, style=AsciiStyle()).by_attr())         
    #print('\nArbre des symboles:\n\n', RenderTree(ArbreSymb[0][2], style=AsciiStyle()).by_attr()) 
    ArbreSymbList = [node for node in PreOrderIter(ArbreSymb[0][2])]
    ArbreCodeList = [node for node in PreOrderIter(ArbreCodes)]

    for i in range(len(ArbreSymbList)):
        if ArbreSymbList[i].is_leaf:  # Génère des codes pour les feuilles seulement
            temp = list(filter(lambda x: x[0] == ArbreSymbList[i].name, dictionnaire))
            if temp:
                indice = dictionnaire.index(temp[0])
                dictionnaire[indice][1] = ArbreCodeList[i].name
 
    message_code = []
    longueur = 0 
    for i in range(len(message)):
        substitution = list(filter(lambda x: x[0] == message[i], dictionnaire))
        message_code += [substitution[0][1]]
        longueur += max(1,len(substitution[0][1])) 
    
    if returns_statistics:
        longueur_originale = max(1, np.ceil(np.log2(nbsymboles))) * len(message)
        compress_results["longueur_originale"] = longueur_originale
        compress_results["longueur_compressee"] = longueur
        compress_results["taux_compression"] = round(1 - longueur / longueur_originale, 4) 

    compress_results["message_code"] = message_code
    compress_results["dictionnaire"] = dictionnaire

    return compress_results
    

def compress_txt_Huffman(filenumber: str, returns_statistics=False):

    message = ""
    with open("./data/textes/texte_"+ filenumber +".txt") as text:
        for line in text.readlines():
            message += line
    
    return compress_Huffman(message, returns_statistics)


def compress_img_Huffman(filenumber: str, returns_statistics=False):
    """Takes an image indexed by a number and flattens its structure into a single string then calls the compress function"""
    input_image = Image.open(f'./data/images/image_{filenumber}.png') 
    message = ""
    num_bands = input_image.getbands()
    pix_data = list(input_image.getdata())
    message = ''.join(list(map(lambda x: chr(x+1), pix_data if len(num_bands)==1 else [x for sets in pix_data for x in sets])))
    
    return compress_Huffman(message, returns_statistics)


if __name__ == "__main__":
    return_statistics, *opts = get_options.get_options_from_cmd(sys.argv)
    with open("./Huffman_results.txt", 'w') as Huffman_results:
        for i in range(1, 2):
            start = time.perf_counter()
            compression_results = compress_txt_Huffman(str(i), return_statistics)
            end = time.perf_counter()
            Huffman_results.write("Texte: " + str(i) + "\n")
            Huffman_results.write("Message code: " + str(compression_results["message_code"]) + "\n")
            if return_statistics: 
                Huffman_results.write("Longueur originale: " + str(compression_results["longueur_originale"]) + "\n")
                Huffman_results.write("Longueur compressee: " + str(compression_results["longueur_compressee"]) + "\n")
                Huffman_results.write("Taux de compression: " + str(compression_results["taux_compression"]) + "\n")
                Huffman_results.write("Temps de compression: " + str((end - start)) + "\n")
                Huffman_results.write("\n")

            start = time.perf_counter()
            compression_results = compress_img_Huffman(str(i), return_statistics)
            end = time.perf_counter()

            Huffman_results.write("Image: " + str(i) + "\n")
            Huffman_results.write("Image code: " + str(compression_results["message_code"]) + "\n")
            
            if return_statistics:
                Huffman_results.write("Longueur originale: " + str(compression_results["longueur_originale"]) + "\n")
                Huffman_results.write("Longueur compressee: " + str(compression_results["longueur_compressee"]) + "\n")
                Huffman_results.write("Taux de compression: " + str(compression_results["taux_compression"]) + "\n")
                Huffman_results.write("Temps de compression: " + str((end - start)) + "\n")
                Huffman_results.write("\n")
