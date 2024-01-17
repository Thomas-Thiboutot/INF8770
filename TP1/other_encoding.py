import numpy as np
import time
from anytree import Node, RenderTree, PreOrderIter, AsciiStyle

def compress_txt_Huffman(filenumber: int):
    print(f'fichier: {filenumber}')
    Message = ""
    f = open("./data/textes/texte_"+ filenumber +".txt")
    for line in f.readlines():
        Message += line
        
    #Liste qui sera modifié jusqu'à ce qu'elle contienne seulement la racine de l'arbre
    ArbreSymb =[[Message[0], Message.count(Message[0]), Node(Message[0])]] 
    #dictionnaire obtenu à partir de l'arbre.
    dictionnaire = [[Message[0], '']]
    nbsymboles = 1
    """Compression begins"""
    start = time.perf_counter()
    #Recherche des feuilles de l'arbre
    for i in range(1,len(Message)):
        if not list(filter(lambda x: x[0] == Message[i], ArbreSymb)):
            ArbreSymb += [[Message[i], Message.count(Message[i]),Node(Message[i])]]
            dictionnaire += [[Message[i], '']]
            nbsymboles += 1

    longueurOriginale = np.ceil(np.log2(nbsymboles))*len(Message)    
    ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])
    #print(ArbreSymb)
    while len(ArbreSymb) > 1:
        #Fusion des noeuds de poids plus faibles
        symbfusionnes = ArbreSymb[0][0] + ArbreSymb[1][0] 
        #Création d'un nouveau noeud
        noeud = Node(symbfusionnes)
        temp = [symbfusionnes, ArbreSymb[0][1] + ArbreSymb[1][1], noeud]
        #Ajustement de l'arbre pour connecter le nouveau avec ses parents 
        ArbreSymb[0][2].parent = noeud
        ArbreSymb[1][2].parent = noeud
        #Enlève les noeuds fusionnés de la liste de noeud à fusionner.
        del ArbreSymb[0:2]
        #Ajout du nouveau noeud à la liste et tri.
        ArbreSymb += [temp]
        #Pour affichage de l'arbre ou des sous-branches
        #print('\nArbre actuel:\n\n')
        #for i in range(len(ArbreSymb)):
        #    if len(ArbreSymb[i][0]) > 1:
        #        print(RenderTree(ArbreSymb[i][2], style=AsciiStyle()).by_attr())   
        ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])  
        #print(ArbreSymb)
        ArbreCodes = Node('')
    noeud = ArbreCodes
    #print([node.name for node in PreOrderIter(ArbreSymb[0][2])])
    parcoursprefix = [node for node in PreOrderIter(ArbreSymb[0][2])]
    parcoursprefix = parcoursprefix[1:len(parcoursprefix)] #ignore la racine

    Prevdepth = 0 #pour suivre les mouvements en profondeur dans l'arbre
    for node in parcoursprefix:  #Liste des noeuds 
        if Prevdepth < node.depth: #On va plus profond dans l'arbre, on met un 0
            temp = Node(noeud.name + '0')
            noeud.children = [temp]
            if node.children: #On avance le "pointeur" noeud si le noeud ajouté a des enfants.
                noeud = temp
        elif Prevdepth == node.depth: #Même profondeur, autre feuille, on met un 1
            temp = Node(noeud.name + '1')
            noeud.children = [noeud.children[0], temp]  #Ajoute le deuxième enfant
            if node.children: #On avance le "pointeur" noeud si le noeud ajouté a des enfants.
                noeud = temp
        else:
            for i in range(Prevdepth-node.depth): #On prend une autre branche, donc on met un 1
                noeud = noeud.parent #On remontre dans l'arbre pour prendre la prochaine branche non explorée.
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
        if ArbreSymbList[i].is_leaf: #Génère des codes pour les feuilles seulement
            temp = list(filter(lambda x: x[0] == ArbreSymbList[i].name, dictionnaire))
            if temp:
                indice = dictionnaire.index(temp[0])
                dictionnaire[indice][1] = ArbreCodeList[i].name

    #print(dictionnaire) 
    MessageCode = []
    longueur = 0 
    for i in range(len(Message)):
        substitution = list(filter(lambda x: x[0] == Message[i], dictionnaire))
        MessageCode += [substitution[0][1]]
        longueur += len(substitution[0][1]) 
        #print(MessageCode)
    end = time.perf_counter()
    """Compression ends"""
    
    print(f'Longueur = {longueur}')
    print(f'Longueur originale= {longueurOriginale}')
    print(f'Taux de compression =  {round(1-longueur/longueurOriginale, 4)}')
    print(f'Temps de compression= {round((end-start)/(10^6), 6)} ms\n')

if __name__ == "__main__":

    for i in range(1,6):
        compress_txt_Huffman(str(i))
        