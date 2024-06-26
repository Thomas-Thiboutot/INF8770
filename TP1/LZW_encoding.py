import time
import numpy as np
from PIL import Image
import get_options
import sys


# Cette fonction est un code modifié du cours qui se trouve sur: https://github.com/gabilodeau/INF8770/blob/master/Codage%20Huffman.ipynb
def compress_LZW(message: str, returns_statistics: bool) -> dict:
    compress_result = {}

    dictsymb = [message[0]]
    dictbin = ["{:b}".format(0)]
    nbsymboles = 1

    for i in range(1, len(message)):
        if message[i] not in dictsymb:
            dictsymb += [message[i]]
            dictbin += ["{:b}".format(nbsymboles)] 
            nbsymboles += 1
            
    longueur_originale = max(1, np.ceil(np.log2(nbsymboles))) * len(message)   
    
    for i in range(nbsymboles):
        dictbin[i] = "{:b}".format(i).zfill(int(np.ceil(np.log2(nbsymboles))))
    dictsymb.sort()
    compress_result["dictionnaire_initial"] = np.transpose([dictsymb, dictbin])
    
    i = 0
    message_code = []
    longueur = 0
    
    while i < len(message):
        precsouschaine = message[i]  # sous-chaine qui sera codé
        souschaine = message[i]  # sous-chaine qui sera codé + 1 caractère (pour le dictionnaire)
        
        # Cherche la plus grande sous-chaine. On ajoute un caractère au fur et à mesure.
        while souschaine in dictsymb and i < len(message):
            i += 1
            precsouschaine = souschaine
            if i < len(message):  # Si on a pas atteint la fin du message
                souschaine += message[i]
          
        # Codage de la plus grande sous-chaine à l'aide du dictionnaire  
        codebinaire = [dictbin[dictsymb.index(precsouschaine)]]
        message_code += codebinaire
        longueur += len(codebinaire[0])

        # Ajout de la sous-chaine codé + symbole suivant dans le dictionnaire.
        if i < len(message):
            dictsymb += [souschaine]
            dictbin += ["{:b}".format(nbsymboles)] 
            nbsymboles += 1

        # Ajout de 1 bit si requis
        if np.ceil(np.log2(nbsymboles)) > len(message_code[-1]):
            for j in range(nbsymboles):
                dictbin[j] = "{:b}".format(j).zfill(int(np.ceil(np.log2(nbsymboles))))
    
    compress_result["message_code"] = message_code

    if returns_statistics:
        compress_result["dictionnaire_final"] = np.transpose([dictsymb, dictbin])
        compress_result["longueur_originale"] = longueur_originale
        compress_result["longueur_compressee"] = longueur        
        compress_result["taux_compression"] = round(1 - longueur / longueur_originale, 4) 

    return compress_result


def compress_txt_LZW(filenumber: str, returns_statistics: bool):
    message = "" 
    with open("./data/textes/texte_"+ filenumber +".txt") as text:
        for line in text.readlines():
            message += line

    return compress_LZW(message, returns_statistics)       


def compress_img_LZW(filenumber: str, returns_statistics: bool):
    input_image = Image.open(f'./data/images/image_{filenumber}.png') 
    num_bands = input_image.getbands()
    pix_data = list(input_image.getdata())
    message = ''.join(list(map(lambda x: chr(x+1), pix_data if len(num_bands)==1 else [x for sets in pix_data for x in sets])))
    
    return compress_LZW(message, returns_statistics)


if __name__ == "__main__":
    return_statistics: bool = get_options.get_options_from_cmd(sys.argv)["return_statistics"]

    with open("./LZW_results.txt", 'w', encoding="utf-8") as LZW_results:
        for i in range(1, 6):
            start = time.perf_counter()
            compression_results = compress_txt_LZW(str(i), return_statistics)
            end = time.perf_counter()

            LZW_results.write("Texte: " + str(i) + "\n")
            LZW_results.write("Message code: " + str(compression_results["message_code"]) + "\n")
            LZW_results.write("Dictionnaire initial: " + str(compression_results["dictionnaire_initial"]) + "\n")

            if return_statistics:
                LZW_results.write("Longueur originale: " + str(compression_results["longueur_originale"]) + "\n")
                LZW_results.write("Longueur compressee: " + str(compression_results["longueur_compressee"]) + "\n")
                LZW_results.write("Taux de compression: " + str(compression_results["taux_compression"]) + "\n")
                LZW_results.write("Temps de compression: " + str((end - start)) + "\n")
                LZW_results.write("\n")

            start = time.perf_counter()
            compression_results = compress_img_LZW(str(i), return_statistics)
            end = time.perf_counter()

            LZW_results.write("Image: " + str(i) + "\n")
            LZW_results.write("Message code: " + str(compression_results["message_code"]) + "\n")
            LZW_results.write("Dictionnaire initial: " + str(compression_results["dictionnaire_initial"]) + "\n")

            if return_statistics:
                LZW_results.write("Longueur originale: " + str(compression_results["longueur_originale"]) + "\n")
                LZW_results.write("Longueur compressee: " + str(compression_results["longueur_compressee"]) + "\n")
                LZW_results.write("Taux de compression: " + str(compression_results["taux_compression"]) + "\n")
                LZW_results.write("Temps de compression: " + str((end - start)) + "\n")
                LZW_results.write("\n")
