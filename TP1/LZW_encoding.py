## Inspiré du code du cours 
import time
import numpy as np
from PIL import Image

def compress_text_LZW(filenumber: str, is_text: bool, message: str):

    Message = "" if is_text else message
    
    if is_text:
        f = open("./data/textes/texte_"+ filenumber +".txt")
        for line in f.readlines():
            Message += line
    dictsymb =[Message[0]]
    dictbin = ["{:b}".format(0)]
    nbsymboles = 1
    for i in range(1,len(Message)):
        if Message[i] not in dictsymb:
            dictsymb += [Message[i]]
            dictbin += ["{:b}".format(nbsymboles)] 
            nbsymboles +=1
            
    longueurOriginale = np.ceil(np.log2(nbsymboles))*len(Message) if (nbsymboles != 1) else len(Message)    
    
    for i in range(nbsymboles):
        dictbin[i] = "{:b}".format(i).zfill(int(np.ceil(np.log2(nbsymboles))))
    dictsymb.sort()
    dictionnaire = np.transpose([dictsymb,dictbin])
    #print(dictionnaire) 
    i=0;
    MessageCode = []
    longueur = 0
    start = time.perf_counter()
    while i < len(Message):
        precsouschaine = Message[i] #sous-chaine qui sera codé
        souschaine = Message[i] #sous-chaine qui sera codé + 1 caractère (pour le dictionnaire)
        #Cherche la plus grande sous-chaine. On ajoute un caractère au fur et à mesure.
        while souschaine in dictsymb and i < len(Message):
            i += 1
            precsouschaine = souschaine
            if i < len(Message):  #Si on a pas atteint la fin du message
                souschaine += Message[i]  
        #Codage de la plus grande sous-chaine à l'aide du dictionnaire  
        codebinaire = [dictbin[dictsymb.index(precsouschaine)]]
        MessageCode += codebinaire
        longueur += len(codebinaire[0]) 
        #Ajout de la sous-chaine codé + symbole suivant dans le dictionnaire.
        if i < len(Message):
            dictsymb += [souschaine]
            dictbin += ["{:b}".format(nbsymboles)] 
            nbsymboles +=1
        #Ajout de 1 bit si requis
        if np.ceil(np.log2(nbsymboles)) > len(MessageCode[-1]):
            for j in range(nbsymboles):
                dictbin[j] = "{:b}".format(j).zfill(int(np.ceil(np.log2(nbsymboles))))
    ## print(MessageCode)
    end = time.perf_counter()
    dictionnaire = np.transpose([dictsymb,dictbin])
    #print(dictionnaire) 
    tauxcompression = 1 - longueur/longueurOriginale
    print("Taux de compression :" + str(tauxcompression))
    print("Temps de codage: "+ str(end-start)+'\n')
    print("Longueur = {0}".format(longueur))
    print("Longueur originale = {0}".format(longueurOriginale))

## References: https://www.geeksforgeeks.org/how-to-manipulate-the-pixel-values-of-an-image-using-python/
def compress_img_LZW(filenumber: str):
    print("image numéro: " + filenumber)
    input_image = Image.open("./data/images/image_"+ filenumber +".png") 
    
    num_bands = input_image.getbands()

    width, height = input_image.size 
    message =""
    for i in range(width): 
        for j in range(height): 
            pixel = input_image.getpixel((i, j)) 
            if len(num_bands) == 1:
                pixel = intToAscii(pixel)
            else:
                pixel = list(map(intToAscii, pixel))
            
            for i in range(0, len(pixel)):
                message += pixel[i]
    compress_text_LZW(filenumber, False, message)
     
    ## print(len(message))      
def intToAscii(number):
    return chr(number)

if __name__ == "__main__":
    ##for i in range(1,6):
    ##    compress_text_LZW(str(i),True, "")
    for i in range(1,6):
        compress_img_LZW(str(i))
        
