
import os
import csv



#funció que crea els arxius d'excel per poder processar el dataset per l'entrenament amb triplets
with open('Dataset_dvd2.csv','w')as t:
    writer=csv.writer(t)
    header=['Image_name','subject','pose','glasses','beard','mustache'] #Capçalera, nom de les columnes
    writer.writerow(header)
    datadir='colorferet/dvd2/data/ground_truths/name_value' #Directori on buscar els arxius


    list_dir= sorted(os.listdir(datadir))  #Crea una llista ordenada de les carpetes de dins del directori per poder després obtenir les labels
    if '.DS_Store' in list_dir:
        list_dir.remove('.DS_Store')
    img_labels=[]
    for path,dirs,files in os.walk(datadir):
        for filename in files:
            if filename == '.DS_Store':
                continue
            img_labels.append(os.path.join(path,filename)) #crea una llista amb el nom dels arxius que hi ha dins de cada carpeta del directori

    
    
    for img_label in img_labels: #obté la informació i afegeix-la a l'excel
        with open(img_label,'r') as f:
            txt_lines=f.readlines()
            txt_lines=[e.replace('\n','') for e in txt_lines]
            txt_lines=[e.replace('cfrS','') for e in txt_lines]
            txt_lines=[e.split('=') for e in txt_lines]
            row=[]
            dvd=None
            for e in txt_lines:
                
                if e[0] in header or e[0]=='relative':

                    if e[0]=='relative':
                        
                        e[1]=os.path.join(f'colorferet/dvd{dvd}',e[1].replace('.bz2',''))
                        row.insert(0,e[1])
                    else:
                        if e[0]=='subject':
                            row.append(list_dir.index(e[1]))
                        else:
                            row.append(e[1])
                dvd=e[1] 

            if len(row)!=0:
                writer.writerow(row)
            
                    
                

            
            




