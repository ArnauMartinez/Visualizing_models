
import os
import csv









datadir=['colorferet/dvd2/data/ground_truths/name_value','colorferet/dvd2/data/ground_truths/name_value']

list_dir= sorted(os.listdir(datadir))
if '.DS_Store' in list_dir:
    list_dir.remove('.DS_Store')

img_labels=[]
for path,dirs,files in os.walk(datadir):
    for filename in files:
        if filename == '.DS_Store':
            continue
        img_labels.append(os.path.join(path,filename))


header=['Image_name','subject','pose','glasses','beard','mustache']
#with open('Dataset.csv','w') as f:
    #writer=csv.writer(f)
    #writer.writerow(header)
    #for image in img_names:
        #folder=image.split('/')[-2]
        #label=list_dir.index(folder)
        #row=[image,label]
        #writer.writerow(row)



with open('Dataset_k_1.csv','w')as t:
    writer=csv.writer(t)
    header=['Image_name','subject','pose','glasses','beard','mustache']
    writer.writerow(header)
    datadir=['colorferet/dvd1/data/ground_truths/name_value','colorferet/dvd2/data/ground_truths/name_value']
    for e in datadir:

        list_dir= sorted(os.listdir(e))
        if '.DS_Store' in list_dir:
            list_dir.remove('.DS_Store')
        img_labels=[]
        for path,dirs,files in os.walk(e):
            for filename in files:
                if filename == '.DS_Store':
                    continue
                img_labels.append(os.path.join(path,filename))

        
        
        for img_label in img_labels:
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
                            
                            e[1]=os.path.join(f'/kaggle/input/colorferet/Kaggle_dataset_dvd{dvd}/Kaggle_dataset_dvd{dvd}',e[1].replace('.bz2',''))
                            e[1]=e[1].replace('/data','')
                            row.insert(0,e[1])
                        else:
                            if e[0]=='subject':
                                row.append(list_dir.index(e[1]))
                            else:
                                row.append(e[1])
                    dvd=e[1] 

                if len(row)!=0:
                    writer.writerow(row)
            
                    
                

            
            

    for img_label in img_labels:
        with open(img_label,'r') as f:
            txt_lines=f.readlines()
            txt_lines=[e.replace('\n','') for e in txt_lines]
            txt_lines=[e.replace('cfrS','') for e in txt_lines]
            txt_lines=[e.split('=') for e in txt_lines]
            print(txt_lines)
            break



