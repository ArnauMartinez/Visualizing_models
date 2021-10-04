import os
import bz2
import shutil



directory='/Users/arnaumartinez/Desktop/TR/colorferet/dvd2/data/images'
for path, dirs, files in os.walk(directory):
    for filename in files:
        basename, ext = os.path.splitext(filename)
        if ext.lower() != '.bz2':
            continue
        fullname = os.path.join(path, filename)
        newname = os.path.join(path, basename)
        with bz2.open(fullname) as fh, open(newname, 'wb') as fw:
            shutil.copyfileobj(fh, fw)
        os.remove(fullname)

for path, dirs, files in os.walk(directory):
    for filename in files:
        basename, ext = os.path.splitext(filename)
        if ext.lower() == '.bz2':
            fullname=os.path.join(path, filename)
            os.remove(fullname)
       

