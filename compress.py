import os
import zipfile
 
fantasy_zip = zipfile.ZipFile('g.zip', 'w')
 
for folder, subfolders, files in os.walk('out_folder'):
 
    for file in files:
        if file.endswith('.json'):
            fantasy_zip.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder,file), 'out_folder'), compress_type = zipfile.ZIP_DEFLATED)
 
fantasy_zip.close()