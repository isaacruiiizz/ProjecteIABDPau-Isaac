import os
import glob

def fix_labels_local():
    # Busca recursivament totes les carpetes que es diguin 'labels'
    labels_folders = glob.glob('**/labels', recursive=True)
    
    if not labels_folders:
        print("❌ No s'ha trobat cap carpeta 'labels'. Assegura't d'executar el script a la carpeta correcta.")
        return

    for folder in labels_folders:
        print(f"📂 Processant carpeta: {folder}")
        fixed_count = 0
        
        for filename in os.listdir(folder):
            if filename.endswith('.txt') and filename != 'classes.txt':
                file_path = os.path.join(folder, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                new_lines = []
                modified = False
                for line in lines:
                    # Si comença amb '2 ', ho canviem per '0 ' (other)
                    if line.startswith('2 '):
                        new_lines.append('0 ' + line[2:])
                        modified = True
                    else:
                        new_lines.append(line)
                
                if modified:
                    with open(file_path, 'w', encoding='utf-8', newline='') as f:
                        f.writelines(new_lines)
                    fixed_count += 1
        
        print(f"✅ S'han corregit {fixed_count} fitxers .txt a {folder}")

    # Esborrem el cache de l'Ultralytics si existeix perquè rellegeixi el dataset net
    cache_files = glob.glob('**/labels.cache', recursive=True)
    for cache in cache_files:
        try:
            os.remove(cache)
            print(f"🗑️ Esborrat fitxer de cache vell: {cache}")
        except Exception as e:
            print(f"⚠️ No s'ha pogut esborrar el cache {cache}: {e}")

if __name__ == "__main__":
    fix_labels_local()