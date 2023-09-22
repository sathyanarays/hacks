import yaml
import os

def createFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
file = open('cert-manager.yaml', 'r')
cert_manager_docs = yaml.safe_load_all(file)

target_dir = 'cert-manager/resources/'

count = 0
for doc in cert_manager_docs:
    kind = doc['kind']
    kind_folder = target_dir + kind + "/"
    createFolder(kind_folder)

    namespace = "default"
    if "namespace" in doc["metadata"]:
        namespace = doc["metadata"]["namespace"]
    
    namespaced_folder = kind_folder + namespace + "/"
    createFolder(namespaced_folder)

    doc_string = yaml.dump(doc, default_flow_style=False)
    target_file = open(namespaced_folder + doc["metadata"]["name"] + ".yaml", "w")
    target_file.write(doc_string)
    target_file.close()

file.close()

