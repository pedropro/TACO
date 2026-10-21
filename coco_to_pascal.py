import json
import xml.etree.ElementTree as ET
import os
local_path  = os.path.dirname(os.path.abspath(__file__))

with open("annotations.json") as json_file:
    annotations = json.load(json_file)
    nr_images = len(annotations['images'])
    past_fname = ""    
    
    for i in range(len(annotations['annotations']) - 1):
        annotation = annotations['annotations'][i]
        image_id = annotation.get("image_id")
        image = annotations['images'][image_id]
        path = image['file_name']
        file_name = path.split('/')
        next_id = annotations['annotations'][i+1].get(("image_id"))
        folder = file_name[0]
        file_name = file_name[1]
        width = image.get("width")
        height = image.get("height")
        xmin = int(annotation.get("bbox")[0])
        ymin  = int(annotation.get("bbox")[1])
        xmax = xmin + int(annotation.get("bbox")[2])
        ymax  = ymin + int(annotation.get("bbox")[3])
        categoria = annotation.get("category_id")
        cat = annotations["categories"]
        
        if(past_fname != file_name ):
            mydoc = ET.parse("base.xml")
            root = mydoc.getroot()
            mydoc.find('folder').text = os.path.join(local_path,path)
            mydoc.find('filename').text = file_name
            objt = mydoc.find('object')
            objt.find("name").text = cat[categoria].get("supercategory")
            bndbox = objt.find('bndbox')
            bndbox.find("xmin").text = str(xmin)
            bndbox.find("ymin").text = str(ymin)
            bndbox.find("xmax").text = str(xmax)
            bndbox.find("ymax").text = str(ymax)
                            
        elif past_fname == file_name:
            xml = mydoc.getroot()
            objt = ET.SubElement(xml, 'object')
            name = ET.SubElement(objt, 'name')
            name.text = cat[categoria].get("supercategory")
            pose = ET.SubElement(objt, 'pose')
            pose.text = "Unspecified"
            truncated = ET.SubElement(objt, 'truncated')
            truncated.text = "0"
            difficult = ET.SubElement(objt, 'difficult')
            difficult.text = "0"
            tree = ET.ElementTree(objt)
            bndbox = ET.SubElement(objt, 'bndbox')
            x_min = ET.SubElement(bndbox, 'xmin')
            x_min.text = str(xmin)
            y_min = ET.SubElement(bndbox, 'ymin')
            y_min.text = str(ymin)
            x_max = ET.SubElement(bndbox, 'xmax')
            x_max.text = str(xmax)
            y_max = ET.SubElement(bndbox, 'ymax')
            y_max.text = str(ymax)
            
        past_fname = file_name

        if(image_id != next_id):
            mydoc.write("{}/{}.xml".format(folder,file_name.split('.')[0]))   
            
      