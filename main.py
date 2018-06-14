from flask import Flask,request
from PIL import Image, ImageOps
import numpy as np 
from keras.models import load_model
import urllib.request
app = Flask(__name__)

def transform_image(test_img):
    im = Image.open(test_img)
    desired_size=128
    
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGB", im.size, (255,255,255))
        bg.paste(im, mask=alpha)
        im = bg

    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # use thumbnail() or resize() method to resize the input image
    im = im.resize(new_size, Image.ANTIALIAS)
    
    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size),"white")
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    # contrast stretching
    new_im = ImageOps.equalize(new_im)
    
    return np.array([np.array(new_im)])



def run(image_path):
    #transform
    image = transform_image(image_path)
    print(image.shape)
    
    #Get prediction using the loaded model

    global model
    if not 'model' in globals():
        model = load_model('keras_model.h5')
    predicted = model.predict_classes(image)
    
    labels = {0:'axes',1:'boots',2:'carabiners',3:'crampons',4:'gloves',5:'hardshell_jackets',6:'harnesses',7:'helmets',8:'insulated_jackets',9:'pulleys',10:'rope',11:'tents'} 

    # Return the result
    return labels[predicted[0]]

@app.route('/gear', methods=['GET'])
def recognize_gear():
    print("HELLO")
    image_url = request.args.get('image_url')
    (image_path, headers) = urllib.request.urlretrieve(image_url)

    print(image_path)
       
    category = run(image_path)
    return category

if __name__ == '__main__':
  app.run()
