import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import skimage.io
import cv2
import skimage

def crop_image(x, target_height=227, target_width=227):
    image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))
attention = np.load('attention.npy')
f = open(r'test.txt')
caption = []
line = f.readline()
line = line.split('\t')
line[len(line)-1] = '.'


img = crop_image('4801369809.jpg')

alphas = np.array(attention).swapaxes(1,2)
n_words = alphas.shape[0] + 1
w = np.round(np.sqrt(n_words))
h = np.ceil(np.float32(n_words) / w)

plt.subplot(w, h, 1)
plt.imshow(img)
plt.axis('off')

smooth = True

for ii in range(alphas.shape[0]):
    plt.subplot(w, h, ii + 2)
    lab = line[ii]

    plt.text(0, 1, lab, backgroundcolor='white', fontsize=13)
    plt.text(0, 1, lab, color='black', fontsize=13)
    plt.imshow(img)

    if smooth:
        alpha_img = skimage.transform.pyramid_expand(alphas[ii, 0, :].reshape(16, 16), upscale=16, sigma=20)
    else:
        alpha_img = skimage.transform.resize(alphas[ii, 0, :].reshape(16, 16), [img.shape[0], img.shape[1]])

    plt.imshow(alpha_img, alpha=0.8)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
plt.show()