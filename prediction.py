model.load_weights('Remnet_without_unalt_trian_manip.h5')

def patch_creator(img):
    
    p=0;
    img_in = imread(img)
    m, n = img_in.shape[0:2]                                                                                                                                                  
    a, b = main_size//img_rows, main_size//img_cols
    all_patch = np.zeros((a*b, img_rows, img_cols, 3))
    for k in range(a):
        for l in range (b):
            all_patch[p,:,:,:] = img_in[(k*img_rows):(k+1)*img_rows, (l*img_cols):(l+1)*img_cols, :]
            p+=1
            
    return all_patch

dir_test = '/home/tianlei/Downloads/dresden_256_patches/test/unalt/'

test_images = glob.glob(dir_test + '*/*')
images_all = []
labels_all = []
test_folders = os.listdir(dir_test)

from natsort import natsorted

test_images = natsorted(test_images)
test_folders = natsorted(test_folders)

images = ['/'.join(test_images[i].split(os.sep)[0:-1]) + '/' + '_'.join(test_images[i].split(os.sep)[-1].split('_')[0:4]) for i in range(0,len(test_images),20)]
label = [test_folders.index(test_images[i].split(os.sep)[7]) for i in range(0,len(test_images),20)]
label = [int(i) for i in label]

predicted = []
for i in tqdm(range(540)):
    image_pred = []
    for j in range(patch_no):
        img = patch_creator(images[i] + '_%d.png'%(j+1))
        image_pred.append(model.predict(img))
    image_pred = np.average(image_pred,axis = 1)
    image_pred = image_pred.argmax(axis=1)
    predicted.append(np.argmax(np.bincount(image_pred)))
    
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(label, predicted, normalize=True, sample_weight=None)
