import numpy as np
import matplotlib.pyplot as plt
import config

def valid_keypoints_plot(image, outputs, orig_keypoints, epoch):
    image = image.detach().cpu()
    outputs = outputs.detach().cpu().numpy()
    orig_keypoints = orig_keypoints.detach().cpu().numpy()

    img = image[0]
    output_keypoint = outputs[0]
    orig_keypoint = orig_keypoints[0]
    
    img = np.array(img, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)

    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoint = orig_keypoint.reshape(-1, 2)
    for p in range(output_keypoint.shape[0]):
        plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.')
        plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'b.')
    
    cp = r'D:\AI dev\ML\project\facial detection RGB\outputs'
    plt.savefig(f"{cp}\\val_epoch_{epoch}.png")
    plt.close()
    
def dataset_keypoints_plot(data):
    plt.figure(figsize=(10,10))
    for i in range(9):
        sample = data[i]
        img = sample['image']
        img = np.array(img, dtype="float32")
        img = np.transpose(img, (1, 2, 0))
        plt.subplot(3,3,i+1)
        plt.imshow(img)
        keypoints = sample['keypoints']
        for j in range(len(keypoints)):
            plt.plot(keypoints[j][0], keypoints[j][1], 'r.')
    plt.show()
    plt.close()
        