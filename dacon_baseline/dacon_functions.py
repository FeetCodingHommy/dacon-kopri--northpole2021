import tensorflow as tf
import numpy as np


def train_map_func(x_list, y_list):
    train_x, train_y = [], []
    for path in x_list:
        train_x.append(np.load(path)[:,:,0:1])
    for path in y_list:
        train_y.append(np.load(path)[:,:,0:1])
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    train_x = train_x.astype(np.float32)/250
    train_y = train_y.astype(np.float32)/250
    return train_x, train_y


@tf.function
def loss_function(output, target):
    mae_loss = tf.math.reduce_mean(tf.keras.losses.MAE(output, target))
    return mae_loss


def predict(model, img_path):
    test_imgs=[]
    for path in img_path:
        test_imgs.append(np.load(path)[:,:,0:1].astype(np.float32)/250)
    test_imgs = np.array([test_imgs])
    enc_input = tf.convert_to_tensor(test_imgs)
    pred = model(enc_input)[0].numpy()*250
    
    return pred
