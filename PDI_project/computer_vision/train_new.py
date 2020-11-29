import tensorflow as tf
from PIL import Image
import numpy as np
import math
from skimage import io, color

BatchSize = 1
GreyChannels = 1
Fusion_output = None
FC_Out=None

Low_Weight = {
    'wl1': tf.Variable(tf.random.truncated_normal([3, 3, 1, 64], stddev=0.001)),
    'wl2': tf.Variable(tf.random.truncated_normal([3, 3, 64, 128], stddev=0.001)),
    'wl3': tf.Variable(tf.random.truncated_normal([3, 3, 128, 128], stddev=0.001)),
    'wl4': tf.Variable(tf.random.truncated_normal([3, 3, 128, 256], stddev=0.001)),
    'wl5': tf.Variable(tf.random.truncated_normal([3, 3, 256, 256], stddev=0.001)),
    'wl6': tf.Variable(tf.random.truncated_normal([3, 3, 256, 512], stddev=0.001))
}

Low_Biases={
    'bl1':tf.Variable(tf.random.truncated_normal([64],stddev=0.001)),
    'bl2':tf.Variable(tf.random.truncated_normal([128],stddev=0.001)),
    'bl3':tf.Variable(tf.random.truncated_normal([128],stddev=0.001)),
    'bl4':tf.Variable(tf.random.truncated_normal([256],stddev=0.001)),
    'bl5':tf.Variable(tf.random.truncated_normal([256],stddev=0.001)),
    'bl6':tf.Variable(tf.random.truncated_normal([512],stddev=0.001))
}

Mid_Weight = {
    'wm1': tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], stddev=0.001)),
    'wm2': tf.Variable(tf.random.truncated_normal([3, 3, 512, 256], stddev=0.001)),

}

Mid_Biases={
    'bm1':tf.Variable(tf.random.truncated_normal([512],stddev=0.001)),
    'bm2':tf.Variable(tf.random.truncated_normal([256],stddev=0.001)),

}


Global_Weight = {
    'wg1': tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], stddev=0.001)),
    'wg2': tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], stddev=0.001)),
    'wg3': tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], stddev=0.001)),
    'wg4': tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], stddev=0.001))
}


Global_Biases={
    'bg1':tf.Variable(tf.random.truncated_normal([512],stddev=0.001)),
    'bg2':tf.Variable(tf.random.truncated_normal([512],stddev=0.001)),
    'bg3':tf.Variable(tf.random.truncated_normal([512],stddev=0.001)),
    'bg4':tf.Variable(tf.random.truncated_normal([512],stddev=0.001)),

}

FC_Weight = {
    'wf1': tf.Variable(tf.random.truncated_normal([512*7*7,1024], stddev=0.001)),
    'wf2': tf.Variable(tf.random.truncated_normal([1024, 512], stddev=0.001)),
    'wf3': tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.001)),

}


FC_Biases = {
    'bf1': tf.Variable(tf.random.truncated_normal([1024], stddev=0.001)),
    'bf2': tf.Variable(tf.random.truncated_normal([512], stddev=0.001)),
    'bf3': tf.Variable(tf.random.truncated_normal([256], stddev=0.001)),

}

ColorNet_Weight={

    'wc1': tf.Variable(tf.random.truncated_normal([3, 3, 512, 256], stddev=0.001)),
    'wc2': tf.Variable(tf.random.truncated_normal([3, 3, 256, 128], stddev=0.001)),
    'wc3': tf.Variable(tf.random.truncated_normal([3, 3, 128, 64], stddev=0.001)),
    'wc4': tf.Variable(tf.random.truncated_normal([3, 3, 64, 64], stddev=0.001)),
    'wc5': tf.Variable(tf.random.truncated_normal([3, 3, 64, 32], stddev=0.001)),
    'wc6': tf.Variable(tf.random.truncated_normal([3, 3, 32, 2], stddev=0.001))

}

ColorNet_Biases={

    'bc1': tf.Variable(tf.random.truncated_normal([256], stddev=0.001)),
    'bc2': tf.Variable(tf.random.truncated_normal([128], stddev=0.001)),
    'bc3': tf.Variable(tf.random.truncated_normal([64], stddev=0.001)),
    'bc4': tf.Variable(tf.random.truncated_normal([64], stddev=0.001)),
    'bc5': tf.Variable(tf.random.truncated_normal([32], stddev=0.001)),
    'bc6': tf.Variable(tf.random.truncated_normal([2], stddev=0.001))

}




def Construct_FC(global_cnn_output):

    #print("constructing fully connected ")
    global FC_Out
    features = tf.reshape(global_cnn_output,shape=[-1,512*7*7])
    # haneb2a negarrb 1

    features = tf.add( tf.matmul(features,FC_Weight['wf1']),FC_Biases['bf1'])
    features = tf.nn.relu(features)

    features = tf.add(tf.matmul(features,FC_Weight['wf2']), FC_Biases['bf2'])
    features = tf.nn.relu(features)

    features = tf.add( tf.matmul(features,FC_Weight['wf3']) ,FC_Biases['bf3'])
    features = tf.nn.relu(features)

    FC_Out=features
    #print("Finished constructing fully connected")
    #print("class = ",type(FC_Out))
    return features


def Construct_Fusion(mid_output,global_output):
    #print("constructing fusion started")
    global BatchSize
    global_output = tf.tile(global_output, [1, 28*28])
    global_output= tf.reshape(global_output, [BatchSize, 28, 28, 256])
    Fusion_output = tf.concat([mid_output, global_output], 3)
    #print("constructing fusion finished")
    return Fusion_output

def Normlization(Value,MinVale,MaxValue,MinNormalizeValue,MaxNormalizeVale):
    '''
    normalize the Input
    :param value: pixl value
    :param MinVale:Old min Vale
    :param MaxValue: Old Max vale
    :return: Normailed Input between 0 1
    '''
    Value = MinNormalizeValue + (((MaxNormalizeVale-MinNormalizeValue)*(Value- MinVale))/(MaxValue-MinVale))
    return Value

def DeNormlization(Value,MinVale,MaxValue,MinNormalizeValue,MaxNormalizeVale ):
    '''
    :param Value:
    :param MinVale:
    :param MaxValue:
    :param MinNormalizeValue:
    :param MaxNormalizeVale:
    :return:
    '''
    Value = MinNormalizeValue + (((MaxNormalizeVale-MinNormalizeValue)*(Value- MinVale))/(MaxValue-MinVale))
    return Value

def F_Norm(Tens):
    return tf.reduce_sum(input_tensor=Tens**2)**0.5


def Construct_Graph(input):

    lowconv1 = tf.nn.relu(tf.nn.conv2d(input=input, filters=Low_Weight['wl1'], strides=[1, 2, 2, 1], padding='SAME') + Low_Biases['bl1'])
    lowconv2 = tf.nn.relu(tf.nn.conv2d(input=lowconv1, filters=Low_Weight['wl2'], strides=[1, 1, 1, 1], padding='SAME') + Low_Biases['bl2'])
    lowconv3 = tf.nn.relu(tf.nn.conv2d(input=lowconv2, filters=Low_Weight['wl3'], strides=[1, 2, 2, 1], padding='SAME') + Low_Biases['bl3'])
    lowconv4 = tf.nn.relu(tf.nn.conv2d(input=lowconv3, filters=Low_Weight['wl4'], strides=[1, 1, 1, 1], padding='SAME') + Low_Biases['bl4'])
    lowconv5 = tf.nn.relu(tf.nn.conv2d(input=lowconv4, filters=Low_Weight['wl5'], strides=[1, 2, 2, 1], padding='SAME') + Low_Biases['bl5'])
    lowconv6 = tf.nn.relu(tf.nn.conv2d(input=lowconv5, filters=Low_Weight['wl6'], strides=[1, 1, 1, 1], padding='SAME') + Low_Biases['bl6'])
    #Mid
    midconv1=tf.nn.relu(tf.nn.conv2d(input=lowconv6,filters=Mid_Weight['wm1'],strides=[1,1,1,1],padding='SAME')+Mid_Biases['bm1'])
    midconv2=tf.nn.relu(tf.nn.conv2d(input=midconv1,filters=Mid_Weight['wm2'],strides=[1,1,1,1],padding='SAME')+Mid_Biases['bm2'])


    #Global
    globalconv1=tf.nn.relu(tf.nn.conv2d(input=lowconv6, filters=Global_Weight['wg1'], strides=[1, 2, 2, 1], padding='SAME')+Global_Biases['bg1'])
    globalconv2=tf.nn.relu(tf.nn.conv2d(input=globalconv1, filters=Global_Weight['wg2'], strides=[1, 1, 1, 1], padding='SAME')+Global_Biases['bg2'])
    globalconv3=tf.nn.relu(tf.nn.conv2d(input=globalconv2, filters=Global_Weight['wg3'], strides=[1, 2, 2, 1], padding='SAME')+Global_Biases['bg3'])
    globalconv4=tf.nn.relu(tf.nn.conv2d(input=globalconv3, filters=Global_Weight['wg4'], strides=[1, 1, 1, 1], padding='SAME')+Global_Biases['bg4'])

    MM=Construct_FC(globalconv4)

    Fuse = Construct_Fusion(midconv2, FC_Out)

    colconv1 = tf.nn.relu(tf.nn.conv2d(input=Fuse,filters=ColorNet_Weight['wc1'],strides=[1,1,1,1],padding='SAME')+ColorNet_Biases['bc1'])
    colconv2 = tf.nn.relu(tf.nn.conv2d(input=colconv1,filters=ColorNet_Weight['wc2'],strides=[1,1,1,1],padding='SAME')+ColorNet_Biases['bc2'])
    colconv2_UpSample = tf.image.resize(colconv2, [56, 56], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    colconv3 = tf.nn.relu(tf.nn.conv2d(input=colconv2_UpSample,filters=ColorNet_Weight['wc3'],strides=[1,1,1,1],padding='SAME')+ColorNet_Biases['bc3'])
    colconv4 = tf.nn.relu(tf.nn.conv2d(input=colconv3,filters=ColorNet_Weight['wc4'],strides=[1,1,1,1],padding='SAME')+ColorNet_Biases['bc4'])
    colconv4_UpSample = tf.image.resize(colconv4,[112,112], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    colconv5 = tf.nn.relu(tf.nn.conv2d(input=colconv4_UpSample,filters=ColorNet_Weight['wc5'],strides=[1,1,1,1],padding='SAME')+ColorNet_Biases['bc5'])
    colconv6 = tf.nn.relu(tf.nn.conv2d(input=colconv5,filters=ColorNet_Weight['wc6'],strides=[1,1,1,1],padding='SAME')+ColorNet_Biases['bc6'])

    output=tf.image.resize(colconv6,[224,224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return  output
