import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, SeparableConv2D, MaxPool2D, BatchNormalization, ReLU, Dense, Flatten,  SeparableConv2D
from tensorflow.keras.initializers import HeNormal

class Solar_Classifier(keras.Model):
    def __init__(self):
        super(Solar_Classifier,self).__init__()
        self.max_pool = MaxPool2D(pool_size=2,strides=2,padding='same')
        self.layer1 = Sequential([
            SeparableConv2D(64,kernel_size=3,padding='same', kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU()
        ])
        self.layer2 = Sequential([
            SeparableConv2D(64,kernel_size=3,padding='same',kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU()
        ])
        self.layer3 = Sequential([
            SeparableConv2D(128,kernel_size=3,padding='same',kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU()
        ])
        self.layer4 = Sequential([
            SeparableConv2D(128,kernel_size=3,padding='same',kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU()
        ])
        self.layer5 =Sequential([
            SeparableConv2D(256,kernel_size=3,padding='same',kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU()
        ])
        self.layer6 = Sequential([
            SeparableConv2D(256,kernel_size=3,padding='same',kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU()
        ])
        self.layer7 = Sequential([
            SeparableConv2D(512,kernel_size=3,padding='same',kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU()
        ])
        self.layer8 = Sequential([
            SeparableConv2D(512,kernel_size=3,padding='same',kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU()
        ])
        self.classifier = Sequential([
            Flatten(),
            Dense(4096,kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(),
            Dropout(0.5),
            Dense(4096,kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU(),
            Dropout(0.5),
            Dense(5)
        ])

        #for m in self.modules():
            #if not Solar_Classifier:
                #HeNormal()

    def call(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.max_pool(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.max_pool(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.max_pool(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.max_pool(out)
        out = self.layer8(out)
        out = self.layer8(out)
        out = self.max_pool(out)
       
        out = self.classifier(out)
        
        return out

    def graph(self):
        return Sequential([self.layer1,self.layer2,self.maxPool,self.layer3,self.layer4,self.maxPool,self.layer5,self.layer6,self.maxPool,self.layer7,self.layer8, self.maxPool,self.layer8,self.layer8,self.maxPool,self.classifier])
