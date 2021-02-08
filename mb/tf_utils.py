from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Subtract

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import mean_squared_error

from tensorflow.keras.models import load_model

import tensorflow as tf
# my custom loss
import tensorflow.keras.backend as K

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.python.keras.layers import Lambda 

class MyNormalizationLayer(Layer):
    """
    reversible normalization layers 

    xx = mb.tf_utils.MyNormalizationLayer(name="outIni")(x_input)
    xx = Reshape((n,1))(xx) #????
    xx = YYY(xxx)
    .. .. ..
    .. .. ..
    out  = mb.tf_utils.MyNormalizationLayer(name="out")(xx, x_ref=x_input) # Re-Scale ,  using x_input as norm reference, need be same shape

    IF NOT BELIVE IF IT DOES WORK RUN:

    x_test = rand(1024,1024); x_test[rand(1024,1024)>.7]=0
    n = x_test.shape[1]
    xxx_in = Input( shape=(n,1) )
    xxx    = mb.tf_utils.MyNormalizationLayer(name="outIni")(xxx_in)
    xxx = MaxPooling1D(pool_size=2)(xxx)
    out    = mb.tf_utils.MyNormalizationLayer(name="out")(xxx, x_ref=xxx_in) # Re-Scale ,  using x_input as norm reference, need be same shape

    model = Model(inputs=xxx_in,outputs=out)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01))
    yyy = model(x_test)#[:,0,0]
    plot(yyy[1,:,0])
    plot(x_test[1,::2])

    """
    def __init__(self,  **kwargs):
        super(MyNormalizationLayer, self).__init__(**kwargs)
        
    def call(self, x,x_ref=None):
        if x_ref is None:
            norm = tf.norm(x,axis=1)
            return  tf.einsum('ijk,ik->ijk',x,1/(norm+1e-6))
            #return K.l2_normalize(x,axis=1)*0+x #K.dot(x, self.kernel)
        else:
            normRef = tf.norm(x_ref,axis=1)
            out = tf.einsum('ijk,ik->ijk',x,normRef+1e-6)
            return out
            #return  #x*normRef #K.l2_normalize(x,axis=1) #K.dot(x, self.kernel)    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.input_shape[1])


def negative_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)



# TF in layer manual convolution
#
# bbb_in = Input( shape=(n,1) )
# ddd1 = Conv1D(1, len(diff_kernel), name='diff',padding="same", use_bias=False, activation=None,trainable=False,kernel_initializer=tf.keras.initializers.constant(diff_kernel))(bbb_in) # complicated way calculating derivative of 
# ddd2 = Conv1D(1, len(diff_kernel), name='diff2',padding="same", use_bias=False, activation=None,trainable=False,kernel_initializer=tf.keras.initializers.constant(diff_kernel))(ddd1) # complicated way calculating derivative of predic
# modD = Model(inputs=bbb_in, outputs=[ddd1,ddd2])
# bb1,bb2 = modD(nrm(bl_train[:,::2]))


# LAMBDA
#baseline =  Lambda(lambda x: tf.math.reduce_mean(x,axis=-1))(baseline)
#baselineOUT = Reshape((-1,1),name='baseline')(baseline)
