import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import numpy as np


class BaseModel(tf.keras.Model):

    def __init__(self, out_dim=32):
        super().__init__()
        self.fc_state = layers.Dense(128, activation='tanh')
        self.fc_operations = layers.Dense(128, activation='tanh')
        self.flatten = layers.Flatten()
        self.cat = layers.Concatenate(axis=1)
        self.fc2 = layers.Dense(256, activation='relu')
        self.fc3 = layers.Dense(out_dim, activation='softmax')
    
    def call(self, states, operations):
        state_ebd = self.fc_state(states)
        op_ebd = self.fc_operations(self.flatten(operations))
        x = self.cat([state_ebd, op_ebd])
        x = self.fc2(x)
        return self.fc3(x)


class SimpleTestModel(tf.keras.Model):
    def __init__(self, out_dim=32):
        super().__init__()
        self.fc_state = layers.Dense(32, use_bias=False)
        a_out = self.fc_state(tf.convert_to_tensor([[0] * 32]))
        self.fc_state.set_weights([np.eye(32, dtype=np.complex)])
        # print(self.fc_state.get_weights())
        
    def call(self, states, operations):
        state_ebd = self.fc_state(states)
        return state_ebd
