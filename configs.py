# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_mynet_config():
    """Returns configuration for mynet."""
    config = ml_collections.ConfigDict()
    #config for gpu settings
    config.gpu ='cuda'
    
    #config for training/testing data
    config.data = ml_collections.ConfigDict()
    config.data.train_path = '../UofSC_Train'
    config.data.test_path = '../UofSC_Test'
    
    #config for checkpoint path
    config.load_model = False
    config.ckpt_path = './pre_trained/UofSC_ChBest.pt'
    
    #config for learning rate
    config.learning_rate = 3e-4
    config.lambda_rot = 0.001
    
    #config for vision transformer
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 512
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    
    return config

def test():
    
    config = get_mynet_config()
    print (config.data.train_path)
    
#test()
