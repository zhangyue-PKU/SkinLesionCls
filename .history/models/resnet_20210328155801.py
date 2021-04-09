import keras 
from models import Backbone
from keras.applications import resnet50 as keras_resnet50
from keras import backend as K
from models.submodels.classification import default_classification_model
from misc_utils.model_utils import plot_model
from misc_utils.model_utils import load_model_from_run
from misc_utils.model_utils import save_model_to_run
from misc_utils.model_utils import load_model_weights_from
from misc_utils.print_utils import on_aws



class ResNetBackbone(Backbone):
    def __init__(self, backbone_name='resnet50'):
        super(ResNetBackbone, self).__init__(backbone_name)
        self.custom_objects['keras_resnet50'] = keras_resnet50

    def build_base_model(self, inputs, **kwarg):
        # create the vgg backbone
        if self.backbone_name == 'resnet50':
            inputs = keras.layers.Lambda(lambda x: keras_resnet50.preprocess_input(x))(inputs)
            resnet = keras_resnet50.ResNet50(input_tensor=inputs,
                                             include_top=False,
                                             weights=None)
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone_name))

        return resnet

    def classification_model(self,
                             num_dense_layers=2,
                             num_dense_units=256,
                             dropout_rate=0.2,
                             num_classes=7,
                             pooling='avg',
                             name='default_resnet_classification_model',
                             input_shape = 224,
                             save_to = "",
                             lr=1
                             **kwargs):
        """ Returns a classifier model using the correct backbone.
        """
        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape, input_shape, 3)
        else:
            input_shape = (3, input_shape, input_shape)
        
        t = keras.layers.Input(shape = input_shape)
        #input shape is processed in base_model
        base_model = self.build_base_model(input_tensor=t)
        x = base_model.ouput

        #output is softmax output of last dense layer
        output = default_classification_model(input_tensor=x,
                                              input_shape=base_model.ouput_shape[1:],
                                              num_classes=num_classes,
                                              num_dense_layers=num_dense_layers,
                                              num_dense_units=num_dense_units,
                                              dropout_rate=dropout_rate,
                                              pooling=pooling, 
                                              kernel_regularizer='L2'                            
                                              )
        model = keras.models.Model(inputs = t, outputs=output, name=name)
        if save_to:
            save_model_to_run(model, save_to)

        compile_model(model=model,
                    num_classes=num_classes,
                    metrics='acc',
                    loss='ce',
                    lr=lr)
        



        

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['resnet50', ]

        if self.backbone_name not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone_name,
                                                                                       allowed_backbones))
