class Model:
    # Expects input to have batch dimension
    def predict(self, x):
        raise NotImplementedError


class ConvolutionalNetworkModel(Model):
    def predict(self, x):
        raise NotImplementedError

    def get_last_conv_layer(self):
        raise NotImplementedError
