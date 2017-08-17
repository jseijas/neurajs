const ActivationLayer = require('./activation_layer');

class LinearLayer extends ActivationLayer {
  activationFunction(x) {
    return x;
  }

  getBackwardGradient(inputWeight, outputWeight, outputGradient) {
    return outputGradient;
  }
}

module.exports = LinearLayer;