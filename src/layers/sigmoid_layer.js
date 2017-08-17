const ActivationLayer = require('./activation_layer');

class SigmoidLayer extends ActivationLayer {
  activationFunction(x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }

  getBackwardGradient(inputWeight, outputWeight, outputGradient) {
    return outputWeight * (1.0 - outputWeight) * outputGradient;
  }
}

module.exports = SigmoidLayer;