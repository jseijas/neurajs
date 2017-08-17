const ActivationLayer = require('./activation_layer');

class ReluLayer extends ActivationLayer {
  activationFunction(x) {
    return x < 0 ? 0 : x;
  }

  getBackwardGradient(inputWeight, outputWeight, outputGradient) {
    return (outputWeight <= 0) ? 0 : outputGradient;
  }
}

module.exports = ReluLayer;