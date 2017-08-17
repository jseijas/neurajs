const ActivationLayer = require('./activation_layer');

class TanhLayer extends ActivationLayer {
  activationFunction(x) {
    return Math.tanh(x);
  }

  getBackwardGradient(inputWeight, outputWeight, outputGradient) {
    return (1.0 - (outputWeight * outputWeight)) * outputGradient;
  }
}

module.exports = TanhLayer;