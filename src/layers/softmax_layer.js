const VectorLayer = require('./vector_layer');
const Volume = require('../volume');
const Utils = require('../utils');

class SoftmaxLayer extends VectorLayer {
  forward() {
    var inputWeights = this.parent.volume.w;
    if (this.volume) {
      this.volume.clear();
    } else {
      this.volume = new Volume(1, 1, this.depth, 0);
    }
    var maxValue = inputWeights[0];
    for(var i = 1; i < this.depth; i += 1) {
      if (inputWeights[i] > maxValue) {
        maxValue = inputWeights[i];
      }
    }
    if (!this.es) {
      this.es = Utils.zeros(this.depth);
    }
    var esum = 0;
    for(var i = 0; i < this.depth; i += 1) {
      this.es[i] = Math.exp(inputWeights[i] - maxValue);
      esum += this.es[i];
    }
    for(var i = 0; i < this.depth; i += 1) {
      this.es[i] /= esum;
      this.volume.w[i] = this.es[i];
    }
    return this.volume;
  }

  backward(y) {
    this.parent.volume.clearGrads();
    for (var i = 0; i < this.depth; i += 1) {
      this.parent.volume.dw[i] = this.es[i] - (i === y ? 1.0 : 0.0);
    }
    return -Math.log(this.es[y]);
  }
}

module.exports = SoftmaxLayer;