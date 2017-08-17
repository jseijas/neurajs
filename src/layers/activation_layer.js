const ParentedLayer = require('./parented_layer');

class ActivationLayer extends ParentedLayer {
  forward(volume) {
    if (!this.volume) {
      this.volume = this.parent.volume.cloneAndZero();
    }
    const numElements = this.getSize();
    for (var i = 0; i < numElements; i += 1) {
      this.volume.w[i] = this.activationFunction(this.parent.volume.w[i]);
    }
    return this.volume;
  }

  backward() {
    const numElements = this.parent.volume.w.length;
    for (var i = 0; i < numElements; i += 1) {
      this.parent.volume.dw[i] = this.getBackwardGradient(this.parent.volume.w[i], this.volume.w[i], this.volume.dw[i]);
    }
  }
}

module.exports = ActivationLayer;
