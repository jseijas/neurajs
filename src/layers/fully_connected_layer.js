const VectorLayer = require('./vector_layer');
const Volume = require('../volume');
const Utils = require('../utils');

class FullyConnectedLayer extends VectorLayer {
  constructor(settings) {
    super(settings);
    settings = settings || {};
    this.depth = settings.num_neurons || settings.filters || 1;
    this.l1_decay_mul = settings.l1_decay_mul || 0;
    this.l2_decay_mul = settings.l2_decay_mul || 1;
    this.biases = new Volume(1, 1, this.depth, settings.bias_pref || 0);
    this.filters = [];
    for(var i = 0; i < this.depth; i += 1) { 
      this.filters.push(new Volume(1, 1, this.num_inputs)); 
    }
  }

  forward() {
    var input = this.parent.volume;
    if (!this.volume) {
      this.volume = new Volume(1, 1, this.depth, 0);
    }
    for(var i = 0; i < this.depth;i += 1) {
      var a = 0;
      var wi = this.filters[i].w;
      for(var d = 0; d < this.num_inputs; d += 1) {
        a += input.w[d] * wi[d];
      }
      a += this.biases.w[i];
      this.volume.w[i] = a;
    }
    return this.volume;
  }

  backward() {
    var input = this.parent.volume;
    input.dw = Utils.zeros(input.w.length);
    for(var i = 0; i < this.depth; i += 1) {
      var fi = this.filters[i];
      var dwi = this.volume.dw[i];
      for(var d = 0; d < this.num_inputs; d += 1) {
        input.dw[d] += fi.w[d] * dwi;
        fi.dw[d] += input.w[d] * dwi;
      }
      this.biases.dw[i] += dwi;
    }
  }
}

module.exports = FullyConnectedLayer;