const VectorLayer = require('./vector_layer');
const Utils = require('../utils');

class RegressionLayer extends VectorLayer {
  backward(y) { 
    var input = this.parent.volume;
    input.dw = Utils.zeros(input.w.length);
    var loss = 0;
    var l = this.depth;
    if (typeof y === 'number') {
      y = [y];
      l = 1;
    }
    for (var i = 0; i < l; i += 1) {
      var dy = input.w[i] - y[i];
      input.dw[i] = dy;
      loss += 0.5 * dy * dy;
    }
    return loss;
  }
}

module.exports = RegressionLayer;
