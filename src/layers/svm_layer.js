const VectorLayer = require('./vector_layer');

class SVMLayer extends VectorLayer {
  backward(y) {
    var yscore = this.parent.volume.w[y];
    var loss = 0;
    for (var i = 0; i < this.depth; i += 1) {
      if (y !== i) {
        var ydiff = this.parent.volume[i] - yscore + 1;
        if (ydiff > 0) {
          this.parent.volume.dw[i] += 1;
          this.parent.volume.dw[y] -= 1;
          loss += ydiff;
        }
      }
    }
    return loss;
  }
}

module.exports = SVMLayer;