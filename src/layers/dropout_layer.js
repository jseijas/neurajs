const ParentedLayer = require('./parented_layer');
const Utils = require('../utils');

class DropoutLayer extends ParentedLayer {
  constructor(settings) {
    super(settings);
    settings = settings || {};
    this.drop_prob = settings.drop_prob || 0.5;
    this.dropped = Utils.zeros(this.getSize());
  }

  forward(isTraining = false) {
    this.volume = this.parent.volume.clone();
    var numElements = this.parent.getSize();
    if (isTraining) {
      for (var i = 0; i < numElements; i += 1) {
        this.dropped[i] = Math.random() < this.drop_prob;
        if (this.dropped[i]) {
          this.volume.w[i] = 0;
        }
      }
    } else {
      for(var i = 0; i < numElements; i += 1) { 
        this.volume.w[i] *= (1 - this.drop_prob); 
      }
    }
    return this.volume;
  }

  backward() {
    var input = this.parent.volume;
    var numElements = input.w.length;
    input.dw = Utils.zeros(numElements);
    for(var i = 0; i < numElements; i += 1) {
      if(!(this.dropped[i])) { 
        input.dw[i] = this.volume.dw[i];
      }
    }
  }
}

module.exports = DropoutLayer;