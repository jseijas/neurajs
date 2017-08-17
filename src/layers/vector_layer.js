const Layer = require('../layer');

class VectorLayer extends Layer {
  constructor(settings) {
    super(settings);
    settings = settings || {};
    this.width = 1;
    this.height = 1;
    this.num_inputs = this.parent.getSize();
    this.depth = this.num_inputs;
  }
}

module.exports = VectorLayer;