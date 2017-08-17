const Layer = require('../layer');

class ParentedLayer extends Layer {
  constructor(settings) {
    super(settings);
    this.width = this.parent.width;
    this.height = this.parent.height;
    this.depth = this.parent.depth;
  }
}

module.exports = ParentedLayer;