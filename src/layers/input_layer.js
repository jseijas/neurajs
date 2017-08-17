const Layer = require('../layer');

class InputLayer extends Layer {
  constructor(settings) {
    super(settings);
  }

  inputVolume(volume) {
    this.parent = { volume: volume };
  }
}

module.exports = InputLayer;