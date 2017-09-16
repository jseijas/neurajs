const Net = require('./net');
const Trainer = require('./trainer');

class Neurapp {
  constructor() {

  }

  build(def) {
    this.net = new Net();
    this.net.build(def.layers);
    if (def.trainer) {
      this.trainer = new Trainer(this.net, def.trainer);
    } else {
      this.trainer = new Trainer(this.net, { type: 'adadelta', learning_rate: 0.01, momentum: 0.9, batch_size: 5, l2_decay: 0.0 });
    }
  }
}

module.exports = Neurapp;