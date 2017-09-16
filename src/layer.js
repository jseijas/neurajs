const Serializable = require('./serializable');

class Layer extends Serializable {
  constructor(settings) {
    super();
    settings = settings || {};
    this.parent = settings.parent;
    this.width = settings.width || settings.out_sx || settings.sx || settings.width || 1;
    this.height = settings.height || settings.out_sy || settings.sy || settings.height || 1;
    this.depth = settings.depth || settings.out_depth || settings.depth || 0;
    console.log(`Creating layer of type: ${this.getLayerType()}`);
  }

  forward() {
    this.volume = this.parent.volume;
    return this.volume;
  }

  backward() {
  }

  getParamsAndGrads() {
    var response = [];
    if (this.filters) {
      for (var i = 0; i < this.depth; i += 1) {
        response.push({
          params: this.filters[i].w,
          grads: this.filters[i].dw,
          l1_decay_mul: this.l1_decay_mul,
          l2_decay_mul: this.l2_decay_mul
        });
      }
      response.push({
        params: this.biases.w,
        grads: this.biases.dw,
        l1_decay_mul: 0,
        l2_decay_mul: 0
      });
    }
    return response;
  }

  getLayerType() {
    return this.getClassName().slice(0, -5);
  }

  getSize() {
    return this.width * this.height * this.depth;
  }

  clone(opts) {
    let result;
    let settings = JSON.parse(JSON.stringify(this.settings))
    if (opts.isJSON) {
      result = {};
      result.settings = settings;
      result.width = this.width;
      result.height = this.height;
      result.depth = this.depth;
    } else {
      result = new this.constructor(settings);
    }
    if (this.volume) {
      result.volume = this.clone(opts);
    }
    return result;
  }

  assign(src, opts) {
    this.settings = JSON.parse(JSON.stringify(this.settings))
    this.parent = settings.parent;
    this.width = src.width;
    this.height = src.height;
    this.depth = src.depth;
    this.volume.assign(src.volume);
  }
}

module.exports = Layer;
