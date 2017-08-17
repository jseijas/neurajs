const Layer = require('../layer');
const Volume = require('../volume');
const Utils = require('../utils');

class ConvolutionalLayer extends Layer {
  constructor(settings) {
    super(settings);
    settings = settings || {};
    this.depth = settings.filters;
    this.horizontalFilterSize = settings.horizontalFilterSize || settings.filterSize || settings.sx;
    this.verticalFilterSize = settings.verticalFilterSize || settings.filterSize || settings.sy || this.horizontalFilterSize;
    this.stride = settings.stride || 1; 
    this.pad = settings.pad || 0;
    this.l1_decay_mul = settings.l1_decay_mul || 0;
    this.l2_decay_mul = settings.l2_decay_mul || 1;

    this.width = Math.floor((this.parent.width + this.pad * 2 - this.horizontalFilterSize) / this.stride + 1);
    this.height = Math.floor((this.parent.height + this.pad * 2 - this.verticalFilterSize) / this.stride + 1);
    var bias = settings.bias_pref || 0;
    this.filters = [];
    for(var i = 0; i < this.depth; i++) { 
      this.filters.push(new Volume(this.horizontalFilterSize, this.verticalFilterSize, this.parent.depth)); 
    }
    this.biases = new Volume(1, 1, this.depth, bias);
  }

  forward() {
    var numFilter, ay, ax, by, bx, bz;
    var input = this.parent.volume;
    if (!this.volume) {
      this.volume = new Volume(this.width, this.height, this.depth, 0);
    }
    for(numFilter = 0; numFilter < this.depth; numFilter += 1) {
      var filter = this.filters[numFilter];
      var bias = this.biases.w[numFilter];
      var x;
      var y = -this.pad;
      for(ay = 0; ay < this.height; ay += 1) {
        x = -this.pad;
        for(ax = 0; ax < this.width; ax += 1) {
          var a = bias;
          for(by = 0; by < filter.height; by += 1) {
            var oy = y + by;
            if (oy >= 0 && oy < input.height) {
              for(bx = 0; bx < filter.width; bx += 1) {
                var ox = x + bx;
                if(ox >= 0 && ox < input.width) {
                  for(bz = 0; bz < filter.depth; bz += 1) {
                    a += filter.w[((filter.width * by) + bx) * filter.depth + bz] 
                      * input.w[((input.width * oy) + ox) * input.depth + bz];
                  }
                }
              }
            }
          }
          this.volume.set(ax, ay, numFilter, a);
          x += this.stride;
        }
        y += this.stride;
      }
    }
    return this.volume;
  }

  backward() {
    var input = this.parent.volume;
    input.dw = Utils.zeros(input.w.length);
    for (var numFilter = 0; numFilter < this.depth; numFilter += 1) {
      var filter = this.filters[numFilter];
      var x;
      var y = -this.pad;
      for (var ay = 0; ay < this.height; ay += 1) {
        x = -this.pad;
        for (var ax = 0; ax < this.width; ax += 1) {
          var grad = this.volume.get_grad(ax, ay, numFilter);
          this.biases.dw[numFilter] += grad;
          for (var by = 0; by < filter.height; by += 1) {
            var oy = y + by;
            if (oy >= 0 && oy < input.height) {
              for (var bx = 0; bx < filter.width; bx += 1) {
                var ox = x + bx;
                if (ox >= 0 && ox < input.width) {
                  for(var bz = 0; bz < filter.depth; bz += 1) {
                    var ix1 = ((input.width * oy) + ox) * input.depth + bz;
                    var ix2 = ((filter.width * by) + bx) * filter.depth + bz;
                    filter.dw[ix2] += input.w[ix1] * grad;
                    input.dw[ix1] += filter.w[ix2] * grad;
                  }
                }
              }
            }
          }
          x += this.stride;
        }
        y += this.stride;
      }
    }
  }
}

module.exports = ConvolutionalLayer;