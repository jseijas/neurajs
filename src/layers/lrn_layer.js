const ParentedLayer = require('./parented_layer');
const Utils = require('../utils');

class LRNLayer extends ParentedLayer {
  constructor(settings) {
    super(settings);
    settings = settings || {};
    this.k = settings.k;
    this.n = settings.n;
    this.n2 = Math.floor(this.n / 2);
    this.alpha = settings.alpha;
    this.beta = settings.beta;
  }

  forward() {
    var input = this.parent.volume;
    this.volume = input.cloneAndZero();
    this.volumeCache = input.cloneAndZero();
    for (var x = 0; x < input.sx; x += 1) {
      for (var y = 0; y < input.sy; y += 1) {
        for(var i = 0; i < input.depth; i += 1) {
          var den = 0;
          var fromValue = Math.max(0, i - this.n2);
          var toValue = Math.min(i + this.n2, input.depth - 1);
          for(var j = fromValue; j <= toValue; j += 1) {
            var value = input.get(x, y, j);
            den += value * value;
          }
          den = den * (this.alpha / this.n) + this.k;
          this.volumeCache.set(x,y,i,den);
          this.volume.set(x, y, i, input.get(x,y,i) / Math.pow(den, this.beta));
        }
      }
    }
    return this.volume;
  }

  backward() { 
    var input = this.parent.volume;
    input.dw = Utils.zeros(input.w.length);
    for (var x = 0; x < input.sx; x += 1) {
      for (var y = 0; y < input.sy; y += 1) {
        for (var i = 0; i < input.depth; i += 1) {
          var gradi = this.volume.get_grad(x,y,i);
          var S = this.volumeCache.get(x,y,i);
          var SB = Math.pow(S, this.beta);
          var SB2 = SB * SB;
          var fromValue = Math.max(0, i - this.n2);
          var toValue = Math.min(i + this.n2, input.depth - 1);
          for(var j = fromValue; j <= toValue; j += 1) {              
            var aj = input.get(x, y, j); 
            var g = -aj * this.beta * Math.pow(S,this.beta-1) * this.alpha / this.n * 2 * aj;
            if (j === i) {
              g+= SB;
            }
            g /= SB2;
            g *= gradi;
            input.add_grad(x, y, j, g);
          }
        }
      }
    }
  }
}

module.exports = LRNLayer;