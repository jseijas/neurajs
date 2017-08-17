const ParentedLayer = require('./parented_layer');
const Volume = require('../volume');
const Utils = require('../utils');

class MaxoutLayer extends ParentedLayer {
  constructor(settings) {
    super(settings);
    settings = settings || {};
    this.group_size = settings.group_size || 2;
    this.depth = Math.floor(this.parent.depth / this.group_size);
    this.switches = Utils.zeros(this.getSize());
  }  

  forward() {
    var input = this.parent.volume;
    this.volume = new Volume(this.width, this.height, this.depth, 0.0);
    var n=0;
    for (var x = 0; x < input.sx; x += 1) {
      for (var y=0; y < input.sy; y += 1) {
        for (var i = 0; i < this.depth; i += 1) {
          var ix = i * this.group_size;
          var weighti = input.get(x, y, ix);
          var ai = 0;
          for (var j = 1; j < this.group_size; j += 1) {
            var weightj = input.get(x, y, ix+j);
            if (weightj > weighti) {
              weighti = weightj;
              ai = j;
            }
          }
          this.volume.set(x, y, i, weighti);
          this.switches[n] = ix + ai;
          n += 1;
        }
      }
    }
    return this.volume;
  }

  backward() {
    var input = this.parent.volume;
    input.dw = Utils.zeros(input.w.length);
    var n = 0;
    for(var x = 0;x < this.volume.sx; x += 1) {
      for(var y = 0; y < this.volume.sy; y += 1) {
        for(var i = 0; i < this.depth; i += 1) {
          input.set_grad(x,y,this.switches[n], this.volume.get_grad(x,y,i));
          n++;
        }
      }
    }
  }
}

module.exports = MaxoutLayer;