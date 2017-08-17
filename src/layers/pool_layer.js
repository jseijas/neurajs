const Layer = require('../layer');
const Volume = require('../volume');

class PoolLayer extends Layer {
  constructor(settings) {
    super(settings);
    settings = settings || {};
    this.sx = settings.sx;
    this.sy = settings.sy || this.sx;
    this.stride = settings.stride || 2;
    this.pad = settings.pad || 0;
    this.depth = this.parent.depth;
    this.width = Math.floor((this.parent.width + this.pad * 2 - this.sx) / this.stride + 1);
    this.height = Math.floor((this.parent.height + this.pad * 2 - this.sy) / this.stride + 1);
    this.switch = [];
    var numElements = this.getSize();
    for (var i = 0; i < numElements; i++) {
      this.switch.push({ x: -1, y: -1 });
    }
  }

  forward(V) {
    if (this.volume) {
      this.volume.clear();
    } else {
      this.volume = new Volume(this.width, this.height, this.depth, 0.0);
    }
    var n = 0;
    for(var z =0; z < this.depth; z += 1) {
      var x = -this.pad;
      var y;
      for(var ax = 0; ax < this.width; ax +=1) {
        y = -this.pad;
        for(var ay = 0; ay < this.height; ay += 1) {
          var best = {};
          for(var bx = 0; bx < this.sx; bx += 1) {
            var cx = x + bx;
            if (cx >= 0 && cx < this.parent.volume.width) {
              for(var by = 0; by < this.sy; by += 1) {
                var cy = y + by;
                if(cy >= 0 && cy< this.parent.volume.height) {
                  var value = this.parent.volume.get(cx, cy, z);
                  if(best.value === undefined || value > best.value) { 
                    best.value = value; 
                    best.x = cx; 
                    best.y = cy;
                  }
                }
              }
            }
          }
          this.switch[n].x = x;
          this.switch[n].y = y;
          n += 1;
          this.volume.set(ax, ay, z, best.value);
          y += this.stride;
        }
        x += this.stride;
      }
    }
    return this.volume;
  }

  backward() {
    this.parent.volume.clearGrads();
    var n = 0;
    for (var z = 0; z < this.depth; z += 1)  {
      for (var x = 0; x < this.width; x += 1) {
        for (var y = 0; y < this.width; y += 1) {
          this.parent.volume.add_grad(this.switch[n].x, this.switch[n].y, z, this.volume.get_grad(x, y, z));
          n += 1;
        }
      }
    }
  }
}

module.exports = PoolLayer;