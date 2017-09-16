const Serializable = require('./serializable');
const Utils = require('./utils');

class Volume extends Serializable {
  constructor(width, height, depth, c) {
    super();
    this.width = width;
    this.height = height;
    this.depth = depth;
    this.size = this.width * this.height * this.depth;
    this.w = Utils.zeros(this.size);
    this.dw = Utils.zeros(this.size);
    if(c === undefined) {
      var scale = Math.sqrt(1.0/(this.size));
      for (var i = 0; i < this.size; i += 1) { 
        this.w[i] = Utils.randn(0.0, scale);
      }
    } else {
      this.setConst(c);
    }
  }

  getIndex(x, y, z) {
    return (((this.width * y) + x) * this.depth) + z;
  }

  get(x, y, z) {
    return this.w[this.getIndex(x, y, z)];
  }

  set(x, y, z, v) {
    this.w[this.getIndex(x, y, z)] = v;
  }

  add(x, y, z, v) {
    this.w[this.getIndex(x, y, z)] += v;
  }

  get_grad(x, y, z) {
    return this.dw[this.getIndex(x, y, z)];
  }

  set_grad(x, y, z, v) {
    this.dw[this.getIndex(x, y, z)] = v;
  }

  add_grad(x, y, z, v) { 
    this.dw[this.getIndex(x, y, z)] += v;
  }

  clearWeights() {
    for (var i = 0; i < this.w.length; i+= 1) {
      this.w[i] = 0;
    }
  }

  clearGrads() {
    for (var i = 0; i < this.dw.length; i += 1) {
      this.dw[i] = 0;
    }
  }

  clear() {
    this.clearWeights();
    this.clearGrads();
  }

  cloneAndZero() { 
    return new Volume(this.width, this.height, this.depth, 0.0);
  }

  clone(opts) {
    let V;
    if (opts.isJSON) {
      V = { width: this.width, height: this.height, depth: this.depth };
      V.w = Utils.zeros(this.size);
      V.dw = Utils.zeros(this.size);
    } else {
      V = this.cloneAndZero();
    }
    for (var i = 0; i < this.size; i += 1) {
      V.w[i] = this.w[i];
    }
    if (opts.isSnapshot) {
      for (var i = 0; i < this.size; i += 1) {
        V.dw[i] = this.dw[i];
      }
    } else {
      delete V.dw;
    }
    return V;
  }
    
  assign(src, opts) {
    this.width = src.width;
    this.height = src.height;
    this.depth = src.depth;
    this.size = this.width * this.height * this.depth;
    this.w = Utils.zeros(this.size);
    this.dw = Utils.zeros(this.size);
    for (var i = 0; i < this.size; i += 1) {
      this.w[i] = src.w[i];
    }
    if (src.dw) {
      for (var i = 0; i < this.size; i += 1) {
        this.dw[i] = src.dw[i];
      }
    }
  }

  addFrom(V) { 
    for(var k = 0; k < this.size; k += 1) { 
      this.w[k] += V.w[k]; 
    }
  }

  addFromScaled(V, a) { 
    for(var k = 0; k < this.size; k +=1) { 
      this.w[k] += a*V.w[k]; 
    }
  }

  setConst(a) { 
    for(var k = 0; k < this.size; k += 1) { 
      this.w[k] = a; 
    }
  }

  flip() {
    let src = this.clone();
    for (let x = 0; x < this.width; x += 1) {
      for (let y = 0; y < this.height; y += 1) {
        for (let z = 0; z < this.depth; z += 1) {
          this.set(x, y, z, src.get(this.width - x - 1, y, z));
        }
      }
    }
  }

  crop(dx, dy, crop) {
    let result;
    if (crop !== this.width || dx !== 0 || dy !== 0) {
      result = new Volume(crop, crop, this.depth, 0);
      for (let x = 0; x < crop; x += 1) {
        for (let y = 0; y < crop; y += 1) {
          if (x + dx >= 0 && x + dx < this.width && y + dy >= 0 && y + dy < this.height) {
            for (let z = 0; z < this.depth; z += 1) {
              result.set(x, y, z, this.get(x + dx, y + dy, z));
            }
          }
        }
      }
    } else {
      result = this.clone();
    }  
    return result;
  }

  augment(crop, dx, dy, flip = false) {
    dx = dx || Utils.randi(0, this.width - crop);
    dy = dy || Utils.randi(0, this.height - crop);
    const result = this.crop(dx, dy, crop);
    if (flip) {
      result.flip();
    }
    return result;
  }
}

module.exports = Volume;
