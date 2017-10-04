(function(f){if(typeof exports==="object"&&typeof module!=="undefined"){module.exports=f()}else if(typeof define==="function"&&define.amd){define([],f)}else{var g;if(typeof window!=="undefined"){g=window}else if(typeof global!=="undefined"){g=global}else if(typeof self!=="undefined"){g=self}else{g=this}g.neura = f()}})(function(){var define,module,exports;return (function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
module.exports = {
  Utils : require('./utils'),
  Serializable : require('./serializable'),
  Layer : require('./layer'),
  Volume : require('./volume'),
  ActivationLayer : require('./layers/activation_layer'),
  ConvolutionalLayer : require('./layers/convolutional_layer'),
  DropoutLayer : require('./layers/dropout_layer'),
  FullyConnectedLayer : require('./layers/fully_connected_layer'),
  InputLayer : require('./layers/input_layer'),
  LinearLayer: require('./layers/linear_layer'),
  LRNLayer : require('./layers/lrn_layer'),
  MaxoutLayer : require('./layers/maxout_layer'),
  ParentedLayer : require('./layers/parented_layer'),
  PoolLayer : require('./layers/pool_layer'),
  RegresionLayer : require('./layers/regression_layer'),
  ReluLayer : require('./layers/relu_layer'),
  SigmoidLayer : require('./layers/sigmoid_layer'),
  SoftmaxLayer : require('./layers/softmax_layer'),
  SVMLayer : require('./layers/svm_layer'),
  TanhLayer : require('./layers/tanh_layer'),
  VectorLayer : require('./layers/vector_layer'),
  Net : require('./net'),
  Trainer: require('./trainer'),
  Neurapp: require('./neurapp')
}

},{"./layer":2,"./layers/activation_layer":3,"./layers/convolutional_layer":4,"./layers/dropout_layer":5,"./layers/fully_connected_layer":6,"./layers/input_layer":7,"./layers/linear_layer":8,"./layers/lrn_layer":9,"./layers/maxout_layer":10,"./layers/parented_layer":11,"./layers/pool_layer":12,"./layers/regression_layer":13,"./layers/relu_layer":14,"./layers/sigmoid_layer":15,"./layers/softmax_layer":16,"./layers/svm_layer":17,"./layers/tanh_layer":18,"./layers/vector_layer":19,"./net":20,"./neurapp":21,"./serializable":22,"./trainer":23,"./utils":24,"./volume":25}],2:[function(require,module,exports){
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

},{"./serializable":22}],3:[function(require,module,exports){
const ParentedLayer = require('./parented_layer');

class ActivationLayer extends ParentedLayer {
  forward(volume) {
    if (!this.volume) {
      this.volume = this.parent.volume.cloneAndZero();
    }
    const numElements = this.getSize();
    for (var i = 0; i < numElements; i += 1) {
      this.volume.w[i] = this.activationFunction(this.parent.volume.w[i]);
    }
    return this.volume;
  }

  backward() {
    const numElements = this.parent.volume.w.length;
    for (var i = 0; i < numElements; i += 1) {
      this.parent.volume.dw[i] = this.getBackwardGradient(this.parent.volume.w[i], this.volume.w[i], this.volume.dw[i]);
    }
  }
}

module.exports = ActivationLayer;

},{"./parented_layer":11}],4:[function(require,module,exports){
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
},{"../layer":2,"../utils":24,"../volume":25}],5:[function(require,module,exports){
const ParentedLayer = require('./parented_layer');
const Utils = require('../utils');

class DropoutLayer extends ParentedLayer {
  constructor(settings) {
    super(settings);
    settings = settings || {};
    this.drop_prob = settings.drop_prob || 0.5;
    this.dropped = Utils.zeros(this.getSize());
  }

  forward(isTraining = false) {
    this.volume = this.parent.volume.clone();
    var numElements = this.parent.getSize();
    if (isTraining) {
      for (var i = 0; i < numElements; i += 1) {
        this.dropped[i] = Math.random() < this.drop_prob;
        if (this.dropped[i]) {
          this.volume.w[i] = 0;
        }
      }
    } else {
      for(var i = 0; i < numElements; i += 1) { 
        this.volume.w[i] *= (1 - this.drop_prob); 
      }
    }
    return this.volume;
  }

  backward() {
    var input = this.parent.volume;
    var numElements = input.w.length;
    input.dw = Utils.zeros(numElements);
    for(var i = 0; i < numElements; i += 1) {
      if(!(this.dropped[i])) { 
        input.dw[i] = this.volume.dw[i];
      }
    }
  }
}

module.exports = DropoutLayer;
},{"../utils":24,"./parented_layer":11}],6:[function(require,module,exports){
const VectorLayer = require('./vector_layer');
const Volume = require('../volume');
const Utils = require('../utils');

class FullyConnectedLayer extends VectorLayer {
  constructor(settings) {
    super(settings);
    settings = settings || {};
    this.depth = settings.neurons || settings.num_neurons || settings.filters || 1;
    this.l1_decay_mul = settings.l1_decay_mul || 0;
    this.l2_decay_mul = settings.l2_decay_mul || 1;
    this.biases = new Volume(1, 1, this.depth, settings.bias_pref || 0);
    this.filters = [];
    for(var i = 0; i < this.depth; i += 1) { 
      this.filters.push(new Volume(1, 1, this.num_inputs)); 
    }
  }

  forward() {
    var input = this.parent.volume;
    if (!this.volume) {
      this.volume = new Volume(1, 1, this.depth, 0);
    }
    for(var i = 0; i < this.depth;i += 1) {
      var a = 0;
      var wi = this.filters[i].w;
      for(var d = 0; d < this.num_inputs; d += 1) {
        a += input.w[d] * wi[d];
      }
      a += this.biases.w[i];
      this.volume.w[i] = a;
    }
    return this.volume;
  }

  backward() {
    var input = this.parent.volume;
    input.dw = Utils.zeros(input.w.length);
    for(var i = 0; i < this.depth; i += 1) {
      var fi = this.filters[i];
      var dwi = this.volume.dw[i];
      for(var d = 0; d < this.num_inputs; d += 1) {
        input.dw[d] += fi.w[d] * dwi;
        fi.dw[d] += input.w[d] * dwi;
      }
      this.biases.dw[i] += dwi;
    }
  }
}

module.exports = FullyConnectedLayer;
},{"../utils":24,"../volume":25,"./vector_layer":19}],7:[function(require,module,exports){
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
},{"../layer":2}],8:[function(require,module,exports){
const ActivationLayer = require('./activation_layer');

class LinearLayer extends ActivationLayer {
  activationFunction(x) {
    return x;
  }

  getBackwardGradient(inputWeight, outputWeight, outputGradient) {
    return outputGradient;
  }
}

module.exports = LinearLayer;
},{"./activation_layer":3}],9:[function(require,module,exports){
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
},{"../utils":24,"./parented_layer":11}],10:[function(require,module,exports){
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
},{"../utils":24,"../volume":25,"./parented_layer":11}],11:[function(require,module,exports){
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
},{"../layer":2}],12:[function(require,module,exports){
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
},{"../layer":2,"../volume":25}],13:[function(require,module,exports){
const VectorLayer = require('./vector_layer');
const Utils = require('../utils');

class RegressionLayer extends VectorLayer {
  backward(y) { 
    var input = this.parent.volume;
    input.dw = Utils.zeros(input.w.length);
    var loss = 0;
    var l = this.depth;
    if (typeof y === 'number') {
      y = [y];
      l = 1;
    }
    for (var i = 0; i < l; i += 1) {
      var dy = input.w[i] - y[i];
      input.dw[i] = dy;
      loss += 0.5 * dy * dy;
    }
    return loss;
  }
}

module.exports = RegressionLayer;

},{"../utils":24,"./vector_layer":19}],14:[function(require,module,exports){
const ActivationLayer = require('./activation_layer');

class ReluLayer extends ActivationLayer {
  activationFunction(x) {
    return x < 0 ? 0 : x;
  }

  getBackwardGradient(inputWeight, outputWeight, outputGradient) {
    return (outputWeight <= 0) ? 0 : outputGradient;
  }
}

module.exports = ReluLayer;
},{"./activation_layer":3}],15:[function(require,module,exports){
const ActivationLayer = require('./activation_layer');

class SigmoidLayer extends ActivationLayer {
  activationFunction(x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }

  getBackwardGradient(inputWeight, outputWeight, outputGradient) {
    return outputWeight * (1.0 - outputWeight) * outputGradient;
  }
}

module.exports = SigmoidLayer;
},{"./activation_layer":3}],16:[function(require,module,exports){
const VectorLayer = require('./vector_layer');
const Volume = require('../volume');
const Utils = require('../utils');

class SoftmaxLayer extends VectorLayer {
  forward() {
    var inputWeights = this.parent.volume.w;
    if (this.volume) {
      this.volume.clear();
    } else {
      this.volume = new Volume(1, 1, this.depth, 0);
    }
    var maxValue = inputWeights[0];
    for(var i = 1; i < this.depth; i += 1) {
      if (inputWeights[i] > maxValue) {
        maxValue = inputWeights[i];
      }
    }
    if (!this.es) {
      this.es = Utils.zeros(this.depth);
    }
    var esum = 0;
    for(var i = 0; i < this.depth; i += 1) {
      this.es[i] = Math.exp(inputWeights[i] - maxValue);
      esum += this.es[i];
    }
    for(var i = 0; i < this.depth; i += 1) {
      this.es[i] /= esum;
      this.volume.w[i] = this.es[i];
    }
    return this.volume;
  }

  backward(y) {
    this.parent.volume.clearGrads();
    for (var i = 0; i < this.depth; i += 1) {
      this.parent.volume.dw[i] = this.es[i] - (i === y ? 1.0 : 0.0);
    }
    return -Math.log(this.es[y]);
  }
}

module.exports = SoftmaxLayer;
},{"../utils":24,"../volume":25,"./vector_layer":19}],17:[function(require,module,exports){
const VectorLayer = require('./vector_layer');

class SVMLayer extends VectorLayer {
  backward(y) {
    var yscore = this.parent.volume.w[y];
    var loss = 0;
    for (var i = 0; i < this.depth; i += 1) {
      if (y !== i) {
        var ydiff = this.parent.volume[i] - yscore + 1;
        if (ydiff > 0) {
          this.parent.volume.dw[i] += 1;
          this.parent.volume.dw[y] -= 1;
          loss += ydiff;
        }
      }
    }
    return loss;
  }
}

module.exports = SVMLayer;
},{"./vector_layer":19}],18:[function(require,module,exports){
const ActivationLayer = require('./activation_layer');

class TanhLayer extends ActivationLayer {
  activationFunction(x) {
    return Math.tanh(x);
  }

  getBackwardGradient(inputWeight, outputWeight, outputGradient) {
    return (1.0 - (outputWeight * outputWeight)) * outputGradient;
  }
}

module.exports = TanhLayer;
},{"./activation_layer":3}],19:[function(require,module,exports){
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
},{"../layer":2}],20:[function(require,module,exports){
const ConvolutionalLayer = require('./layers/convolutional_layer');
const DropoutLayer = require('./layers/dropout_layer');
const FullyConnectedLayer = require('./layers/fully_connected_layer');
const InputLayer = require('./layers/input_layer');
const LinearLayer = require('./layers/linear_layer');
const LRNLayer = require('./layers/lrn_layer');
const MaxoutLayer = require('./layers/maxout_layer');
const PoolLayer = require('./layers/pool_layer');
const RegressionLayer = require('./layers/regression_layer');
const ReluLayer = require('./layers/relu_layer');
const SigmoidLayer = require('./layers/sigmoid_layer');
const SoftmaxLayer = require('./layers/softmax_layer');
const SVMLayer = require('./layers/svm_layer');
const TanhLayer = require('./layers/tanh_layer');
const Serializable = require('./serializable');

class Net extends Serializable {
  constructor() {
    super();
    this.layers = [];
    this.layerClasses = {};
    this.registerLayers();
  }

  registerLayer(layerName, layerClass) {
    this.layerClasses[layerName] = layerClass;
  }

  registerLayers() {
    this.registerLayer('conv', ConvolutionalLayer);
    this.registerLayer('dropout', DropoutLayer);
    this.registerLayer('fc', FullyConnectedLayer);
    this.registerLayer('input', InputLayer);
    this.registerLayer('linear', LinearLayer);
    this.registerLayer('lrn', LRNLayer);
    this.registerLayer('maxout', MaxoutLayer);
    this.registerLayer('pool', PoolLayer);
    this.registerLayer('regression', RegressionLayer);
    this.registerLayer('relu', ReluLayer);
    this.registerLayer('sigmoid', SigmoidLayer);
    this.registerLayer('softmax', SoftmaxLayer);
    this.registerLayer('svm', SVMLayer);
    this.registerLayer('tanh', TanhLayer);
  }

  createLayer(def) {
    return new this.layerClasses[def.type](def);
  }

  addPrelayer(def) {
    if (def.type === 'softmax' || def.type === 'svm' || def.type === 'regression') {
      this.addLayer({ type: 'fc', num_neurons: def.neurons || def.num_classes || def.num_neurons });
    }    
  }

  addPostlayer(def) {
    if (def.activation) {
      let sonDef = { type: def.activation };
      if (def.activation === 'maxout') {
        sonDef.group_size = def.group_size || 2;
      }
      this.addLayer(sonDef);
    }
    if (def.drop_prob && def.type !== 'dropout') {
      this.addLayer({ type: 'dropout', drop_prob: def.drop_prob });
    }
  }

  addLayer(def) {
    this.addPrelayer(def);
    if ((def.type === 'fc' || def.type === 'conv') && def.bias_pref === undefined) {
      def.bias_pref = def.activation === 'relu' ? 0.1 : 0;
    }
    def.parent = this.layers.length > 0 ? this.layers[this.layers.length -1] : undefined;
    this.layers.push(this.createLayer(def));
    this.addPostlayer(def);
  }

  makeLayers(defs) {
    this.layers = [];
    for (var i = 0; i < defs.length; i += 1) {
      this.addLayer(defs[i]);
    }  
  }

  build(def) {
    let defs = [];
    if (def.input) {
      def.input.type = 'input';
      defs.push(def.input);
    }
    for (let i = 0; i < def.layers.length; i += 1) {
      let layer = def.layers[i];
      if (!layer.iterate) {
        layer.iterate = 1;
      }
      if (!layer.type) {
        layer.type = 'fc';
      }
      for (let j = 0; j < layer.iterate; j += 1) {
        defs.push(layer);
      }
    }
    if (def.output) {
      defs.push(def.output);
    }
    this.makeLayers(defs);
  }

  getInputLayer() {
    return this.layers[0];
  }

  forward(volume, isTraining = false) {
    this.layers[0].inputVolume(volume);
    for (var i = 0; i < this.layers.length; i += 1) {
      this.layers[i].forward(isTraining);
    }
    return this.layers[this.layers.length - 1].volume;
  }

  backward(y) {
    const loss = this.layers[this.layers.length -1].backward(y);
    for (var i = this.layers.length -2; i >= 0; i -= 1) {
      this.layers[i].backward();
    }
    return loss;
  }

  getPrediction() {
    let probs = this.layers[this.layers.length - 1].volume.w;
    let maxv = probs[0];
    let maxi = 0;
    for (var i = 1; i < probs.length; i += 1) {
      if (probs[i] > maxv) {
        maxv = probs[i];
        maxi = i;
      }
    }
    return maxi;
  }

  getParamsAndGrads() {
    var response = [];
    for (var i = 0; i < this.layers.length; i += 1) {
      var layer_response = this.layers[i].getParamsAndGrads();
      for (var j = 0; j < layer_response.length; j += 1) {
        response.push(layer_response[j]);
      }
    }
    return response;
  }

  clone(opts) {
    const result = opts.isJSON ? {} : new Net();
    for (let i = 0; i < this.layers.length; i += 1) {
      result.layers.add(this.layers[i].clone(opts));
    }
    return result;
  }

  assign(src, opts) {
    this.layers = [];
    for (let i = 0; i < src.layers.length; i += 1) {
      let srclayer = src.layers[i];
      if (i > 0) {
        srclayer.settings.parent = this.layers[i - 1];
      }
      let layer = createLayer(srclayer.settings);
      layer.volume.assign(srclayer.volume);
      this.layers.add(layer);
    }
  }
}

module.exports = Net;
},{"./layers/convolutional_layer":4,"./layers/dropout_layer":5,"./layers/fully_connected_layer":6,"./layers/input_layer":7,"./layers/linear_layer":8,"./layers/lrn_layer":9,"./layers/maxout_layer":10,"./layers/pool_layer":12,"./layers/regression_layer":13,"./layers/relu_layer":14,"./layers/sigmoid_layer":15,"./layers/softmax_layer":16,"./layers/svm_layer":17,"./layers/tanh_layer":18,"./serializable":22}],21:[function(require,module,exports){
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
},{"./net":20,"./trainer":23}],22:[function(require,module,exports){
class Serializable {
  clone(isJSON, opts) {
  }

  assign(opts) {
  }

  toJSON(opts) {
    opts.isJSON = true;
    const result = this.clone(opts);
    result.className = this.getClassName();
    return JSON.stringify(result);
  }

  fromJSON(opts) {
    this.assign(JSON.parse(json, opts));    
  }

  getClassName() {
    return this.constructor.name;
  }
}

module.exports = Serializable;
},{}],23:[function(require,module,exports){
const Utils = require('./utils');

class Trainer {
  constructor(net, settings) {
    this.net = net;
    settings = settings || {};
    this.learning_rate = settings.learning_rate || 0.01;
    this.l1_decay = settings.l1_decay || 0;
    this.l2_decay = settings.l2_decay || 0;
    this.batch_size = settings.batch_size || 1;
    this.methodName = settings.method || 'sgd';
    this.momentum = settings.momentum || 0.9;
    this.ro = settings.ro || 0.95;
    this.eps = settings.eps || 1e-6;
    this.iteration = 0;
    this.gsum = [];
    this.xsum = [];
    this.initializeMethods();
  }

  initializeMethods() {
    this.methods = {};
    this.methods['adam'] = this.adamMethod.bind(this);
    this.methods['adagrad'] = this.adagradMethod.bind(this);
    this.methods['windowgrad'] = this.windowgradMethod.bind(this);
    this.methods['adadelta'] = this.adadeltaMethod.bind(this);
    this.methods['nesterov'] = this.nesterovMethod.bind(this);
    this.methods['sgd'] = this.sgdMethod.bind(this);
  }

  getMethod() {
    return this.methods[this.methodName] || this.methods['sgd'];
  }

  adamMethod(i, j, gij) {
    var gsumi = this.gsum[i];
    var xsumi = this.xsum[i];
    gsumi[j] = gsumi[j] * this.beta1 + (1- this.beta1) * gij; 
    xsumi[j] = xsumi[j] * this.beta2 + (1-this.beta2) * gij * gij;
    var biasCorr1 = gsumi[j] / (1 - Math.pow(this.beta1, this.iteration));
    var biasCorr2 = xsumi[j] / (1 - Math.pow(this.beta2, this.iteration));
    return - this.learning_rate * biasCorr1 / (Math.sqrt(biasCorr2) + this.eps);
  }

  adagradMethod(i, j, gij) {
    var gsumi = this.gsum[i];
    gsumi[j] = gsumi[j] + gij * gij;
    return this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij;
  }

  windowgradMethod(i, j, gij) {
    var gsumi = this.gsum[i];
    gsumi[j] = this.ro * gsumi[j] + (1 - this.ro) * gij * gij;
    return this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij;
  }

  adadeltaMethod(i, j, gij) {
    var gsumi = this.gsum[i];
    var xsumi = this.xsum[i];
    gsumi[j] = this.ro * gsumi[j] + (1 - this.ro) * gij * gij;
    var dx = - Math.sqrt((xsumi[j] + this.eps) / (gsumi[j] + this.eps)) * gij;
    xsumi[j] = this.ro * xsumi[j] + (1 - this.ro) * dx * dx;
    return dx;
  }

  nesterovMethod(i, j, gij) {
    var gsumi = this.gsum[i];
    gsumi[j] = gsumi[j] * this.momentum + this.learning_rate * gij;
    return this.momentum * gsumi[j] - (1.0 + this.momentum) * gsumi[j];
  }

  sgdMethod(i, j, gij) {
    if (this.momentum > 0) {
      var gsumi = this.gsum[i];
      var dx = this.momentum * gsumi[j] - this.learning_rate * gij;
      gsumi[j] = dx;
      return dx;
    }
    return (-this.learning_rate) * gij;
  }

  propagate(x, y, result) {
    let start = performance.now();
    this.net.forward(x, true);
    result.fwd_time = performance.now() - start;
    start = performance.now();
    result.cost_loss = this.net.backward(y);
    result.bwd_time = performance.now() - start;
  }

  initializeSums(pglist) {
    const needXsum = (this.methodName === 'adadelta' || this.methodName === 'adam');
    if (this.gsum.length === 0 && (this.methodName != 'sgd' || this.momentum > 0)) {
      for (var i = 0; i < pglist.length; i += 1) {
        this.gsum.push(Utils.zeros(pglist[i].params.length));
        if (needXsum) {
          this.xsum.push(Utils.zeros(pglist[i].params.length));
        }
      }
    }
  }

  iterate(pglist, method, result) {
    for (var i = 0; i < pglist.length; i += 1) {
      var pg = pglist[i];
      var params = pg.params;
      var gradients = pg.grads;
      var l1_decay = this.l1_decay * (pg.l1_decay_mul || 1);
      var l2_decay = this.l2_decay * (pg.l2_decay_mul || 1);
      var plen = params.length;
      for (var j = 0; j < plen; j += 1) {
        result.l1_decay_loss += l1_decay * Math.abs(params[j]);
        result.l2_decay_loss += l2_decay * params[j] * params[j] / 2;
        var gij = (l1_decay * (params[j] > 0 ? 1 : -1) + l2_decay * (params[j]) + gradients[j]) / this.batch_size;
        params[j] += method(i, j, gij);
        gradients[j] = 0;
      }
    }
  }

  train(x, y) {
    var result = { l1_decay_loss: 0, l2_decay_loss: 0 };
    this.propagate(x, y, result);
    var method = this.getMethod();
    this.iteration += 1;
    if (this.iteration % this.batch_size === 0) {
      var pglist = this.net.getParamsAndGrads();
      this.initializeSums(pglist);
      this.iterate(pglist, method, result);
    }
    result.loss = result.cost_loss + result.l1_decay_loss + result.l2_decay_loss;
    return result;
  }
}

module.exports = Trainer;
},{"./utils":24}],24:[function(require,module,exports){
/**
 * Static class with different transversal utility functions.
 */
class Utils {
  static gaussRandom() {
    if (Utils.gaussRandomReturnV) {
      Utils.gaussRandomReturnV = false;
      return Utils.gaussRandomVValue;
    }
    let u, v;
    let r = 0;
    while (r === 0 || r > 1) {
      u = (2 * Math.random()) - 1;
      v = (2 * Math.random()) - 1;
      r = (u * u) + (v * v);
    }
    const c = Math.sqrt((-2 * Math.log(r)) / r);
    Utils.gaussRandomVValue = v * c;
    Utils.gaussRandomReturnV = true;
    return u * c;
  }

  static randf(min, max) {
    return (Math.random() * (max - min)) + min;
  }

  static randi(min, max) {
    return Math.floor(Utils.randf(min, max));
  }

  static randn(mu, std) {
    return mu + (Utils.gaussRandom() * std);
  }

  static newArray(n, defaultValue = 0) {
    if (n === undefined || n === null || isNaN(n)) {
      return [];
    }
    const result = new Float64Array(n);
    if (defaultValue !== 0) {
      for (var i = 0; i < n; i += 1) {
        result[i] = defaultValue;
      }
    }
    return result;
  }

  static zeros(n) {
    return Utils.newArray(n);
  }

  static maxmin(arr) {
    if (!arr || !arr.length || arr.length === 0) {
      return {};
    }
    const result = { maxValue: arr[0], minValue: arr[0], maxIndex: 0, minIndex: 0 };
    const numElements = arr.length;
    for (var i = 0; i < numElements; i += 1) {
      if (arr[i] > result.maxValue) {
        result.maxValue = arr[i];
        result.maxIndex = i;
      } else if (arr[i] < result.minValue) {
        result.minValue = arr[i];
        result.minIndex = i;
      }
    }
    result.delta = result.maxValue - result.minValue;
    return result;
  }

  static maxValue(arr) {
    return Utils.maxmin(arr).maxValue;
  }

  static maxIndex(arr) {
    return Utils.maxmin(arr).maxIndex;
  }

  static minValue(arr) {
    return Utils.maxmin(arr).minValue;
  }

  static minIndex(arr) {
    return Utils.maxmin(arr).minIndex;
  }

  static posElement(arr, element) {
    var numElements = arr.length;
    for (var i = 0; i < numElements; i += 1) {
      if (arr[i] === element) {
        return i;
      }
    }
    return -1;
  }

  static arrayContains(arr, element) {
    return Utils.posElement(arr, element) > -1;
  }

  static arrayUnique(arr) {
    const result = [];
    const dict = {};
    const numElements = arr.length;
    for (var i = 0; i < numElements; i += 1) {
      if (dict[arr[i]] !== true) {
        result.push(arr[i]);
        dict[arr[i]] = true;
      }
    }
    return result;
  }

  static randomPermutation(n) {
    const result = [];
    for (var i = 0; i < n; i += 1) {
      result[i] = i;
    }
    let maxpos = n;
    while (maxpos--) {
      const currentpos = Math.floor(Math.random() * (maxpos + 1));
      const temp = result[maxpos];
      result[maxpos] = result[currentpos];
      result[currentpos] = temp;
    }
    return result;
  }

  static weightedSample(list, probabilities) {
    const p = Utils.randf(0, 1);
    let cumulative = 0;
    for (var i = 0; i < list.length; i += 1) {
      cumulative += probabilities[i];
      if (p < cumulative) {
        return list[i];
      }
    }
    return undefined;
  }
}

Utils.gaussRandomReturnV = false;
Utils.gaussRandomVValue = 0.0;

module.exports = Utils;

},{}],25:[function(require,module,exports){
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
    opts = opts | {};
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

},{"./serializable":22,"./utils":24}]},{},[1])(1)
});