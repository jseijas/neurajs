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

class Net {
  constructor() {
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
      this.addLayer({ type: 'fc', num_neurons: def.num_classes || def.num_neurons });
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
}

module.exports = Net;