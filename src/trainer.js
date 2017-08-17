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
      dx = this.momentum * gsumi[j] -this.learning_rate * gij;
      gsumi[j] = dx;
      return dx;
    }
    return -this.learning_rate * gij;
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