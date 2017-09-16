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