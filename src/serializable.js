class Serializable {
  clone() {
  }

  assign() {
  }

  toJSON() {
    return JSON.stringify(this.clone(true));
  }

  fromJSON(json) {
    this.assign(JSON.parse(json, true));    
  }

  getClassName() {
    return this.constructor.name;
  }
}

module.exports = Serializable;