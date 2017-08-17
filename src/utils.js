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
