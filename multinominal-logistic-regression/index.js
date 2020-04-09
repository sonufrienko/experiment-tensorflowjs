const loadCSV = require('../utils/load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  converters: {
    mpg: (value) => {
      const mpg = parseFloat(value);

      if (mpg < 15) {
        return [1, 0, 0];
      } else if (mpg < 30) {
        return [0, 1, 0];
      } else {
        return [0, 0, 1];
      }
    }
  },
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['mpg'],
});

const regression = new LogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.5,
});

regression.train();

// test our accuracy
console.log(regression.test(testFeatures, _.flatMap(testLabels)));

// make our prediction
const predictionValue = regression
  .predict([
    [215, 440, 2.16], // horsepower', 'displacement', 'weight'
  ])
  .print();

// visualize a plot
plot({
  x: regression.costHistory.reverse(),
});
