const loadCSV = require('../utils/load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  converters: {
    passedemissions: (value) => (value === 'TRUE' ? 1 : 0),
  },
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['passedemissions'],
});

console.log(labels);

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.5,
});

regression.train();

// test our accuracy
console.log(regression.test(testFeatures, testLabels));

// make our prediction
const predictionValue = regression
  .predict([
    [80, 97, 1.065], // horsepower', 'displacement', 'weight'
  ])
  .print();

// visualize a plot
plot({
  x: regression.costHistory.reverse(),
});
