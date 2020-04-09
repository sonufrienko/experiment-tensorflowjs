const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot')

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg'],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10,
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteraion #',
  yLabel: 'Mean Squared Error',
  title: 'MSE history',
  name: 'mse-history'
})

const predictionValue = regression.predict([
  [120, 2, 380] // horsepower', 'weight', 'displacement'
]).arraySync()[0][0];

console.log({
  prediction: predictionValue,
  r2
});