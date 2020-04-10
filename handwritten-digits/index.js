// const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
// const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

const trainingData = mnist.training(0, 1000);
const testingData = mnist.testing(0, 100);

const features = trainingData.images.values.map((image) => _.flatMap(image));
const encodedLabels = trainingData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const regression = new LogisticRegression(features, encodedLabels, {
  learningRate: 1,
  iterations: 20,
  batchSize: 100,
});

regression.train();

const testFeatures = testingData.images.values.map((image) => _.flatMap(image));
const testEncodedLabels = testingData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log(`Accuracy = ${accuracy}`);
