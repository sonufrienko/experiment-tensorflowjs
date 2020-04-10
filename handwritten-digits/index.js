const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

const loadTrainingData = () => {
  const trainingData = mnist.training(0, 60000);
  const features = trainingData.images.values.map((image) => _.flatMap(image));
  const encodedLabels = trainingData.labels.values.map((label) => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
  });

  return { features, labels: encodedLabels }
}

const loadTestingData = () => {
  const testingData = mnist.testing(0, 10000);
  const testFeatures = testingData.images.values.map((image) => _.flatMap(image));
  const testEncodedLabels = testingData.labels.values.map((label) => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
  });

  return { testFeatures, testLabels: testEncodedLabels }
}

const { features, labels } = loadTrainingData();
const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 80,
  batchSize: 500,
});

regression.train();

const { testFeatures, testLabels } = loadTestingData();
const accuracy = regression.test(testFeatures, testLabels);
console.log(`Accuracy = ${accuracy}`);

plot({
  x: regression.costHistory.reverse()
})