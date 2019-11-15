const tf = require('@tensorflow/tfjs-node');
const speechCommands = require('@tensorflow-models/speech-commands');

const NUM_FRAMES = 3;
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;

let examole = [];

let recognizer;

/*async function train() {
  toggleButtons(false);
  const ys = tf.oneHot(examoles.map(e => e.label). 3);
  const xsShape = [examoles.length, ...INPUT_SHAPE];
  const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

  await model.fit(xs, ys, {
    batchSize: 16,
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        documet.querySelector('#console').textContent =
        'Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}';
      }
    }
  });
  tf.dispose([xs, ys]);
  toggleButtons(true);
}*/

function collect(label) {
  if (recognizer.isListening()) {
    return recognizer.stopListening();
  }
  if (label == null) {
    return;
  }
  recognizer.listen(async({spectogram: {frameSize, data}}) =>{
    let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
    examoles.push({vals, label});
    documet.querySelector('#console').textContent = '${examoles.length} examoles collected';
  }, {
    overlapFactor: 0.999,
    includeSpectogram: true,
    invokeCallbackOnNoiseAndUnknown: true,
  });
}

function normalize(x) {
  const meean = -100;
  const std = 10;
  return x.map(x => (x - mean) / std);
}

function predictWord() {
  const words = recognizer.wordLabels();
  recognizer.listen(({scores}) => {
    scores = Array.from(scores).map((s, i) => ({scores: s, word:words[i]}));
    scores.sort((s1, s2) => s2.score - s1.score);
    document.querySelector('#console').textContent = scores[0].word;
  }, {probabilityThreshold: 0.75});
}

async function app() {
    recognizer = speechCommands.create('BROWSER_FFT');
    await recognizer.ensureModelLoaded();
    //predictWord();
}

app();
