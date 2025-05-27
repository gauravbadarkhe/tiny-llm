const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");
const SaveEveryEpoch = require("./SaveEveryEpoch");

// CONFIG
const v = "word-llm-0.03";
const seqLength = 20;
const EPOCHS = 30;
const BATCH_SIZE = 128;

// STEP 1: Load and preprocess word list
const rawWords = fs.readFileSync("data/words.txt", "utf-8");
const cleanText = rawWords
  .split("\n")
  .filter((w) => w.length >= 3 && w.length <= 12)
  .join(" ")
  .toLowerCase();

console.log(`Loaded ${cleanText.length} characters`);

const chars = [...new Set(cleanText)];
const vocabSize = chars.length;
console.log("Vocab Size:", vocabSize);

const charToIdx = Object.fromEntries(chars.map((ch, i) => [ch, i]));
const idxToChar = Object.fromEntries(chars.map((ch, i) => [i, ch]));

const encoded = [...cleanText].map((ch) => charToIdx[ch]);

// STEP 2: Prepare training data
const inputs = [];
const labels = [];

for (let i = 0; i < encoded.length - seqLength; i++) {
  inputs.push(encoded.slice(i, i + seqLength));
  labels.push(encoded[i + seqLength]);
}

const xs = tf.tensor2d(inputs, [inputs.length, seqLength], "int32");
const ys = tf.tensor1d(new Float32Array(labels), "float32"); // float32 for loss

// STEP 3: Build model
const model = tf.sequential();
model.add(
  tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: 64,
    inputLength: seqLength,
  }),
);
model.add(tf.layers.lstm({ units: 128, returnSequences: true }));
model.add(tf.layers.lstm({ units: 128 }));
model.add(tf.layers.dense({ units: vocabSize, activation: "softmax" }));

model.compile({
  optimizer: tf.train.adam(0.001),
  loss: "sparseCategoricalCrossentropy",
  metrics: ["accuracy"],
});

model.summary();

// STEP 4: Train
async function train() {
  await model.fit(xs, ys, {
    epochs: 30,
    batchSize: 128,
    validationSplit: 0.1,
    callbacks: [
      tf.callbacks.earlyStopping({ monitor: "val_loss", patience: 3 }),
      SaveEveryEpoch(model, `model-${v}`),
    ],
  });
  await model.save(`file://model-${v}`);
  console.log("✅ Model saved");
}

// STEP 5: Generate words
function sampleWithTemperature(probabilities, temperature = 1.0) {
  const logits = tf.div(tf.log(probabilities), tf.scalar(temperature));
  const scaled = tf.softmax(logits).asType("float32");
  const idx = tf.multinomial(scaled, 1).dataSync()[0];
  return idx;
}

async function generateWords(seed, numWords = 20, temperature = 0.8) {
  const model = await tf.loadLayersModel(
    `file://model-word-llm-0.03/epoch-4/model.json`,
  );

  let inputSeq = [...seed.toLowerCase()].map((c) => charToIdx[c] || 0);
  if (inputSeq.length < seqLength) {
    inputSeq = Array(seqLength - inputSeq.length)
      .fill(0)
      .concat(inputSeq);
  } else {
    inputSeq = inputSeq.slice(-seqLength);
  }

  let currentWord = seed;
  const outputWords = [];

  for (let i = 0; i < numWords * 15; i++) {
    const inputTensor = tf.tensor2d([inputSeq], [1, seqLength]);
    const prediction = model.predict(inputTensor).squeeze();
    const nextCharIdx = sampleWithTemperature(prediction, temperature);
    const nextChar = idxToChar[nextCharIdx];

    if (nextChar === " ") {
      if (currentWord.length > 2 && currentWord.length < 12) {
        outputWords.push(currentWord);
        if (outputWords.length >= numWords) break;
      }
      currentWord = "";
    } else {
      currentWord += nextChar;
    }

    inputSeq.push(nextCharIdx);
    inputSeq.shift();
  }

  console.log("📚 Generated Words:\n", [...new Set(outputWords)].join(", "));
}

// train();
generateWords("test", 30, 0.7);
