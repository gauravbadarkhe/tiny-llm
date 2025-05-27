const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs-node");

function saveEveryEpoch(model, basePath) {
  return new tf.CustomCallback({
    onEpochEnd: async (epoch, logs) => {
      const epochDir = path.join(basePath, `epoch-${epoch + 1}`);
      const fullPath = `file://${epochDir}`;

      // ✅ Ensure parent directory exists
      if (!fs.existsSync(basePath)) {
        fs.mkdirSync(basePath, { recursive: true });
      }

      await model.save(fullPath);
      console.log(`📦 Saved model after epoch ${epoch + 1} -> ${fullPath}`);
    },
  });
}

module.exports = saveEveryEpoch;
