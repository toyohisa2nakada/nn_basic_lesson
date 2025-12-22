/*
学習のために分離したこのファイルは、github pagesを通して読み込まれる。
ただデバッグ用としてローカルから読み込む場合は、scriptタブを以下のにする。
<script src="https://localhost/iU/講義関連/システム設計演習/課題検討/ニューラルネットワークの計算方法/app/machine_learning_internal.js"></script>
*/

/*
参考URL
https://github.com/tensorflow/tfjs
https://js.tensorflow.org/api/latest/
https://js.tensorflow.org/api_vis/latest/
https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html?hl=ja#0
*/

/*
重みに関するメモ
const weights = model.getWeights();
2ニューロン、バイアスあり、出力1、バイアスなしの場合の重み
weights[0]: 中間層の重み 形状[1,2] 例[[0.1,-0.2]]
weights[1]: 中間層のバイアス 形状[2] 例[-2,1]
weights[2]: 出力層の重み 形状[2,1] 例[[1.2],[-0.1]]
*/

/*
モデルの概要をtfvisで描画する、のメモ
tfvis.show.modelSummary({ name: "Model Summary" }, model);
*/


async function getDataset(data) {
  if (data === undefined) {
    return {};
  }
  const series = {
    x: data.map((e) => e[0]),
    y: data.map((e) => e[1]),
  };
  return {
    values: data.map((e) => ({ x: e[0], y: e[1] })),
    ranges: {
      x: [Math.min(...series.x) - 0.5, Math.max(...series.x) + 0.5],
      y: [Math.min(...series.y) - 2.5, Math.max(...series.y) + 2.5],
    },
    tensors: {
      x: tf.tensor2d(series.x, [data.length, 1]),
      y: tf.tensor2d(series.y, [data.length, 1]),
    },
  };
}
function createModel({ units, useBias, LearningRate }) {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [1],
      units,
      useBias,
      activation: "tanh",
    }),
  );
  model.add(tf.layers.dense({ units: 1, useBias: false }));
  model.compile({
    optimizer: tf.train.adam(LearningRate),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"],
  });
  return model;
}
async function learn({ model, values, ranges, tensors, epochs }) {
  const history = await model.fit(tensors.x, tensors.y, {
    batchSize: tensors.x.shape[0],
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "学習回数と誤差(MSE)" },
      ["mse"],
      { height: 100, callbacks: ["onEpochEnd"] },
    ),
  });
  // console.log(model.layers[0].getWeights())
  // console.log(model.layers[1].getWeights())
  const weights = model.getWeights();
  // console.log(weights)
  updateLearnedParams({
    w1: model.layers[0].getWeights()[0].dataSync(),
    b: model.layers[0].getWeights()[1]?.dataSync(),
    w2: model.layers[1].getWeights()[0].dataSync(),
  });
  const predicted = await transform({
    model,
    range: ranges.x,
    interval: 0.1,
  });
  updateScatterplot({ values: [predicted, values], ranges });
}
function setupVisor({ onStart }) {
  const visorElement = document.querySelector(".visor");
  visorElement.style.width = "100%";
  visorElement.style.left = "0";

  [".visor-controls", ".visor-tabs"].forEach((e) => {
    const elem = visorElement.querySelector(e);
    elem.style.display = "none";
  });

  const learningPanelElem = tfvis.visor().surface({ name: "学習開始と学習された重み" });
  const button = document.createElement("button");
  button.textContent = "学習";
  button.id = "startLearning"
  button.addEventListener("click", async e => {
    button.disabled = true;
    await onStart();
    button.textContent = "追加学習";
    button.disabled = false;
  });
  learningPanelElem.container.appendChild(button);

  function createCompactNumberCss() {
    const style = document.createElement("style");
    style.textContent = `
      .compact-number {
        color: navy;
        margin: 0 4px;
        font-size: 0.7em;
        width: 5ch;
        box-sizing: border-box;
        border: none;
        outline: none;
        padding: 0;
        appearance: textfield;
      }
      .compact-number::-webkit-inner-spin-button,
      .compact-number::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
      }
    `;
    document.head.appendChild(style);
  }

  function createNeuronPanel(no) {
    const panel = document.createElement("span");
    panel.appendChild(Object.assign(document.createElement("span"), { textContent: `n${no + 1}`, }));
    panel.style.marginLeft = "5px";
    const wrap = document.createElement("span");
    panel.appendChild(wrap);
    wrap.style.border = "1px solid #ccc";
    wrap.style.marginLeft = "2px";

    function createElem(name) {
      const span = document.createElement("span");
      span.appendChild(Object.assign(document.createElement("span"), { textContent: name }));
      const text = document.createElement("input");
      span.appendChild(text);
      text.type = "number";
      text.readOnly = true;
      text.id = `learned_${no}_${name}`;
      text.classList.add("compact-number");
      return span;
    }
    ["w1", "b", "w2"].forEach(e => {
      wrap.appendChild(createElem(e));
    })
    return panel;
  }

  createCompactNumberCss();
  for (let i = 0; i < 2; i += 1) {
    learningPanelElem.container.appendChild(createNeuronPanel(i));
  }
}
function updateScatterplot({ values, ranges }) {
  tfvis.render.scatterplot(
    { name: "ニューラルネットワークの出力値と教師データ" },
    { values, series: ["出力値", "教師データ"] },
    { xAxisDomain: ranges.x, yAxisDomain: ranges.y, height: 200, width: 450 },
  );
}
async function transform({ model, range, interval }) {
  const x = [];
  for (let i = 0; range[0] + interval * i < range[1]; i += 1) {
    x.push(parseFloat((range[0] + interval * i).toFixed(10)));
  }
  const y = await model.predict(tf.tensor2d(x, [x.length, 1])).data();
  return x.map((_, i) => ({ x: x[i], y: y[i] }));
}

function updateLearnedParams(params) {
  console.log(params)
  for (let i = 0; i < params.w1.length; i += 1) {
    Object.keys(params).forEach((e) => {
      if (params[e] !== undefined) {
        document.querySelector(`#learned_${i}_${e}`).value = parseFloat(params[e][i].toFixed(3));
      }
    });
  }
}
