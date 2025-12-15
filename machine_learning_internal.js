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
  button.addEventListener("click", onStart);
  learningPanelElem.container.appendChild(button);

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
      text.value = "";
      text.id = `learned_${no}_${name}`;
      text.style.width = "5em";
      text.style.border = "none";
      text.style.outline = "none";
      text.style.fontSize = "0.8em";
      text.style.marginLeft = "4px";
      text.style.padding = "0px";
      text.style.color = "navy";
      return span;
    }
    ["w1", "b", "w2"].forEach(e => {
      wrap.appendChild(createElem(e));
    })
    return panel;
  }

  for (let i = 0; i < 2; i += 1) {
    learningPanelElem.container.appendChild(createNeuronPanel(i));
  }
}
function updateScatterplot({ values, ranges }) {
  tfvis.render.scatterplot(
    { name: "ニューラルネットワークの出力値と教師データ" },
    { values, series: ["出力値", "教師データ"] },
    { xAxisDomain: ranges.x, yAxisDomain: ranges.y, height: 200, width: 550 },
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
  for (let i = 0; i < 2; i += 1) {
    Object.keys(params).forEach((e) => {
      document.querySelector(`#learned_${i}_${e}`).value = parseFloat(params[e][i].toFixed(3));
    });
  }
}
