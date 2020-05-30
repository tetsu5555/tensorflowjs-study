console.log('Hello TensorFlow');

async function getData() {
  const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');  
  const carsData = await carsDataReq.json();  
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  }))
  .filter(car => (car.mpg != null && car.horsepower != null));
  
  return cleaned;
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential(); 
  
  // 1つのinput layer(dense layer)を追加
  // dense layerはmatrix(weights)でinputをかけてbiasと呼ばれる数値を結果に加えるlayer（よくわからん）
  // inputはhoursepower1つなので、inputShapeは[1]とする
  // useBiasはdefaultでtrueなため、指定する必要はないらしい
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
  
  // output layerを追加
  // 1つの値をoutputしてもらうため、unit: 1を指定する
  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: 'Horsepower v MPG'},
    {values}, 
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );

  /**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any 
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    // シャッフルする理由？
    // → データセットはbatchと呼ばれる小さなsubsetに分割され、それに基づいて学習される
    // シャッフルすることでデータを分割した時にそれぞれのbatchがいろんなデータを持つようにすることで以下のことを達成する
    // - 与えられたデータの順番に依存しない依存しない学習
    // 学習アルゴリズムに渡す前にデータをシャッフルするのがbest practice
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    // データを正規化するのはbest practiceらしい
    // Normalization is important because the internals of many machine learning models you will build with tensorflow.js are designed to work with numbers that are not too big.
    // Common ranges to normalize data to include 0 to 1 or -1 to 1.
    // You will have more success training your models if you get into the habit of normalizing your data to some reasonable range.
    // これどういうことだ？
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();  
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });  
}

  // More code will be added below
  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({name: 'Model Summary'}, model);
}

document.addEventListener('DOMContentLoaded', run);
