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

  // More code will be added below
  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({name: 'Model Summary'}, model);
}

document.addEventListener('DOMContentLoaded', run);
