import * as tf from 'https://cdnjs.cloudflare.com/ajax/libs/tensorflow/1.0.1/tf.js';


TensorLinearModel= function(data)
{
	 
	 let model=tf.sequential();
	let hidden = tf.layers.dense({
    units: data.hidden_unit,
    inputShape:data.input_shape,
    activation: 'sigmoid'
     })
	let output = tf.layers.dense({
    units: 2,
    activation: 'sigmoid'
  });
  model.add(hidden);
  model.add(output);
  const LEARNING_RATE = 0.25;
  const optimizer = tf.train.sgd(LEARNING_RATE);

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return (model.summary())
  //return("yes");
}
