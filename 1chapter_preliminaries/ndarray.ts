import * as tf from '@tensorflow/tfjs-node-gpu';
import {Tensor1D} from "@tensorflow/tfjs";

let x: Tensor1D = tf.range(0, 12)
x.print(true)
console.log(x.shape)
// console.log(tf.size(x))

let X: Tensor1D = x.reshape([3, 4])
X.print(true)

