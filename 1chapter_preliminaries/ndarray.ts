import * as tf from '@tensorflow/tfjs-node-gpu';
import {Tensor1D} from "@tensorflow/tfjs";

let x: Tensor1D = tf.range(12, 1)
console.log(x)
