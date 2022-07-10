import * as tf from '@tensorflow/tfjs';
import {Tensor1D} from "@tensorflow/tfjs";

(function () {
    let x: Tensor1D = tf.range(0, 12)

    let outText: string = '';
    outText += `\nrange(0,12):\n`
    outText += `${x}\n`
    outText += `\nshape:\n`
    outText += `${x.shape}\n`

    outText += `\nreshape([3,4]):\n`
    outText += `${x.reshape([3, 4])}\n`
    outText += `\nx.reshape([3, 4]).shape:\n`
    outText += `${x.reshape([3, 4]).shape}\n`

    outText += `\ntf.zeros([2, 3, 4]):\n`
    outText += `${tf.zeros([2, 3, 4])}\n`

    outText += `\ntf.ones([2, 3, 4]):\n`
    outText += `${tf.ones([2, 3, 4])}\n`

    outText += `\n${tf.randomNormal([3, 4])}:\n`
    outText += `${tf.randomNormal([3, 4])}\n`

    outText += `\ntf.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]):\n`
    outText += `${tf.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])}\n`

    outText += `\ntf.add([1.0, 2, 4, 8], [2.0, 2, 2, 2]):\n`
    outText += `${tf.add([1.0, 2, 4, 8], [2.0, 2, 2, 2])}\n`

    outText += `\ntf.sub([1.0, 2, 4, 8], [2.0, 2, 2, 2]):\n`
    outText += `${tf.sub([1.0, 2, 4, 8], [2.0, 2, 2, 2])}\n`

    outText += `\ntf.mul([1.0, 2, 4, 8], [2.0, 2, 2, 2]):\n`
    outText += `${tf.mul([1.0, 2, 4, 8], [2.0, 2, 2, 2])}\n`

    outText += `\ntf.div([1.0, 2, 4, 8], [2.0, 2, 2, 2]):\n`
    outText += `${tf.div([1.0, 2, 4, 8], [2.0, 2, 2, 2])}\n`

    outText += `\ntf.**([1.0, 2, 4, 8], [2.0, 2, 2, 2]):\n`
    outText += `${'???'}\n`

    outText += `\ntf.exp([1.0, 2, 4, 8]):\n`
    outText += `${tf.exp([1.0, 2, 4, 8])}\n`

    let X: Tensor1D = tf.range(0, 12, 1, 'float32').reshape([3, 4])
    let Y: Tensor1D = tf.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    outText += `\nlet X: Tensor1D = tf.range(0, 12, 1, 'float32').reshape([3, 4]):\n`
    outText += `${tf.range(0, 12, 1, 'float32').reshape([3, 4])}\n`
    outText += `\nlet Y: Tensor1D = tf.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]):\n`
    outText += `${tf.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])}\n`
    // outText += `\ntf.range(0, 12, 1, 'float32').reshape([3, 4]):\n`
    // outText += `${tf.range(0, 12, 1, 'float32').reshape([3, 4])}\n`

    tf.concat([X, Y], 0), tf.concat([X, Y], 1)

    let outEle = document.getElementById('ndarray')
    if (outEle !== null) {
        outEle.innerText = outText
    }
})();
